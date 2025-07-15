import requests
from bs4 import BeautifulSoup
import os
import json
import re
import time as time_module
import sys
from datetime import datetime, timedelta, date, time
from dateutil.relativedelta import relativedelta
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scraper_tdnet_search.init_db_search import Disclosure, engine
from src.classifier.tdnet_classifier import classify_disclosure_title
from src.scraper_tdnet_search.google_auth import get_authenticated_session

Session = sessionmaker(bind=engine)

# Global authentication variables
authenticated_session = None

def get_or_create_authenticated_session():
    """Get or create the authenticated session for TDnet Search."""
    global authenticated_session
    
    if authenticated_session is None:
        print("Creating new authenticated session...")
        authenticated_session, _ = get_authenticated_session(headless=True)
        
        if authenticated_session is None:
            print("❌ Failed to create authenticated session")
            return None
        else:
            print("✓ Authenticated session created successfully")
    
    return authenticated_session

def sanitize_filename(filename):
    """
    Sanitize the filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    if len(sanitized) > 150:
        sanitized = sanitized[:150]
    return sanitized

def extract_disclosure_id_from_url(tdnet_url):
    """
    Extract disclosure ID from TDnet Search URL.
    
    Args:
        tdnet_url (str): TDnet Search URL (e.g., "https://tdnet-search.appspot.com/140120191227442453.pdf")
        
    Returns:
        str or None: Disclosure ID or None if not found
    """
    if not tdnet_url:
        return None
    
    # Extract filename from URL
    filename = tdnet_url.split('/')[-1]
    
    # Remove .pdf extension to get disclosure ID
    if filename.endswith('.pdf'):
        disclosure_id = filename[:-4]
        return disclosure_id
    
    return None

def construct_jpx_mirror_url(company_code, disclosure_id):
    """
    Construct JPX mirror URL for direct PDF access.
    
    Args:
        company_code (str): Company code
        disclosure_id (str): Disclosure ID extracted from TDnet URL
        
    Returns:
        str: JPX mirror URL
    """
    return f"https://www2.jpx.co.jp/disc/{company_code}/{disclosure_id}.pdf"

def download_file(url, save_path, file_type="PDF", authenticated_session=None):
    """
    Download file (PDF or XBRL) from URL and save to specified path.
    
    Args:
        url (str): URL of the file
        save_path (str): Path to save the file
        file_type (str): Type of file ("PDF" or "XBRL")
        authenticated_session (requests.Session): Authenticated session object
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Skip if file already exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"{file_type} already exists: {save_path}")
            return True
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Use authenticated session if provided, otherwise create simple session for JPX URLs
        if authenticated_session:
            session_obj = authenticated_session
        else:
            # For JPX mirror URLs, we don't need authentication - use simple session
            session_obj = requests.Session()
            session_obj.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            })
        
        # Set appropriate headers for file type
        if file_type.upper() == "PDF":
            session_obj.headers.update({
                "Accept": "application/pdf,*/*",
                "Referer": "https://tdnet-search.appspot.com/"
            })
        else:  # XBRL/ZIP
            session_obj.headers.update({
                "Accept": "application/zip,*/*",
                "Referer": "https://tdnet-search.appspot.com/"
            })
        
        print(f"Downloading {file_type} from: {url}")
        response = session_obj.get(url, timeout=30)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        expected_types = ['application/pdf', 'pdf'] if file_type.upper() == "PDF" else ['application/zip', 'zip', 'application/octet-stream']
        
        if not any(t in content_type for t in expected_types):
            print(f"Response is not a {file_type} (content-type: {content_type})")
            # Check if response content looks like an error page
            if len(response.content) < 1000 and b'error' in response.content.lower():
                print(f"Response appears to be an error page")
                return False
        
        # Save the file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(save_path)
        if file_size > 100:  # At least 100 bytes for a valid file
            print(f"Successfully downloaded {file_type}: {file_size} bytes")
            return True
        else:
            print(f"Downloaded file is too small (likely an error page): {file_size} bytes")
            os.remove(save_path)  # Remove the invalid file
            return False
            
    except Exception as e:
        print(f"Error downloading {file_type} from {url}: {e}")
        return False

def add_disclosure_to_db(disclosure_data):
    """
    Add a single disclosure to the database and commit immediately.
    
    Args:
        disclosure_data (dict): Dictionary containing disclosure information
        
    Returns:
        bool: True if successfully added, False otherwise
    """
    db_session = Session()
    try:
        existing = db_session.query(Disclosure).filter_by(
            company_code=disclosure_data['company_code'],
            disclosure_date=disclosure_data['disclosure_date'],
            title=disclosure_data['title']
        ).first()
        
        if not existing:
            disclosure = Disclosure(**disclosure_data)
            db_session.add(disclosure)
            db_session.commit()
            print(f"Saved to database: {disclosure_data['company_code']} - {disclosure_data['title']}")
            return True
        else:
            print(f"Skipping duplicate: {disclosure_data['company_code']} - {disclosure_data['title']}")
            return False
    except Exception as e:
        db_session.rollback()
        print(f"Error saving disclosure to database: {e}")
        return False
    finally:
        db_session.close()

def parse_disclosure_row(row_soup):
    """
    Parse a single disclosure row from the TDnet Search HTML.
    
    Args:
        row_soup: BeautifulSoup object representing a table row
        
    Returns:
        dict or None: Parsed disclosure data or None if parsing failed
    """
    try:
        cells = row_soup.find_all('td')
        if len(cells) < 4:
            return None
        
        # Extract date and time from first cell
        date_time_text = cells[0].get_text(strip=True)
        # The actual format has a non-breaking space (\xa0) that might be within the string
        # Replace any non-breaking spaces with regular spaces
        date_time_text = date_time_text.replace('\xa0', ' ')
        
        if ' ' in date_time_text:
            parts = date_time_text.split(' ')
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
            else:
                date_part = parts[0]
                time_part = "00:00"
        else:
            # Fallback if no time part
            date_part = date_time_text
            time_part = "00:00"
        
        # Parse date
        try:
            year, month, day = map(int, date_part.split('/'))
            disclosure_date = datetime(year, month, day).date()
        except ValueError as e:
            print(f"Error parsing date '{date_part}': {e}")
            return None
        
        # Parse time
        try:
            hour, minute = map(int, time_part.split(':'))
            disclosure_time = time(hour, minute)
        except ValueError as e:
            print(f"Error parsing time '{time_part}': {e}")
            disclosure_time = time(0, 0)  # Default to midnight
        
        # Extract company code from second cell
        code_link = cells[1].find('a')
        company_code = code_link.get_text(strip=True) if code_link else cells[1].get_text(strip=True)
        
        # Extract company name from third cell
        name_link = cells[2].find('a')
        company_name = name_link.get_text(strip=True) if name_link else cells[2].get_text(strip=True)
        
        # Extract title from fourth cell - title might not always have a link
        title_text = cells[3].get_text(strip=True)
        
        # Look for PDF URL - could be in title cell or we might need to construct it
        pdf_url = None
        title_link = cells[3].find('a')
        if title_link and title_link.get('href'):
            pdf_url = title_link.get('href')
            # Make sure PDF URL is absolute
            if pdf_url and not pdf_url.startswith('http'):
                pdf_url = f"https://tdnet-search.appspot.com{pdf_url}"
        
        return {
            'disclosure_date': disclosure_date,
            'time': disclosure_time,
            'company_code': company_code,
            'company_name': company_name,
            'title': title_text,
            'pdf_url': pdf_url
        }
        
    except Exception as e:
        print(f"Error parsing disclosure row: {e}")
        return None

def extract_financial_metrics(content_row):
    """
    Extract XBRL URL and other information from the content row.
    
    Args:
        content_row: BeautifulSoup object representing the content row
        
    Returns:
        dict: Dictionary with extracted information
    """
    try:
        content_cell = content_row.find('td')
        if not content_cell:
            return {}
        
        # Look for the metrics line (contains PER, PBR, DIVIDEND)
        metrics_div = content_cell.find('div', {'align': 'left'})
        if not metrics_div:
            return {}
        
        # Extract XBRL URL if present
        xbrl_url = None
        xbrl_link = metrics_div.find('a', href=lambda x: x and 'zip' in x.lower())
        if xbrl_link:
            xbrl_relative = xbrl_link.get('href')
            if xbrl_relative and not xbrl_relative.startswith('http'):
                xbrl_url = f"https://tdnet-search.appspot.com{xbrl_relative}"
        
        return {
            'xbrl_url': xbrl_url,
            'exchange': '',  # TDnet Search doesn't clearly separate this
            'update_history': None
        }
        
    except Exception as e:
        print(f"Error extracting financial metrics: {e}")
        return {}

def scrape_tdnet_search_page(query, page=1, check_existing=True):
    """
    Scrape a single page from TDnet Search.
    
    Args:
        query (str): Search query (e.g., "date<=2025-04-01")
        page (int): Page number to scrape
        check_existing (bool): Whether to check for existing disclosures in DB
        
    Returns:
        tuple: (disclosures_list, last_disclosure_date)
    """
    base_url = "https://tdnet-search.appspot.com/search"
    
    # Create authenticated session for this scraping session
    authenticated_session = get_or_create_authenticated_session()
    
    # Construct URL with query and page
    encoded_query = quote(query)
    url = f"{base_url}?query={encoded_query}&page={page}"
    
    print(f"Scraping page {page}: {url}")
    
    try:
        response = authenticated_session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if we've reached the end (no more results)
        if "no information found" in response.text.lower() or "該当する情報はありません" in response.text:
            print(f"Reached end of results on page {page}")
            return [], None
        
        # Find the main table with results
        table = soup.find('table')
        if not table:
            print(f"No table found on page {page}")
            return [], None
        
        tbody = table.find('tbody')
        if not tbody:
            tbody = table  # Fallback if no tbody
        
        rows = tbody.find_all('tr')
        if not rows:
            print(f"No rows found on page {page}")
            return [], None
        
        # Load directories from config
        config = load_directory_config()
        pdf_base_dir = config['pdf_directory']
        xbrl_base_dir = config['xbrl_directory']
        
        # Get existing disclosures for duplication check
        existing_disclosures = set()
        if check_existing:
            db_session = Session()
            try:
                records = db_session.query(
                    Disclosure.company_code,
                    Disclosure.disclosure_date,
                    Disclosure.title
                ).all()
                
                for record in records:
                    existing_disclosures.add((
                        record.company_code,
                        record.disclosure_date.isoformat(),
                        record.title
                    ))
            finally:
                db_session.close()
        
        disclosures = []
        last_disclosure_date = None
        
        # FIXED: Simple iteration through rows looking for disclosure rows (4 cells)
        for row_index, row in enumerate(rows):
            cells = row.find_all('td')
            
            # Check if this is a disclosure header row (4 cells)
            if len(cells) == 4:
                # Parse the disclosure data
                disclosure_data = parse_disclosure_row(row)
                if not disclosure_data:
                    continue
                
                # Update last_disclosure_date as soon as we have valid disclosure_data from any row on the page.
                # This ensures that even if all disclosures on this page are duplicates and skipped,
                # we still report the date of the last seen item for pagination purposes.
                last_disclosure_date = disclosure_data['disclosure_date']

                # Check for duplicate
                if check_existing:
                    key = (
                        disclosure_data['company_code'],
                        disclosure_data['disclosure_date'].isoformat(),
                        disclosure_data['title']
                    )
                    if key in existing_disclosures:
                        print(f"Skipping existing disclosure: {disclosure_data['company_code']} - {disclosure_data['title']}")
                        continue
                
                # Look for the content row (next row should have colspan="4")
                content_row = None
                metrics_info = {}
                if row_index + 1 < len(rows):
                    potential_content = rows[row_index + 1]
                    content_cells = potential_content.find_all('td')
                    if len(content_cells) == 1 and content_cells[0].get('colspan') == '4':
                        content_row = potential_content
                        metrics_info = extract_financial_metrics(content_row)
                
                # Create date folder for both PDF and XBRL
                date_folder = disclosure_data['disclosure_date'].strftime('%Y-%m-%d')
                pdf_dir = os.path.join(pdf_base_dir, date_folder)
                xbrl_dir = os.path.join(xbrl_base_dir, date_folder)
                
                for directory in [pdf_dir, xbrl_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                
                # Generate file names
                sanitized_title = sanitize_filename(disclosure_data['title'])
                safe_time = disclosure_data['time'].strftime('%H-%M')
                
                # Download PDF if URL is available
                pdf_path = None
                if disclosure_data['pdf_url']:
                    # Extract disclosure ID from TDnet URL and construct JPX mirror URL
                    disclosure_id = extract_disclosure_id_from_url(disclosure_data['pdf_url'])
                    if disclosure_id:
                        jpx_pdf_url = construct_jpx_mirror_url(disclosure_data['company_code'], disclosure_id)
                        
                        pdf_filename = f"{safe_time}_{disclosure_data['company_code']}_{sanitized_title}.pdf"
                        pdf_path = os.path.join(pdf_dir, pdf_filename)
                        
                        # Use regular requests session (no authentication needed for JPX)
                        pdf_downloaded = download_file(jpx_pdf_url, pdf_path, "PDF", None)
                        if not pdf_downloaded:
                            pdf_path = None
                    else:
                        print(f"Could not extract disclosure ID from URL: {disclosure_data['pdf_url']}")
                else:
                    print(f"No PDF URL found for {disclosure_data['company_code']} - {disclosure_data['title']}")
                
                # Download XBRL if URL is available
                xbrl_path = None
                if metrics_info.get('xbrl_url'):
                    xbrl_filename = f"{safe_time}_{disclosure_data['company_code']}_{sanitized_title}.zip"
                    xbrl_path = os.path.join(xbrl_dir, xbrl_filename)
                    xbrl_downloaded = download_file(metrics_info['xbrl_url'], xbrl_path, "XBRL", authenticated_session)
                    if not xbrl_downloaded:
                        xbrl_path = None
                
                # Classify the disclosure title
                category, subcategory = classify_disclosure_title(disclosure_data['title'])
                
                final_disclosure_data = {
                    'disclosure_date': disclosure_data['disclosure_date'],
                    'time': disclosure_data['time'],
                    'company_code': disclosure_data['company_code'],
                    'company_name': disclosure_data['company_name'],
                    'title': disclosure_data['title'],
                    'xbrl_path': xbrl_path,
                    'pdf_path': pdf_path,
                    'exchange': metrics_info.get('exchange', ''),
                    'update_history': metrics_info.get('update_history'),
                    'page_number': page,
                    'category': category,
                    'subcategory': subcategory
                }
                
                # Add to database (save disclosure metadata even without files)
                if add_disclosure_to_db(final_disclosure_data):
                    disclosures.append(final_disclosure_data)
                
                # Brief delay between processing
                time_module.sleep(0.5)
        
        return disclosures, last_disclosure_date
        
    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        return [], None

def scrape_ghost_disclosures_for_date(target_date, base_filter=None, filter_depth=0, max_pages_per_query=11):
    """
    Handle ghost disclosures for a specific date by applying recursive filtering.
    
    Args:
        target_date (date): The specific date to scrape ghost disclosures for
        base_filter (str, optional): Base filter to apply (e.g., "pbr<1")
        filter_depth (int): Current recursion depth for filter subdivision
        max_pages_per_query (int): Maximum pages to scrape per query
        
    Returns:
        int: Total number of new disclosures added for this date
    """
    target_date_str = target_date.strftime('%Y-%m-%d')
    total_added_for_date = 0
    
    # Define filter ranges for recursive subdivision
    if filter_depth == 0:
        # First level: pbr<1 and pbr>=1
        filters_to_try = ["pbr<1", "pbr>=1"]
    elif filter_depth == 1:
        # Second level: further subdivide based on base_filter
        if "pbr<1" in base_filter:
            filters_to_try = ["pbr<0.5", "pbr>=0.5 AND pbr<1"]
        elif "pbr>=1" in base_filter:
            filters_to_try = ["pbr>=1 AND pbr<2", "pbr>=2"]
        else:
            filters_to_try = [base_filter] if base_filter else [""]
    elif filter_depth == 2:
        # Third level: even more granular
        if "pbr<0.5" in base_filter:
            filters_to_try = ["pbr<0.25", "pbr>=0.25 AND pbr<0.5"]
        elif "pbr>=0.5 AND pbr<1" in base_filter:
            filters_to_try = ["pbr>=0.5 AND pbr<0.75", "pbr>=0.75 AND pbr<1"]
        elif "pbr>=1 AND pbr<2" in base_filter:
            filters_to_try = ["pbr>=1 AND pbr<1.5", "pbr>=1.5 AND pbr<2"]
        elif "pbr>=2" in base_filter:
            filters_to_try = ["pbr>=2 AND pbr<5", "pbr>=5"]
        else:
            filters_to_try = [base_filter] if base_filter else [""]
    else:
        # Maximum depth reached, just try the current filter
        filters_to_try = [base_filter] if base_filter else [""]
    
    for current_filter in filters_to_try:
        print(f"\n--- Ghost Disclosure Scraping for {target_date_str} with filter: {current_filter or 'none'} (depth {filter_depth}) ---")
        
        # Construct query for exact date with filter
        if current_filter:
            query = f"date={target_date_str} AND {current_filter}"
        else:
            query = f"date={target_date_str}"
        
        page = 1
        filter_added = 0
        earliest_date_in_filter = None
        latest_date_in_filter = None
        
        # Process pages for current filtered query
        while page <= max_pages_per_query:
            disclosures, last_date_on_page = scrape_tdnet_search_page(
                query, page, check_existing=True
            )
            
            filter_added += len(disclosures)
            if last_date_on_page:
                earliest_date_in_filter = last_date_on_page
                if latest_date_in_filter is None:
                    latest_date_in_filter = last_date_on_page
            
            if not last_date_on_page:
                print(f"Reached end of results for filtered query on page {page}.")
                break
            
            page += 1
            if page <= max_pages_per_query:
                time_module.sleep(1)  # Shorter delay for ghost disclosure queries
        
        total_added_for_date += filter_added
        print(f"Filter '{current_filter or 'none'}' completed: {filter_added} disclosures added.")
        
        # Check if this filter still has ghost disclosures (reached max pages and last disclosure is target date)
        has_ghost_disclosures = (
            page > max_pages_per_query and 
            earliest_date_in_filter == target_date and 
            filter_depth < 3  # Limit recursion depth
        )
        
        if has_ghost_disclosures:
            print(f"Ghost disclosures detected in filter '{current_filter}', recursing to depth {filter_depth + 1}...")
            recursive_added = scrape_ghost_disclosures_for_date(
                target_date, current_filter, filter_depth + 1, max_pages_per_query
            )
            total_added_for_date += recursive_added
        
        time_module.sleep(2)  # Brief pause between filters
    
    print(f"Completed ghost disclosure scraping for {target_date_str}: {total_added_for_date} total disclosures added.")
    return total_added_for_date

def load_directory_config():
    """Load directory configuration from JSON file."""
    try:
        with open('directories.json', 'r') as f:
            config = json.load(f)
        return {
            'pdf_directory': config.get('tdnet_search_pdf_directory', 'pdfs_tdnet_search'),
            'xbrl_directory': config.get('tdnet_search_xbrl_directory', 'xbrls_tdnet_search')
        }
    except FileNotFoundError:
        return {
            'pdf_directory': os.path.join(os.getcwd(), 'pdfs_tdnet_search'),
            'xbrl_directory': os.path.join(os.getcwd(), 'xbrls_tdnet_search')
        }

def print_scraping_header(message):
    """Print a standardized scraping header."""
    print(f"\n{'='*80}")
    print(f"=== {message} ===")
    print(f"{'='*80}\n")

def print_scraping_footer(total_added):
    """Print a standardized scraping footer."""
    print(f"\n{'='*80}")
    print(f"=== Scraping completed ===")
    print(f"Total disclosures added to database: {total_added}")
    print(f"{'='*80}\n")

def scrape_single_day(target_date_str, max_pages_per_query=11):
    """
    Scrape disclosures for a single specific date.

    Args:
        target_date_str (str): The date to scrape in 'YYYY-MM-DD' format.
        max_pages_per_query (int): Maximum pages to scrape per query iteration.

    Returns:
        int: Total number of disclosures added to the database
    """
    print_scraping_header(f"Scraping TDnet Search for date: {target_date_str}")
    total_added = _core_scrape_backwards(target_date_str, target_date_str, max_pages_per_query)
    print_scraping_footer(total_added)
    return total_added

def scrape_tdnet_search_historical(start_date_str=None, max_pages_per_query=11):
    """
    Scrape historical data from TDnet Search, working backwards from start_date indefinitely.
    
    Args:
        start_date_str (str, optional): Starting date in 'YYYY-MM-DD' format. If None, defaults to one month ago.
        max_pages_per_query (int): Maximum pages to scrape per query before updating date
        
    Returns:
        int: Total number of disclosures added to the database
    """
    if start_date_str is None:
        # Default to one month ago
        one_month_ago = date.today() - relativedelta(months=1)
        start_date_str = one_month_ago.strftime('%Y-%m-%d')
    
    print_scraping_header(f"Starting TDnet Search historical scrape from {start_date_str} (backwards, unlimited)")
    total_added = _core_scrape_backwards(start_date_str, None, max_pages_per_query)
    print_scraping_footer(total_added)
    return total_added

def scrape_specific_date_range(oldest_date_inclusive_str, newest_date_inclusive_str, max_pages_per_query=11):
    """
    Scrape disclosures for a specific date range (inclusive), working backwards from the newest date.

    Args:
        oldest_date_inclusive_str (str): The oldest date to include in 'YYYY-MM-DD' format.
        newest_date_inclusive_str (str): The newest date to include in 'YYYY-MM-DD' format.
        max_pages_per_query (int): Maximum pages to scrape per underlying query iteration.

    Returns:
        int: Total number of disclosures added to the database
    """
    print_scraping_header(f"Scraping TDnet Search for date range: {oldest_date_inclusive_str} to {newest_date_inclusive_str} (inclusive, backwards)")
    total_added = _core_scrape_backwards(newest_date_inclusive_str, oldest_date_inclusive_str, max_pages_per_query)
    print_scraping_footer(total_added)
    return total_added

def _core_scrape_backwards(newest_date_inclusive_str, oldest_date_inclusive_str=None, max_pages_per_query=11):
    """
    Core function that scrapes backwards from newest_date to oldest_date (or indefinitely if oldest_date is None).
    
    Args:
        newest_date_inclusive_str (str): The newest date to start from in 'YYYY-MM-DD' format.
        oldest_date_inclusive_str (str, optional): The oldest date to stop at in 'YYYY-MM-DD' format. 
                                                   If None, scrapes indefinitely until no more data.
        max_pages_per_query (int): Maximum pages to scrape per query iteration.
        
    Returns:
        int: Total number of disclosures added to the database
    """
    try:
        newest_target_date_obj = datetime.strptime(newest_date_inclusive_str, '%Y-%m-%d').date()
        if oldest_date_inclusive_str:
            oldest_target_date_obj = datetime.strptime(oldest_date_inclusive_str, '%Y-%m-%d').date()
            if oldest_target_date_obj > newest_target_date_obj:
                print(f"Oldest date {oldest_date_inclusive_str} must be before or same as newest date {newest_date_inclusive_str}.")
                return 0
        else:
            oldest_target_date_obj = None
    except ValueError:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD.")
        return 0

    total_added_overall = 0
    iteration = 0
    
    # Start with newest date as current upper bound (inclusive)
    current_query_upper_bound = newest_target_date_obj
    
    # Main scraping loop
    while True:
        iteration += 1
        current_upper_bound_str = current_query_upper_bound.strftime('%Y-%m-%d')
        
        # Construct query
        if oldest_target_date_obj:
            oldest_bound_str = oldest_target_date_obj.strftime('%Y-%m-%d')
            query = f"date<={current_upper_bound_str} AND date>={oldest_bound_str}"
            print(f"\n--- Iteration {iteration}: Querying date <= {current_upper_bound_str} AND date >= {oldest_bound_str} ---")
        else:
            query = f"date<={current_upper_bound_str}"
            print(f"\n--- Iteration {iteration}: Querying date <= {current_upper_bound_str} ---")
        
        page = 1
        iteration_added = 0
        earliest_date_found_this_iteration = None
        
        # Process pages for current query
        while page <= max_pages_per_query:
            disclosures, last_date_on_page = scrape_tdnet_search_page(
                query, page, check_existing=True
            )
            
            iteration_added += len(disclosures)
            if last_date_on_page:
                earliest_date_found_this_iteration = last_date_on_page
            
            if not last_date_on_page:
                print(f"Reached end of results for current query on page {page}.")
                break
            
            page += 1
            if page <= max_pages_per_query:
                time_module.sleep(2)
        
        total_added_overall += iteration_added
        print(f"Iteration {iteration} completed: {iteration_added} disclosures added.")
        
        # Check termination conditions
        if earliest_date_found_this_iteration is None:
            print(f"No disclosures found in iteration {iteration}. Ending scrape.")
            break
        
        # For range scraping: check if we've gone beyond the target oldest date
        if oldest_target_date_obj and earliest_date_found_this_iteration < oldest_target_date_obj:
            print(f"Earliest disclosure found ({earliest_date_found_this_iteration.strftime('%Y-%m-%d')}) is older than target ({oldest_target_date_obj.strftime('%Y-%m-%d')}). Ending scrape.")
            break
        
        # For unlimited historical scraping: check if no new items were added
        if not oldest_target_date_obj and iteration_added == 0:
            print("No more historical data available.")
            break
        
        # --- GHOST DISCLOSURE DETECTION AND HANDLING ---
        # Detect ghost disclosures: if we processed max pages and the earliest disclosure date 
        # equals the current query upper bound, there might be ghost disclosures beyond page 11
        has_ghost_disclosures = (
            page > max_pages_per_query and 
            earliest_date_found_this_iteration == current_query_upper_bound
        )
        
        if has_ghost_disclosures:
            print(f"🔍 Ghost disclosures detected for {current_query_upper_bound.strftime('%Y-%m-%d')}!")
            print(f"   Last disclosure on page {max_pages_per_query} is from the same date as query upper bound.")
            print(f"   Starting filtered ghost disclosure scraping...")
            
            ghost_added = scrape_ghost_disclosures_for_date(
                current_query_upper_bound, None, 0, max_pages_per_query
            )
            total_added_overall += ghost_added
            print(f"✓ Ghost disclosure scraping completed: {ghost_added} additional disclosures added.")
        
        # Set up next iteration query
        if iteration_added == 0 and not has_ghost_disclosures:
            # If no new disclosures were added AND no ghost disclosures detected, advance to dates strictly older than the earliest found
            print(f"No new disclosures added. Advancing query to be strictly older than {earliest_date_found_this_iteration.strftime('%Y-%m-%d')}.")
            current_query_upper_bound = earliest_date_found_this_iteration - timedelta(days=1)
        else:
            # Standard advancement: continue from the earliest date found
            # If we handled ghost disclosures, we've exhausted the current date, so move to the previous day
            if has_ghost_disclosures:
                print(f"Ghost disclosures handled for {current_query_upper_bound.strftime('%Y-%m-%d')}. Moving to previous day.")
                current_query_upper_bound = current_query_upper_bound - timedelta(days=1)
            else:
                current_query_upper_bound = earliest_date_found_this_iteration
        
        # For range scraping: check if next iteration would be outside target range
        if oldest_target_date_obj and current_query_upper_bound < oldest_target_date_obj:
            print(f"Next iteration's date ({current_query_upper_bound.strftime('%Y-%m-%d')}) would be older than target ({oldest_target_date_obj.strftime('%Y-%m-%d')}). Ending scrape.")
            break
        
        time_module.sleep(3)
    
    return total_added_overall

if __name__ == "__main__":
    print("Starting TDnet Search disclosure scraper...")
    
    try:
        total_added = 0
        
        if len(sys.argv) > 1:
            if sys.argv[1] == "historical":
                # Run historical scrape
                start_date = sys.argv[2] if len(sys.argv) > 2 else None
                total_added = scrape_tdnet_search_historical(start_date)
            elif sys.argv[1] == "range":
                # Run specific date range scrape
                if len(sys.argv) > 3:
                    start_date = sys.argv[2]
                    end_date = sys.argv[3]
                    total_added = scrape_specific_date_range(start_date, end_date)
                else:
                    print("Please provide both start and end dates for range scraping")
                    sys.exit(1)
            elif sys.argv[1] == "date":
                # Run single date scrape
                if len(sys.argv) > 2:
                    target_date = sys.argv[2]
                    total_added = scrape_single_day(target_date)
                else:
                    print("Please provide a date in YYYY-MM-DD format")
                    sys.exit(1)
            else:
                print("Usage:")
                print("  Historical scrape: python tdnet_search_scraper.py historical [YYYY-MM-DD]")
                print("  Date range scrape: python tdnet_search_scraper.py range YYYY-MM-DD YYYY-MM-DD")
                print("  Single date scrape: python tdnet_search_scraper.py date YYYY-MM-DD")
                sys.exit(1)
        else:
            # Default: run historical scrape from one month ago
            total_added = scrape_tdnet_search_historical()
        
        print(f"Scraping completed. Total disclosures added: {total_added}")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        # Clean up authentication resources
        print("Cleaning up authentication resources...")
        authenticated_session = None
        print("✓ Cleanup completed") 