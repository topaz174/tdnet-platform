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
import zipfile

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.database.init_db import Disclosure, engine  # Changed to tdnet database
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
            print("Failed to create authenticated session")
            return None
        else:
            print("Authenticated session created successfully")
    
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

def download_xbrl_file(url: str, zip_path: str, authenticated_session):
    """Download, extract, and clean-up an XBRL ZIP.

    Parameters
    ----------
    url : str
        Remote ZIP file URL.
    zip_path : str
        Local path where the ZIP should be temporarily stored.
        After extraction the *.zip is deleted.
    authenticated_session : requests.Session
        Pre-authenticated TDnet session.

    Returns
    -------
    str | None
        Path to the extracted directory if successful, otherwise *None*.
    """
    try:
        # If already extracted previously, skip.
        extract_dir = os.path.splitext(zip_path)[0]
        if os.path.isdir(extract_dir):
            print(f"Extracted XBRL already present: {extract_dir}")
            return extract_dir

        # Otherwise (re-)download.
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        sess = authenticated_session
        sess.headers.update({
            "Accept": "application/zip,*/*",
            "Referer": "https://tdnet-search.appspot.com/"
        })

        print(f"Downloading XBRL ZIP: {url}")
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get('content-type', '').lower()
        if 'zip' not in content_type and 'application/octet-stream' not in content_type:
            print(f"Unexpected content-type ({content_type}). Aborting.")
            return None

        with open(zip_path, 'wb') as fh:
            fh.write(resp.content)

        # Basic size sanity-check.
        if os.path.getsize(zip_path) < 512:  # <0.5 KB implies error page
            print("ZIP too small - likely an error page. Removing...")
            os.remove(zip_path)
            return None

        # Extract
        print(f"Extracting → {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        os.remove(zip_path)
        print("Extraction completed & ZIP removed")
        return extract_dir

    except Exception as e:
        print(f"Error downloading/extracting XBRL: {e}")
        # Ensure partial files are cleaned to avoid corrupt re-use
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

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
        
        date_time_text = cells[0].get_text(strip=True).replace('\xa0', ' ')
        
        if ' ' in date_time_text:
            parts = date_time_text.split(' ')
            date_part = parts[0]
            time_part = parts[1] if len(parts) >= 2 else "00:00"
        else:
            date_part = date_time_text
            time_part = "00:00"
        
        try:
            year, month, day = map(int, date_part.split('/'))
            disclosure_date = datetime(year, month, day).date()
        except ValueError as e:
            print(f"Error parsing date '{date_part}': {e}")
            return None
        
        try:
            hour, minute = map(int, time_part.split(':'))
            disclosure_time = time(hour, minute)
        except ValueError as e:
            print(f"Error parsing time '{time_part}': {e}")
            disclosure_time = time(0, 0)
        
        company_code = cells[1].get_text(strip=True)
        company_name = cells[2].get_text(strip=True)
        title_text = cells[3].get_text(strip=True)
        
        return {
            'disclosure_date': disclosure_date,
            'time': disclosure_time,
            'company_code': company_code,
            'company_name': company_name,
            'title': title_text,
        }
        
    except Exception as e:
        print(f"Error parsing disclosure row: {e}")
        return None

def extract_financial_metrics(content_row):
    """
    Extract XBRL URL and other information from the content row.
    """
    try:
        content_cell = content_row.find('td')
        if not content_cell: return {}
        
        metrics_div = content_cell.find('div', {'align': 'left'})
        if not metrics_div: return {}
        
        xbrl_url = None
        xbrl_link = metrics_div.find('a', href=lambda x: x and 'zip' in x.lower())
        if xbrl_link:
            xbrl_relative = xbrl_link.get('href')
            if xbrl_relative and not xbrl_relative.startswith('http'):
                xbrl_url = f"https://tdnet-search.appspot.com{xbrl_relative}"
        
        return { 'xbrl_url': xbrl_url }
        
    except Exception as e:
        print(f"Error extracting financial metrics: {e}")
        return {}

def scrape_tdnet_search_page(query, page=1, check_existing=True):
    """
    Scrape a single page from TDnet Search.
    """
    base_url = "https://tdnet-search.appspot.com/search"
    authenticated_session = get_or_create_authenticated_session()
    
    encoded_query = quote(query)
    url = f"{base_url}?query={encoded_query}&page={page}"
    
    print(f"Scraping page {page}: {url}")
    
    try:
        response = authenticated_session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        if "no information found" in response.text.lower() or "該当する情報はありません" in response.text:
            print(f"Reached end of results on page {page}")
            return [], None
        
        table = soup.find('table')
        if not table: return [], None
        
        tbody = table.find('tbody') or table
        rows = tbody.find_all('tr')
        if not rows: return [], None
        
        config = load_directory_config()
        xbrl_base_dir = config['xbrl_directory']
        
        existing_disclosures = set()
        if check_existing:
            db_session = Session()
            try:
                records = db_session.query(Disclosure.company_code, Disclosure.disclosure_date, Disclosure.title).all()
                existing_disclosures = {
                    (r.company_code, r.disclosure_date.isoformat(), r.title) for r in records
                }
            finally:
                db_session.close()
        
        disclosures = []
        last_disclosure_date = None
        
        for row_index, row in enumerate(rows):
            if len(row.find_all('td')) == 4:
                disclosure_data = parse_disclosure_row(row)
                if not disclosure_data: continue
                
                last_disclosure_date = disclosure_data['disclosure_date']

                key = (
                    disclosure_data['company_code'],
                    disclosure_data['disclosure_date'].isoformat(),
                    disclosure_data['title']
                )
                if check_existing and key in existing_disclosures:
                    print(f"Skipping existing disclosure: {key[0]} - {key[2]}")
                    continue
                
                metrics_info = {}
                if row_index + 1 < len(rows):
                    potential_content = rows[row_index + 1]
                    content_cells = potential_content.find_all('td')
                    if len(content_cells) == 1 and content_cells[0].get('colspan') == '4':
                        metrics_info = extract_financial_metrics(potential_content)
                
                date_folder = disclosure_data['disclosure_date'].strftime('%Y-%m-%d')
                xbrl_dir = os.path.join(xbrl_base_dir, date_folder)
                os.makedirs(xbrl_dir, exist_ok=True)
                
                sanitized_title = sanitize_filename(disclosure_data['title'])
                safe_time = disclosure_data['time'].strftime('%H-%M')
                
                extracted_path = None
                if metrics_info.get('xbrl_url'):
                    xbrl_filename = f"{safe_time}_{disclosure_data['company_code']}_{sanitized_title}.zip"
                    zip_full_path = os.path.join(xbrl_dir, xbrl_filename)

                    extracted_path = download_xbrl_file(
                        metrics_info['xbrl_url'], zip_full_path, authenticated_session
                    )

                xbrl_path = extracted_path  # may be None if failed
                
                category, subcategory = classify_disclosure_title(disclosure_data['title'])
                
                final_disclosure_data = {
                    'disclosure_date': disclosure_data['disclosure_date'],
                    'time': disclosure_data['time'],
                    'company_code': disclosure_data['company_code'],
                    'company_name': disclosure_data['company_name'],
                    'title': disclosure_data['title'],
                    'xbrl_url': metrics_info.get('xbrl_url'),
                    'xbrl_path': xbrl_path,
                    'pdf_path': None, # No PDF downloads
                    'exchange': '',
                    'update_history': None,
                    'page_number': page,
                    'category': category,
                    'subcategory': subcategory
                }
                
                if add_disclosure_to_db(final_disclosure_data):
                    disclosures.append(final_disclosure_data)
                
                time_module.sleep(0.5)
        
        return disclosures, last_disclosure_date
        
    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        return [], None

def load_directory_config():
    """Load directory configuration from JSON file."""
    try:
        with open('directories.json', 'r') as f:
            config = json.load(f)
        return {
            'xbrl_directory': config.get('xbrls_directory', 'xbrls')
        }
    except FileNotFoundError:
        return {
            'xbrl_directory': os.path.join(os.getcwd(), 'xbrls')
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

def scrape_company_earnings_historical(company_code, start_date_str=None, max_pages_per_query=11):
    """
    Scrape historical earnings reports for a single company.
    """
    if start_date_str is None:
        start_date_str = date.today().strftime('%Y-%m-%d')
    
    print_scraping_header(f"Starting historical earnings scrape for company {company_code} from {start_date_str}")
    
    try:
        newest_target_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    except ValueError:
        print(f"Error: Invalid date format for start_date. Please use YYYY-MM-DD.")
        return 0

    total_added_overall = 0
    current_query_upper_bound = newest_target_date_obj
    
    while True:
        current_upper_bound_str = current_query_upper_bound.strftime('%Y-%m-%d')
        query = f"code={company_code}0 AND title:決算短信 AND date<={current_upper_bound_str}"
        print(f"\n--- Querying for company {company_code}, earnings reports <= {current_upper_bound_str} ---")
        
        page = 1
        iteration_added = 0
        earliest_date_found_this_iteration = None
        
        while page <= max_pages_per_query:
            disclosures, last_date_on_page = scrape_tdnet_search_page(
                query, page, check_existing=True
            )
            
            iteration_added += len(disclosures)
            if last_date_on_page:
                earliest_date_found_this_iteration = last_date_on_page
            
            if not last_date_on_page:
                print(f"Reached end of results for this company on page {page}.")
                break
            
            page += 1
            if page <= max_pages_per_query:
                time_module.sleep(2)
        
        total_added_overall += iteration_added
        print(f"Iteration completed: {iteration_added} disclosures added.")
        
        if earliest_date_found_this_iteration is None or iteration_added == 0:
            print("No more historical earnings reports found for this company.")
            break
        
        # There's no risk of "ghost" disclosures with a company code filter,
        # so we can directly advance the date.
        current_query_upper_bound = earliest_date_found_this_iteration - timedelta(days=1)
        
        time_module.sleep(3)
    
    print_scraping_footer(total_added_overall)
    return total_added_overall

if __name__ == "__main__":
    print("Starting Company Earnings Historical Scraper...")
    
    company_code = input("Please enter the company code to scrape: ").strip()
    if not company_code.isdigit() or len(company_code) != 4:
        print("Invalid company code. Please enter a 4-digit code.")
        sys.exit(1)
        
    start_date = None
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
        print(f"Starting scrape from date: {start_date}")

    try:
        total_added = scrape_company_earnings_historical(company_code, start_date)
        print(f"Scraping completed. Total disclosures added: {total_added}")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Cleaning up authentication resources...")
        authenticated_session = None
        print("Cleanup completed") 