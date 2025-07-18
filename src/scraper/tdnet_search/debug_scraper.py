import requests
from bs4 import BeautifulSoup
import os
import json
import re
import time as time_module
import sys
from datetime import datetime, timedelta, date, time
from dateutil.relativedelta import relativedelta
from urllib.parse import quote
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from classifier.rules.classifier import classify_disclosure_title
from scraper.tdnet_search.google_auth import get_authenticated_session

# Global variables
authenticated_session = None
debug_log_file = "debug_output.log"

def log_output(message, also_print=True):
    """Write message to both file and stdout."""
    if also_print:
        print(message)
    with open(debug_log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_or_create_authenticated_session():
    """Get or create the authenticated session for TDnet Search."""
    global authenticated_session
    
    if authenticated_session is None:
        log_output("Creating new authenticated session...")
        authenticated_session, _ = get_authenticated_session(headless=True)
        
        if authenticated_session is None:
            log_output("❌ Failed to create authenticated session")
            return None
        else:
            log_output("✓ Authenticated session created successfully")
    
    return authenticated_session

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

def print_disclosure_debug(disclosure_data):
    """
    Print disclosure information in a simple debug format.
    
    Args:
        disclosure_data (dict): Dictionary containing disclosure information
    """
    messages = [
        f"📄 {disclosure_data['disclosure_date']} {disclosure_data['time'].strftime('%H:%M')} | "
        f"{disclosure_data['company_code']} - {disclosure_data['company_name']}",
        f"   📋 {disclosure_data['title']}",
        f"   🏷️  Category: {disclosure_data.get('category', 'Unknown')} / {disclosure_data.get('subcategory', 'Unknown')}"
    ]
    
    if disclosure_data.get('pdf_url'):
        disclosure_id = extract_disclosure_id_from_url(disclosure_data['pdf_url'])
        messages.append(f"   📎 PDF ID: {disclosure_id}")
    
    if disclosure_data.get('xbrl_url'):
        messages.append(f"   💾 XBRL: Available")
    
    messages.append("")  # Empty line at end
    
    for msg in messages:
        log_output(msg)

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
            log_output(f"Error parsing date '{date_part}': {e}")
            return None
        
        # Parse time
        try:
            hour, minute = map(int, time_part.split(':'))
            disclosure_time = time(hour, minute)
        except ValueError as e:
            log_output(f"Error parsing time '{time_part}': {e}")
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
        log_output(f"Error parsing disclosure row: {e}")
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
        log_output(f"Error extracting financial metrics: {e}")
        return {}

def scrape_tdnet_search_page_debug(query, page=1):
    """
    Scrape a single page from TDnet Search for debugging (no DB operations or downloads).
    
    Args:
        query (str): Search query (e.g., "date<=2025-04-01")
        page (int): Page number to scrape
        
    Returns:
        tuple: (disclosures_list, last_disclosure_date)
    """
    base_url = "https://tdnet-search.appspot.com/search"
    
    # Create authenticated session for this scraping session
    authenticated_session = get_or_create_authenticated_session()
    
    # Construct URL with query and page
    encoded_query = quote(query)
    url = f"{base_url}?query={encoded_query}&page={page}"
    
    log_output(f"🔍 Scraping page {page}: {url}")
    
    try:
        response = authenticated_session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if we've reached the end (no more results)
        if "no information found" in response.text.lower() or "該当する情報はありません" in response.text:
            log_output(f"   ❌ Reached end of results on page {page}")
            return [], None
        
        # Find the main table with results
        table = soup.find('table')
        if not table:
            log_output(f"   ❌ No table found on page {page}")
            return [], None
        
        tbody = table.find('tbody')
        if not tbody:
            tbody = table  # Fallback if no tbody
        
        rows = tbody.find_all('tr')
        if not rows:
            log_output(f"   ❌ No rows found on page {page}")
            return [], None
        
        disclosures = []
        last_disclosure_date = None
        
        # Simple iteration through rows looking for disclosure rows (4 cells)
        for row_index, row in enumerate(rows):
            cells = row.find_all('td')
            
            # Check if this is a disclosure header row (4 cells)
            if len(cells) == 4:
                # Parse the disclosure data
                disclosure_data = parse_disclosure_row(row)
                if not disclosure_data:
                    continue
                
                # Update last_disclosure_date
                last_disclosure_date = disclosure_data['disclosure_date']

                # Look for the content row (next row should have colspan="4")
                content_row = None
                metrics_info = {}
                if row_index + 1 < len(rows):
                    potential_content = rows[row_index + 1]
                    content_cells = potential_content.find_all('td')
                    if len(content_cells) == 1 and content_cells[0].get('colspan') == '4':
                        content_row = potential_content
                        metrics_info = extract_financial_metrics(content_row)
                
                # Classify the disclosure title
                category, subcategory = classify_disclosure_title(disclosure_data['title'])
                
                final_disclosure_data = {
                    'disclosure_date': disclosure_data['disclosure_date'],
                    'time': disclosure_data['time'],
                    'company_code': disclosure_data['company_code'],
                    'company_name': disclosure_data['company_name'],
                    'title': disclosure_data['title'],
                    'pdf_url': disclosure_data['pdf_url'],
                    'xbrl_url': metrics_info.get('xbrl_url'),
                    'exchange': metrics_info.get('exchange', ''),
                    'update_history': metrics_info.get('update_history'),
                    'page_number': page,
                    'category': category,
                    'subcategory': subcategory
                }
                
                # Print disclosure details for debugging
                print_disclosure_debug(final_disclosure_data)
                disclosures.append(final_disclosure_data)
        
        log_output(f"   ✅ Page {page}: {len(disclosures)} disclosures found")
        return disclosures, last_disclosure_date
        
    except Exception as e:
        log_output(f"   ❌ Error scraping page {page}: {e}")
        return [], None

def scrape_ghost_disclosures_for_date_debug(target_date, base_filter=None, filter_depth=0, max_pages_per_query=11):
    """
    Handle ghost disclosures for a specific date by applying recursive filtering (debug version).
    
    Args:
        target_date (date): The specific date to scrape ghost disclosures for
        base_filter (str, optional): Base filter to apply (e.g., "pbr<1")
        filter_depth (int): Current recursion depth for filter subdivision
        max_pages_per_query (int): Maximum pages to scrape per query
        
    Returns:
        int: Total number of disclosures found for this date
    """
    target_date_str = target_date.strftime('%Y-%m-%d')
    total_found_for_date = 0
    
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
        log_output(f"\n👻 Ghost Disclosure Scraping for {target_date_str} with filter: {current_filter or 'none'} (depth {filter_depth})")
        
        # Construct query for exact date with filter
        if current_filter:
            query = f"date={target_date_str} AND {current_filter}"
        else:
            query = f"date={target_date_str}"
        
        page = 1
        filter_found = 0
        earliest_date_in_filter = None
        
        # Process pages for current filtered query
        while page <= max_pages_per_query:
            disclosures, last_date_on_page = scrape_tdnet_search_page_debug(query, page)
            
            filter_found += len(disclosures)
            if last_date_on_page:
                earliest_date_in_filter = last_date_on_page
            
            if not last_date_on_page:
                log_output(f"   🏁 Reached end of results for filtered query on page {page}")
                break
            
            page += 1
            if page <= max_pages_per_query:
                time_module.sleep(0.5)  # Shorter delay for debug
        
        total_found_for_date += filter_found
        log_output(f"   📊 Filter '{current_filter or 'none'}' completed: {filter_found} disclosures found")
        
        # Check if this filter still has ghost disclosures
        has_ghost_disclosures = (
            page > max_pages_per_query and 
            earliest_date_in_filter == target_date and 
            filter_depth < 3  # Limit recursion depth
        )
        
        if has_ghost_disclosures:
            log_output(f"   👻 More ghost disclosures detected in filter '{current_filter}', recursing to depth {filter_depth + 1}...")
            recursive_found = scrape_ghost_disclosures_for_date_debug(
                target_date, current_filter, filter_depth + 1, max_pages_per_query
            )
            total_found_for_date += recursive_found
        
        time_module.sleep(1)  # Brief pause between filters
    
    log_output(f"\n✅ Completed ghost disclosure scraping for {target_date_str}: {total_found_for_date} total disclosures found")
    return total_found_for_date

def _core_scrape_backwards_debug(newest_date_inclusive_str, oldest_date_inclusive_str=None, max_pages_per_query=11):
    """
    Core function that scrapes backwards from newest_date to oldest_date (debug version - no DB/downloads).
    
    Args:
        newest_date_inclusive_str (str): The newest date to start from in 'YYYY-MM-DD' format.
        oldest_date_inclusive_str (str, optional): The oldest date to stop at in 'YYYY-MM-DD' format. 
                                                   If None, scrapes indefinitely until no more data.
        max_pages_per_query (int): Maximum pages to scrape per query iteration.
        
    Returns:
        int: Total number of disclosures found
    """
    try:
        newest_target_date_obj = datetime.strptime(newest_date_inclusive_str, '%Y-%m-%d').date()
        if oldest_date_inclusive_str:
            oldest_target_date_obj = datetime.strptime(oldest_date_inclusive_str, '%Y-%m-%d').date()
            if oldest_target_date_obj > newest_target_date_obj:
                log_output(f"❌ Oldest date {oldest_date_inclusive_str} must be before or same as newest date {newest_date_inclusive_str}.")
                return 0
        else:
            oldest_target_date_obj = None
    except ValueError:
        log_output(f"❌ Error: Invalid date format. Please use YYYY-MM-DD.")
        return 0

    total_found_overall = 0
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
            log_output(f"\n🔄 Iteration {iteration}: Querying date <= {current_upper_bound_str} AND date >= {oldest_bound_str}")
        else:
            query = f"date<={current_upper_bound_str}"
            log_output(f"\n🔄 Iteration {iteration}: Querying date <= {current_upper_bound_str}")
        
        page = 1
        iteration_found = 0
        earliest_date_found_this_iteration = None
        
        # Process pages for current query
        while page <= max_pages_per_query:
            disclosures, last_date_on_page = scrape_tdnet_search_page_debug(query, page)
            
            iteration_found += len(disclosures)
            if last_date_on_page:
                earliest_date_found_this_iteration = last_date_on_page
            
            if not last_date_on_page:
                log_output(f"   🏁 Reached end of results for current query on page {page}")
                break
            
            page += 1
            if page <= max_pages_per_query:
                time_module.sleep(1)
        
        total_found_overall += iteration_found
        log_output(f"   📊 Iteration {iteration} completed: {iteration_found} disclosures found")
        
        # Check termination conditions
        if earliest_date_found_this_iteration is None:
            log_output(f"🏁 No disclosures found in iteration {iteration}. Ending scrape.")
            break
        
        # For range scraping: check if we've gone beyond the target oldest date
        if oldest_target_date_obj and earliest_date_found_this_iteration < oldest_target_date_obj:
            log_output(f"🏁 Earliest disclosure found ({earliest_date_found_this_iteration.strftime('%Y-%m-%d')}) is older than target ({oldest_target_date_obj.strftime('%Y-%m-%d')}). Ending scrape.")
            break
        
        # For unlimited historical scraping: check if no new items were found
        if not oldest_target_date_obj and iteration_found == 0:
            log_output("🏁 No more historical data available.")
            break
        
        # --- GHOST DISCLOSURE DETECTION AND HANDLING ---
        # Detect ghost disclosures: if we processed max pages and the earliest disclosure date 
        # equals the current query upper bound, there might be ghost disclosures beyond page 11
        has_ghost_disclosures = (
            page > max_pages_per_query and 
            earliest_date_found_this_iteration == current_query_upper_bound and
            iteration_found > 0  # Only if we found some disclosures in this iteration
        )
        
        if has_ghost_disclosures:
            log_output(f"👻 Ghost disclosures detected for {current_query_upper_bound.strftime('%Y-%m-%d')}!")
            log_output(f"   Last disclosure on page {max_pages_per_query} is from the same date as query upper bound.")
            log_output(f"   Starting filtered ghost disclosure scraping...")
            
            ghost_found = scrape_ghost_disclosures_for_date_debug(
                current_query_upper_bound, None, 0, max_pages_per_query
            )
            total_found_overall += ghost_found
            log_output(f"✅ Ghost disclosure scraping completed: {ghost_found} additional disclosures found.")
        
        # Set up next iteration query
        if iteration_found == 0:
            # If no disclosures were found, advance to dates strictly older than the earliest found
            log_output(f"   ⏭️  No disclosures found. Advancing query to be strictly older than {earliest_date_found_this_iteration.strftime('%Y-%m-%d')}")
            current_query_upper_bound = earliest_date_found_this_iteration - timedelta(days=1)
        else:
            # Standard advancement: continue from the earliest date found
            # If we handled ghost disclosures, we've exhausted the current date, so move to the previous day
            if has_ghost_disclosures:
                log_output(f"   ⏭️  Ghost disclosures handled for {current_query_upper_bound.strftime('%Y-%m-%d')}. Moving to previous day.")
                current_query_upper_bound = current_query_upper_bound - timedelta(days=1)
            else:
                current_query_upper_bound = earliest_date_found_this_iteration
        
        # For range scraping: check if next iteration would be outside target range
        if oldest_target_date_obj and current_query_upper_bound < oldest_target_date_obj:
            log_output(f"🏁 Next iteration's date ({current_query_upper_bound.strftime('%Y-%m-%d')}) would be older than target ({oldest_target_date_obj.strftime('%Y-%m-%d')}). Ending scrape.")
            break
        
        time_module.sleep(1)
    
    return total_found_overall

def debug_scrape_historical(start_date_str=None, max_pages_per_query=11):
    """
    Debug scrape historical data from TDnet Search (no DB/downloads).
    
    Args:
        start_date_str (str, optional): Starting date in 'YYYY-MM-DD' format. If None, defaults to one month ago.
        max_pages_per_query (int): Maximum pages to scrape per query before updating date
        
    Returns:
        int: Total number of disclosures found
    """
    if start_date_str is None:
        # Default to one month ago
        one_month_ago = date.today() - relativedelta(months=1)
        start_date_str = one_month_ago.strftime('%Y-%m-%d')
    
    log_output(f"\n{'='*80}")
    log_output(f"🐛 DEBUG: Starting TDnet Search scrape from {start_date_str} (backwards, unlimited)")
    log_output(f"{'='*80}")
    
    total_found = _core_scrape_backwards_debug(start_date_str, None, max_pages_per_query)
    
    log_output(f"\n{'='*80}")
    log_output(f"🐛 DEBUG: TDnet Search scrape completed")
    log_output(f"📊 Total disclosures found: {total_found}")
    log_output(f"{'='*80}")
    
    return total_found

def debug_scrape_date_range(oldest_date_inclusive_str, newest_date_inclusive_str, max_pages_per_query=11):
    """
    Debug scrape disclosures for a specific date range (no DB/downloads).

    Args:
        oldest_date_inclusive_str (str): The oldest date to include in 'YYYY-MM-DD' format.
        newest_date_inclusive_str (str): The newest date to include in 'YYYY-MM-DD' format.
        max_pages_per_query (int): Maximum pages to scrape per underlying query iteration.

    Returns:
        int: Total number of disclosures found
    """
    log_output(f"\n{'='*80}")
    log_output(f"🐛 DEBUG: Scraping TDnet Search for date range: {oldest_date_inclusive_str} to {newest_date_inclusive_str} (inclusive, backwards)")
    log_output(f"{'='*80}")

    total_found = _core_scrape_backwards_debug(newest_date_inclusive_str, oldest_date_inclusive_str, max_pages_per_query)
    
    log_output(f"\n{'='*80}")
    log_output(f"🐛 DEBUG: Date range scrape completed")
    log_output(f"📊 Total disclosures found: {total_found}")
    log_output(f"{'='*80}")
    
    return total_found

if __name__ == "__main__":
    # Clear previous log file
    with open(debug_log_file, "w", encoding="utf-8") as f:
        f.write("")  # Clear file
        
    log_output("🐛 Starting TDnet Search DEBUG scraper...")
    log_output("   📝 Note: This is a debug version - no database saving or file downloads!")
    
    try:
        total_found = 0
        
        if len(sys.argv) > 1 and sys.argv[1] == "historical":
            # Run historical debug scrape
            start_date = sys.argv[2] if len(sys.argv) > 2 else None
            total_found = debug_scrape_historical(start_date)
        elif len(sys.argv) > 3 and sys.argv[1] == "range":
            # Run specific date range debug scrape
            start_date = sys.argv[2]
            end_date = sys.argv[3]
            total_found = debug_scrape_date_range(start_date, end_date)
        else:
            # Default: run historical scrape from one month ago
            total_found = debug_scrape_historical()
        
        log_output(f"\n🎉 Debug scraping completed. Total disclosures found: {total_found}")
        
    except KeyboardInterrupt:
        log_output("\n⏹️  Debug scraping interrupted by user.")
    except Exception as e:
        log_output(f"\n❌ Error during debug scraping: {e}")
    finally:
        # Clean up authentication resources
        log_output("🧹 Cleaning up authentication resources...")
        authenticated_session = None
        log_output("✅ Cleanup completed")