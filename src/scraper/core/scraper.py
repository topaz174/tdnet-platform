import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import os
import json
import re
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.utils.init_db import Disclosure, DisclosureCategory, DisclosureSubcategory, DisclosureLabel, Base, engine
from src.classifier.rules.classifier import classify_disclosure_title, classify_and_store_labels


Session = sessionmaker(bind=engine)
session = Session()

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

def download_pdf(url, save_path):
    """
    Download PDF from URL and save to specified path.
    
    Args:
        url (str): URL of the PDF
        save_path (str): Path to save the PDF
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Skip if file already exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"PDF already exists: {save_path}")
            return True
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*"
        }
        
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        
        file_size = os.path.getsize(save_path)
        if file_size > 0:
            print(f"Successfully downloaded PDF: {file_size} bytes")
            return True
        else:
            print(f"Downloaded file is empty")
            return False
            
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return False

def download_xbrl(url, save_path):
    """
    Download XBRL from URL and save to specified path.
    
    Args:
        url (str): URL of the XBRL file
        save_path (str): Path to save the XBRL file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Skip if file already exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"XBRL already exists: {save_path}")
            return True
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/xbrl+xml,application/xml,*/*"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(save_path)
        if file_size > 0:
            print(f"Successfully downloaded XBRL: {file_size} bytes")
            return True
        else:
            print(f"Downloaded XBRL file is empty")
            os.remove(save_path)  # Remove empty file
            return False
            
    except Exception as e:
        print(f"Error downloading XBRL from {url}: {e}")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)  # Remove incomplete file
            except:
                pass
        return False

def get_date_range(days=30):
    """
    Get a range of dates from today back to the specified number of days.
    
    Args:
        days (int): Number of days to go back from today. Default is 30.
    
    Returns:
        list: List of date strings in format 'YYYY/MM/DD'
    """
    today = date.today()
    days_ago = today - timedelta(days=days)
    
    date_range = []
    current_date = today
    
    while current_date >= days_ago:
        date_str = current_date.strftime("%Y/%m/%d")
        date_range.append(date_str)
        current_date -= timedelta(days=1)
    
    return date_range

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
            
            # Now classify and store labels using the new system
            try:
                classify_and_store_labels(disclosure.id, disclosure.title)
                print(f"Saved to database and classified: {disclosure_data['company_code']} - {disclosure_data['title']}")
            except Exception as e:
                print(f"Warning: Failed to classify disclosure {disclosure.id}: {e}")
                # Don't fail the whole operation if classification fails
            
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

def get_date_range_between(start_date, end_date):
    """
    Get a range of dates between start_date and end_date (inclusive).
    
    Args:
        start_date (str): Start date in 'YYYY/MM/DD'
        end_date (str): End date in 'YYYY/MM/DD'
    
    Returns:
        list: List of date strings in format 'YYYY/MM/DD'
    """
    start = datetime.strptime(start_date, "%Y/%m/%d").date()
    end = datetime.strptime(end_date, "%Y/%m/%d").date()
    date_range = []
    current = end
    while current >= start:
        date_range.append(current.strftime("%Y/%m/%d"))
        current -= timedelta(days=1)
    return date_range

def scrape_tdnet_for_date(date_str, check_existing=True, min_time=None, max_time=None):
    """
    Scrape TDnet timely disclosure data for a given date.
    
    Args:
        date_str (str): Date in format 'YYYY/MM/DD'.
        check_existing (bool): Whether to check for existing disclosures in DB
        min_time (datetime.time): Only scrape disclosures after this time (inclusive)
        max_time (datetime.time): Only scrape disclosures before this time (inclusive)
    
    Returns:
        List of disclosure dictionaries
    """
    
    date_folder = date_str.replace('/', '-')
    
    
    # Load directories from config
    directories_file = project_root / 'config' / 'directories.json'
    with open(directories_file, 'r') as f:
        config = json.load(f)
    pdf_base_dir = config['pdf_directory']
    xbrl_base_dir = config['xbrls_directory']
    
    pdf_dir = os.path.join(pdf_base_dir, date_folder)
    xbrl_dir = os.path.join(xbrl_base_dir, date_folder)
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(xbrl_dir):
        os.makedirs(xbrl_dir)
    
    
    date_parts = date_str.split('/')
    date_for_url = f"{date_parts[0]}{date_parts[1]}{date_parts[2]}"
    
    
    base_url = "https://www.release.tdnet.info/inbs/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    
    all_disclosures = []
    added_to_db_count = 0
    
    
    page = 1
    
    while True:
        
        page_url = f"{base_url}I_list_{page:03d}_{date_for_url}.html"
        print(f"Scraping page {page} from: {page_url}")
        
        try:
            
            response = requests.get(page_url, headers=headers)
            
            
            if response.status_code == 404:
                print(f"Page {page} not found. Reached the end of disclosures for {date_str}.")
                break
                
            response.raise_for_status()
            
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            
            main_table = soup.find('table', id='main-list-table')
            
            if not isinstance(main_table, Tag):
                print(f"No disclosure table found on page {page} for {date_str}.")
                break
            
            
            rows = main_table.find_all('tr')
            
            if not rows:
                print(f"No disclosure rows found on page {page} for {date_str}.")
                break
            
            print(f"Found {len(rows)} disclosure rows on page {page} for {date_str}")
            
            # Get existing disclosures for this date to check against
            existing_disclosures = set()
            if check_existing:
                year, month, day = map(int, date_str.split('/'))
                disclosure_date = datetime(year, month, day).date()
                db_session = Session()
                try:
                    records = db_session.query(
                        Disclosure.company_code,
                        Disclosure.disclosure_date,
                        Disclosure.title
                    ).filter(
                        Disclosure.disclosure_date == disclosure_date
                    ).all()
                    
                    for record in records:
                        existing_disclosures.add((
                            record.company_code,
                            record.disclosure_date.isoformat(),
                            record.title
                        ))
                finally:
                    db_session.close()
            
            # Process each row
            for row in rows:
                if not isinstance(row, Tag):
                    continue
                
                cells = row.find_all('td')
                
                if len(cells) < 7:  
                    continue
                
                
                disclosure_time = cells[0].get_text(strip=True)
                company_code = cells[1].get_text(strip=True)
                company_name = cells[2].get_text(strip=True).strip()
                
                # Skip if not within time window
                if min_time is not None or max_time is not None:
                    hour, minute = map(int, disclosure_time.split(':'))
                    disclosure_time_obj = datetime.now().replace(hour=hour, minute=minute, second=0).time()
                    
                    # Check if disclosure time is within the specified window
                    if min_time is not None and disclosure_time_obj < min_time:
                        continue  # Skip this row
                    if max_time is not None and disclosure_time_obj > max_time:
                        continue  # Skip this row
                
                title_cell = cells[3]
                if not isinstance(title_cell, Tag):
                    continue
                link_tag = title_cell.find('a')
                
                if not isinstance(link_tag, Tag):
                    continue  
                
                title = link_tag.get_text(strip=True)
                
                # Parse date from the date_str
                year, month, day = map(int, date_str.split('/'))
                disclosure_date = datetime(year, month, day).date()
                
                # Check if this disclosure already exists in the database
                if check_existing:
                    key = (company_code, disclosure_date.isoformat(), title)
                    if key in existing_disclosures:
                        print(f"Skipping existing disclosure: {company_code} - {title}")
                        continue
                
                pdf_relative_url = link_tag.get('href')
                if not isinstance(pdf_relative_url, str):
                    continue
                pdf_url = f"{base_url}{pdf_relative_url}"
                
                exchange = cells[5].get_text(strip=True).strip()
                update_history = cells[6].get_text(strip=True).strip()
                if update_history == '　　　　　':  
                    update_history = None
                
                sanitized_title = sanitize_filename(title)
                safe_time = disclosure_time.replace(':', '-')

                pdf_filename = f"{safe_time}_{company_code}_{sanitized_title}.pdf"
                pdf_path = os.path.join(pdf_dir, pdf_filename)
                
                pdf_downloaded = download_pdf(pdf_url, pdf_path)
                
                xbrl_cell = cells[4]
                xbrl_link = None
                if isinstance(xbrl_cell, Tag):
                    xbrl_link = xbrl_cell.find('a', {'class': 'style002'})
                
                # Download XBRL if available
                xbrl_url = None
                xbrl_path = None
                has_xbrl = False
                
                if isinstance(xbrl_link, Tag):
                    xbrl_relative_url = xbrl_link.get('href')
                    if isinstance(xbrl_relative_url, str):
                        xbrl_url = f"{base_url}{xbrl_relative_url}"
                        
                    xbrl_filename = f"{safe_time}_{company_code}_{sanitized_title}.zip"
                    xbrl_file_path = os.path.join(xbrl_dir, xbrl_filename)
                    
                    if download_xbrl(xbrl_url, xbrl_file_path):
                        xbrl_path = xbrl_file_path
                        has_xbrl = True
                        print(f"Successfully downloaded XBRL for: {company_code} - {title}")
                    else:
                        print(f"Failed to download XBRL for: {company_code} - {title}")
                
                if pdf_downloaded:
                    hour, minute = map(int, disclosure_time.split(':'))
                    disclosure_time_obj = datetime.now().replace(hour=hour, minute=minute, second=0).time()
                    
                    disclosure_data = {
                        'disclosure_date': disclosure_date,
                        'time': disclosure_time_obj,
                        'company_code': company_code,
                        'company_name': company_name,
                        'title': title,
                        'xbrl_url': xbrl_url,  # Set the XBRL URL if available
                        'xbrl_path': xbrl_path,  # Set the downloaded XBRL path
                        'pdf_path': pdf_path,
                        'exchange': exchange,
                        'update_history': update_history,
                        'page_number': page,
                        'has_xbrl': has_xbrl  # Set based on successful XBRL download
                    }
                    
                    # Add to database immediately after downloading the PDF
                    if add_disclosure_to_db(disclosure_data):
                        added_to_db_count += 1
                    
                    all_disclosures.append(disclosure_data)
                    print(f"Added disclosure: {company_code} - {title}")
                    
                    
                    time.sleep(0.5)
            
            page += 1
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Page {page} not found. Reached the end of disclosures for {date_str}.")
                break
            else:
                print(f"HTTP error scraping page {page} for {date_str}: {e}")
                break
        except Exception as e:
            print(f"Error scraping page {page} for {date_str}: {e}")
            break
    
    print(f"Total disclosures added to database for {date_str}: {added_to_db_count}")
    return all_disclosures

def save_to_database(disclosures):
    """
    Save scraped disclosures to the database.
    
    Args:
        disclosures (list): List of disclosure dictionaries
        
    Returns:
        int: Number of new disclosures added to the database
    """
    if not disclosures:
        return 0
        
    # Since we're adding disclosures immediately in scrape_tdnet_for_date,
    # this function is now mostly for compatibility
    added_count = 0
    
    for disc_data in disclosures:
        if add_disclosure_to_db(disc_data):
            added_count += 1
    
    return added_count

def scrape_all_dates(days=30):
    """
    Scrape TDnet disclosures for multiple dates, from today back to the specified number of days.
    
    Args:
        days (int): Number of days to go back from today. Default is 30.
    
    Returns:
        int: Total number of disclosures added to the database
    """
    date_range = get_date_range(days)
    total_added = 0
    
    for date_str in date_range:
        print(f"\n{'='*80}")
        print(f"=== Starting to scrape TDnet disclosures for {date_str} ===")
        print(f"{'='*80}\n")
        
        disclosures = scrape_tdnet_for_date(date_str, check_existing=True)
        
        if disclosures:
            # The disclosures are already saved to the database in real-time
            print(f"Successfully scraped {len(disclosures)} disclosures for {date_str}.")
            total_added += len(disclosures)
        else:
            print(f"No disclosures found or error occurred for {date_str}.")
        
        
        print(f"Waiting 3 seconds before moving to the next date...")
        time.sleep(3)
    
    return total_added

def scrape_recent_disclosures():
    """
    Scrape only disclosures from the last 5 minutes
    """
    today_str = date.today().strftime("%Y/%m/%d")
    
    print(f"\n{'='*80}")
    print(f"=== Checking for disclosures from the last 5 minutes on {today_str} ===")
    print(f"{'='*80}\n")
    
    # Calculate time 5 minutes ago
    now = datetime.now()
    five_mins_ago = (now - timedelta(minutes=5)).time()
    
    disclosures = scrape_tdnet_for_date(today_str, check_existing=True, min_time=five_mins_ago)
    
    if disclosures:
        # The disclosures are already saved to the database in real-time
        print(f"Successfully scraped {len(disclosures)} recent disclosures for {today_str}.")
        return len(disclosures)
    else:
        print(f"No new disclosures found in the last 5 minutes for {today_str}.")
        return 0

def scrape_custom_date_range(start_date, end_date, start_time=None, end_time=None):
    """
    Scrape TDnet disclosures for a user-defined date range and optional time window.
    
    Args:
        start_date (str): Start date in 'YYYY/MM/DD'
        end_date (str): End date in 'YYYY/MM/DD'
        start_time (str): Optional start time in 'HH:MM' format
        end_time (str): Optional end time in 'HH:MM' format
    
    Returns:
        int: Total number of disclosures added to the database
    """
    date_range = get_date_range_between(start_date, end_date)
    total_added = 0
    
    # Parse time parameters if provided
    min_time_obj = None
    max_time_obj = None
    
    if start_time:
        try:
            hour, minute = map(int, start_time.split(':'))
            min_time_obj = datetime.now().replace(hour=hour, minute=minute, second=0).time()
        except ValueError:
            print(f"Invalid start time format: {start_time}. Expected HH:MM")
            return 0
    
    if end_time:
        try:
            hour, minute = map(int, end_time.split(':'))
            max_time_obj = datetime.now().replace(hour=hour, minute=minute, second=0).time()
        except ValueError:
            print(f"Invalid end time format: {end_time}. Expected HH:MM")
            return 0
    
    # Display time window info if applicable
    time_window_info = ""
    if start_time or end_time:
        time_window_info = f" (Time window: {start_time or '00:00'} - {end_time or '23:59'})"
    
    for date_str in date_range:
        print(f"\n{'='*80}")
        print(f"=== Starting to scrape TDnet disclosures for {date_str}{time_window_info} ===")
        print(f"{'='*80}\n")
        
        disclosures = scrape_tdnet_for_date(date_str, check_existing=True, min_time=min_time_obj, max_time=max_time_obj)
        
        if disclosures:
            added_count = save_to_database(disclosures)
            print(f"Successfully scraped {len(disclosures)} disclosures for {date_str}.")
            print(f"Added {added_count} new disclosures to the database.")
            total_added += added_count
        else:
            print(f"No disclosures found or error occurred for {date_str}.")
        
        print(f"Waiting 3 seconds before moving to the next date...")
        time.sleep(3)
    
    return total_added

if __name__ == "__main__":
    print("Starting TDnet disclosure scraper...")
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Run the full historical scrape if 'full' argument is provided
        print(f"Running full historical scrape (last 30 days)")
        total_added = scrape_all_dates()
        print(f"\n{'='*80}")
        print(f"=== Full scraping completed ===")
        print(f"Total disclosures added to the database: {total_added}")
        print(f"{'='*80}\n")
    elif len(sys.argv) > 2 and sys.argv[1] == "range":
        # Run custom date range scrape with optional time window
        # Formats supported:
        # python tdnet_scraper.py range YYYY/MM/DD YYYY/MM/DD
        # python tdnet_scraper.py range YYYY/MM/DD HH:MM YYYY/MM/DD HH:MM
        
        if len(sys.argv) == 4:
            # Date range only: python tdnet_scraper.py range YYYY/MM/DD YYYY/MM/DD
            start_date = sys.argv[2]
            end_date = sys.argv[3]
            print(f"Running custom date range scrape: {start_date} to {end_date}")
            total_added = scrape_custom_date_range(start_date, end_date)
        elif len(sys.argv) == 6:
            # Date range with time window: python tdnet_scraper.py range YYYY/MM/DD HH:MM YYYY/MM/DD HH:MM
            start_date = sys.argv[2]
            start_time = sys.argv[3]
            end_date = sys.argv[4]
            end_time = sys.argv[5]
            print(f"Running custom date range scrape with time window: {start_date} {start_time} to {end_date} {end_time}")
            total_added = scrape_custom_date_range(start_date, end_date, start_time, end_time)
        else:
            print("Invalid arguments for range command.")
            print("Usage:")
            print("  python tdnet_scraper.py range YYYY/MM/DD YYYY/MM/DD")
            print("  python tdnet_scraper.py range YYYY/MM/DD HH:MM YYYY/MM/DD HH:MM")
            sys.exit(1)
            
        print(f"\n{'='*80}")
        print(f"=== Custom range scraping completed ===")
        print(f"Total disclosures added to the database: {total_added}")
        print(f"{'='*80}\n")
    else:
        # Otherwise run the recent scrape (just last 5 minutes)
        print(f"Running recent scrape (last 5 minutes)")
        total_added = scrape_recent_disclosures()
        print(f"\n{'='*80}")
        print(f"=== Recent scraping completed ===")
        print(f"Total disclosures added to the database: {total_added}")
        print(f"{'='*80}\n") 