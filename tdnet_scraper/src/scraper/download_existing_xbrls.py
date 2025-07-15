import requests
import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.init_db import Disclosure, Base, engine

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

def download_existing_xbrls():
    """
    Download XBRL files for all existing disclosures that have xbrl_url but no xbrl_path.
    """
    # Load directories from config
    with open('directories.json', 'r') as f:
        config = json.load(f)
    xbrl_base_dir = config['xbrls_directory']
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Query disclosures that have xbrl_url but no xbrl_path
        disclosures = session.query(Disclosure).filter(
            Disclosure.xbrl_url.isnot(None),
            Disclosure.xbrl_path.is_(None)
        ).order_by(Disclosure.disclosure_date.desc()).all()
        
        print(f"Found {len(disclosures)} disclosures with XBRL URLs to download")
        
        downloaded_count = 0
        failed_count = 0
        
        for i, disclosure in enumerate(disclosures, 1):
            print(f"\n[{i}/{len(disclosures)}] Processing: {disclosure.company_code} - {disclosure.title}")
            
            # Create date folder
            date_folder = disclosure.disclosure_date.strftime("%Y-%m-%d")
            xbrl_dir = os.path.join(xbrl_base_dir, date_folder)
            
            # Create XBRL filename
            sanitized_title = sanitize_filename(disclosure.title)
            time_str = disclosure.time.strftime("%H-%M") if disclosure.time else "00-00"
            xbrl_filename = f"{time_str}_{disclosure.company_code}_{sanitized_title}.xbrl"
            xbrl_path = os.path.join(xbrl_dir, xbrl_filename)
            
            # Download XBRL
            if download_xbrl(disclosure.xbrl_url, xbrl_path):
                # Update database with xbrl_path
                try:
                    disclosure.xbrl_path = xbrl_path
                    session.commit()
                    print(f"Updated database with XBRL path: {xbrl_path}")
                    downloaded_count += 1
                except Exception as e:
                    print(f"Error updating database: {e}")
                    session.rollback()
                    failed_count += 1
            else:
                failed_count += 1
            
            # Small delay to be respectful to the server
            time.sleep(0.5)
            
            # Progress update every 50 items
            if i % 50 == 0:
                print(f"\nProgress: {i}/{len(disclosures)} processed, {downloaded_count} downloaded, {failed_count} failed")
        
        print(f"\n{'='*80}")
        print(f"XBRL Download Summary:")
        print(f"Total processed: {len(disclosures)}")
        print(f"Successfully downloaded: {downloaded_count}")
        print(f"Failed downloads: {failed_count}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during XBRL download process: {e}")
        session.rollback()
    finally:
        session.close()

def download_missing_xbrls_for_date_range(start_date, end_date):
    """
    Download XBRL files for disclosures in a specific date range that have xbrl_url but no xbrl_path.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    # Load directories from config
    with open('directories.json', 'r') as f:
        config = json.load(f)
    xbrl_base_dir = config['xbrls_directory']
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Parse dates
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Query disclosures in date range that have xbrl_url but no xbrl_path
        disclosures = session.query(Disclosure).filter(
            Disclosure.disclosure_date >= start_date_obj,
            Disclosure.disclosure_date <= end_date_obj,
            Disclosure.xbrl_url.isnot(None),
            Disclosure.xbrl_path.is_(None)
        ).order_by(Disclosure.disclosure_date.desc()).all()
        
        print(f"Found {len(disclosures)} disclosures with XBRL URLs to download between {start_date} and {end_date}")
        
        downloaded_count = 0
        failed_count = 0
        
        for i, disclosure in enumerate(disclosures, 1):
            print(f"\n[{i}/{len(disclosures)}] Processing: {disclosure.company_code} - {disclosure.title}")
            
            # Create date folder
            date_folder = disclosure.disclosure_date.strftime("%Y-%m-%d")
            xbrl_dir = os.path.join(xbrl_base_dir, date_folder)
            
            # Create XBRL filename
            sanitized_title = sanitize_filename(disclosure.title)
            time_str = disclosure.time.strftime("%H-%M") if disclosure.time else "00-00"
            xbrl_filename = f"{time_str}_{disclosure.company_code}_{sanitized_title}.xbrl"
            xbrl_path = os.path.join(xbrl_dir, xbrl_filename)
            
            # Download XBRL
            if download_xbrl(disclosure.xbrl_url, xbrl_path):
                # Update database with xbrl_path
                try:
                    disclosure.xbrl_path = xbrl_path
                    session.commit()
                    print(f"Updated database with XBRL path: {xbrl_path}")
                    downloaded_count += 1
                except Exception as e:
                    print(f"Error updating database: {e}")
                    session.rollback()
                    failed_count += 1
            else:
                failed_count += 1
            
            # Small delay to be respectful to the server
            time.sleep(0.5)
        
        print(f"\n{'='*80}")
        print(f"XBRL Download Summary for {start_date} to {end_date}:")
        print(f"Total processed: {len(disclosures)}")
        print(f"Successfully downloaded: {downloaded_count}")
        print(f"Failed downloads: {failed_count}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during XBRL download process: {e}")
        session.rollback()
    finally:
        session.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download XBRL files for existing disclosures")
    parser.add_argument('--range', nargs=2, metavar=('START_DATE', 'END_DATE'),
                       help='Download XBRLs for specific date range (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    if args.range:
        start_date, end_date = args.range
        print(f"Starting XBRL download for date range: {start_date} to {end_date}")
        download_missing_xbrls_for_date_range(start_date, end_date)
    else:
        print("Starting XBRL download for all existing disclosures...")
        download_existing_xbrls()

if __name__ == "__main__":
    main() 