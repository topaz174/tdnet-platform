#!/usr/bin/env python3
"""
Reset script for TDnet Search scraper - clears database and PDF folder
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scraper.tdnet_search.init_db import Base, engine, Disclosure
from sqlalchemy.orm import sessionmaker

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
            'pdf_directory': 'pdfs_tdnet_search',
            'xbrl_directory': 'xbrls_tdnet_search'
        }

def reset_directory(directory_path, description, date_str=None):
    """Reset a directory by removing and recreating it.
    
    Args:
        directory_path (str): Path to directory
        description (str): Description for logging
        date_str (str, optional): Date string for single-day reset
    """
    try:
        if os.path.exists(directory_path):
            # Count files
            file_count = sum(len(files) for _, _, files in os.walk(directory_path))
            print(f"   Current {description} files: {file_count}")
            
            if file_count > 0:
                shutil.rmtree(directory_path)
                os.makedirs(directory_path, exist_ok=True)
                print(f"   ✓ {description} files deleted")
            else:
                print(f"   ✓ {description} directory already empty")
        else:
            os.makedirs(directory_path, exist_ok=True)
            print(f"   ✓ {description} directory created")
        return True
    except Exception as e:
        print(f"   ✗ Error resetting {description} directory: {e}")
        return False

def reset_database(session, target_date=None):
    """Reset database records.
    
    Args:
        session: SQLAlchemy session
        target_date (date, optional): Specific date to reset
    """
    try:
        if target_date:
            # Get current record count for the day
            record_count = session.query(Disclosure).filter(
                Disclosure.disclosure_date == target_date
            ).count()
            print(f"   Current records for {target_date}: {record_count}")
            
            if record_count > 0:
                # Delete records for the day
                session.query(Disclosure).filter(
                    Disclosure.disclosure_date == target_date
                ).delete()
                session.commit()
                print("   ✓ Disclosure records deleted")
            else:
                print("   ✓ No records found for this date")
        else:
            # Get current record count
            record_count = session.query(Disclosure).count()
            print(f"   Current records in database: {record_count}")
            
            if record_count > 0:
                # Delete all records
                session.query(Disclosure).delete()
                session.commit()
                print("   ✓ All disclosure records deleted")
            else:
                print("   ✓ Database already empty")
        return True
    except Exception as e:
        session.rollback()
        print(f"   ✗ Error resetting database: {e}")
        return False

def reset_single_day(target_date_str):
    """Reset TDnet Search data for a specific date.
    
    Args:
        target_date_str (str): Date in YYYY-MM-DD format
    """
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    except ValueError:
        print("❌ Invalid date format. Please use YYYY-MM-DD.")
        return False
    
    print("=" * 60)
    print(f"RESETTING TDNET SEARCH DATA FOR {target_date_str}")
    print("=" * 60)
    
    # Load configuration
    config = load_directory_config()
    
    # Create date-specific paths
    date_folder = target_date.strftime('%Y-%m-%d')
    pdf_date_dir = os.path.join(config['pdf_directory'], date_folder)
    xbrl_date_dir = os.path.join(config['xbrl_directory'], date_folder)
    
    # Reset database
    print(f"\n1. Resetting TDnet Search Database for {target_date_str}...")
    Session = sessionmaker(bind=engine)
    session = Session()
    success = reset_database(session, target_date)
    session.close()
    if not success:
        return False
    
    # Reset directories
    print(f"\n2. Resetting PDF Directory for {target_date_str}: {pdf_date_dir}")
    if not reset_directory(pdf_date_dir, "PDF"):
        return False
    
    print(f"\n3. Resetting XBRL Directory for {target_date_str}: {xbrl_date_dir}")
    if not reset_directory(xbrl_date_dir, "XBRL"):
        return False
    
    print("\n" + "=" * 60)
    print(f"✓ TDNET SEARCH RESET COMPLETED FOR {target_date_str}")
    print("=" * 60)
    
    return True

def reset_tdnet_search_data():
    """Reset all TDnet Search database and directories."""
    print("=" * 60)
    print("RESETTING ALL TDNET SEARCH DATA")
    print("=" * 60)
    
    # Load configuration
    config = load_directory_config()
    
    # Reset database
    print("\n1. Resetting TDnet Search Database...")
    Session = sessionmaker(bind=engine)
    session = Session()
    success = reset_database(session)
    session.close()
    if not success:
        return False
    
    # Reset directories
    print(f"\n2. Resetting PDF Directory: {config['pdf_directory']}")
    if not reset_directory(config['pdf_directory'], "PDF"):
        return False
    
    print(f"\n3. Resetting XBRL Directory: {config['xbrl_directory']}")
    if not reset_directory(config['xbrl_directory'], "XBRL"):
        return False
    
    print("\n" + "=" * 60)
    print("✓ TDNET SEARCH RESET COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "date":
            if len(sys.argv) > 2:
                date_str = sys.argv[2]
                print(f"This will DELETE ALL TDnet Search data for {date_str}!")
                response = input("\nAre you sure you want to continue? (y/n): ").lower().strip()
                if response in ['yes', 'y']:
                    success = reset_single_day(date_str)
                    if success:
                        print(f"\nReset completed successfully for {date_str}!")
                    else:
                        print(f"\nReset completed with some errors. Check the output above.")
                else:
                    print("Reset cancelled.")
            else:
                print("Please specify a date in YYYY-MM-DD format")
                sys.exit(1)
        else:
            print("This will DELETE ALL TDnet Search data (database records, PDF files, and XBRL files)!")
            print("This action cannot be undone.")
            response = input("\nAre you sure you want to continue? (y/n): ").lower().strip()
            if response in ['yes', 'y']:
                success = reset_tdnet_search_data()
                if success:
                    print("\nReset completed successfully!")
                else:
                    print("\nReset completed with some errors. Check the output above.")
            else:
                print("Reset cancelled.")
    else:
        print("Usage:")
        print("  Reset all data: python reset_tdnet_search.py")
        print("  Reset single date: python reset_tdnet_search.py date YYYY-MM-DD") 