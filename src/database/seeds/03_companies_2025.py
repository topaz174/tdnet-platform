import csv
import sys
from pathlib import Path
import psycopg2
from datetime import datetime

# Add project root to path (use this as is)
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

CSV_FILE_PATH = project_root / "config" / "data" / "japan_company_master_utf8.csv"

# Import unified config
from config.config import DB_URL

def parse_array_field(value):
    """Parse semicolon-separated values into PostgreSQL array format"""
    if not value or value.strip() == '':
        return None
    # Split by semicolon and clean up whitespace
    items = [item.strip() for item in value.split(';') if item.strip()]
    if not items:
        return None
    return items

def parse_date(date_str):
    """Parse date string, return None if empty or invalid"""
    if not date_str or date_str.strip() == '':
        return None
    # Handle special values that indicate no date
    if date_str.strip() in ['#N/A Field Not Applicable', '有', '無']:
        return None
    try:
        # Try to parse M/D/YYYY format
        return datetime.strptime(date_str.strip(), '%m/%d/%Y').date()
    except ValueError:
        print(f"Warning: Could not parse date '{date_str}', setting to NULL")
        return None

def parse_timestamp(ts_str):
    """Parse timestamp string, return None if empty or invalid"""
    if not ts_str or ts_str.strip() == '':
        return None
    try:
        # Try to parse M/D/YYYY H:MM format
        return datetime.strptime(ts_str.strip(), '%m/%d/%Y %H:%M')
    except ValueError:
        print(f"Warning: Could not parse timestamp '{ts_str}', setting to NULL")
        return None

def load_companies_from_csv():
    """Load companies data from CSV file into the database"""
    
    print(f"Loading companies from: {CSV_FILE_PATH}")
    
    # Connect to database
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    try:
        # Clear existing data
        print("Clearing existing companies data...")
        cursor.execute("TRUNCATE TABLE companies RESTART IDENTITY CASCADE")
        
        # Read and insert CSV data
        with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            insert_query = """
                INSERT INTO companies (
                    securities_code, ticker, company_name_japanese, company_name_english,
                    company_name_kana, sector_japanese, sector_english, company_address,
                    corporate_number, listing_classification, consolidation_yes_no,
                    listing_date, market_status, aliases, keywords, fiscal_year_end,
                    edinet_code, bloomberg_code, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            count = 0
            for row in reader:
                # Parse the data with proper type conversion
                data = (
                    row['securities_code'],
                    row['ticker'] or None,
                    row['company_name_japanese'],
                    row['company_name_english'] or None,
                    row['company_name_kana'] or None,
                    row['sector_japanese'] or None,
                    row['sector_english'] or None,
                    row['company_address'] or None,
                    row['corporate_number'] or None,
                    row['listing_classification'] or None,
                    row['consolidation_yes_no'] or None,
                    parse_date(row['listing_date']),
                    row['market_status'] or None,
                    parse_array_field(row['aliases']),
                    parse_array_field(row['keywords']),
                    row['fiscal_year_end'] or None,
                    row['EDINET_code'] or None,
                    row['Bloomberg_code'] or None,
                    parse_timestamp(row['created_at']),
                    parse_timestamp(row['updated_at'])
                )
                
                cursor.execute(insert_query, data)
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} companies...")
        
        # Commit the transaction
        conn.commit()
        print(f"Successfully loaded {count} companies into the database")
        
    except Exception as e:
        conn.rollback()
        print(f"Error loading companies: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_companies_from_csv()