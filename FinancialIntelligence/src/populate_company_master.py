#!/usr/bin/env python3
"""
Script to populate company_master table with data from japan_company_master_utf8.csv
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
from datetime import datetime
import sys

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'tdnet'),
            user=os.getenv('DB_USER', 'claudiu'),
            password=os.getenv('DB_PASSWORD', ''),
            port=os.getenv('DB_PORT', '5432')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def clean_data(df):
    """Clean and prepare data for database insertion"""
    print("Cleaning and preparing data...")
    
    # Check for and handle duplicates in securities_code
    if 'securities_code' in df.columns:
        initial_count = len(df)
        duplicates = df[df.duplicated(subset=['securities_code'], keep=False)]
        
        if len(duplicates) > 0:
            print(f"Found {len(duplicates)} duplicate securities_code entries:")
            print(duplicates[['securities_code', 'company_name_japanese']].to_string())
            
            # Keep the last occurrence of each duplicate (most recent data)
            df = df.drop_duplicates(subset=['securities_code'], keep='last')
            print(f"Removed {initial_count - len(df)} duplicate records, keeping the last occurrence of each")
    
    # Fix column name casing to match database schema
    column_mapping = {
        'EDINET_code': 'edinet_code',
        'Bloomberg_code': 'bloomberg_code'
    }
    df = df.rename(columns=column_mapping)
    
    # Replace NaN with None for proper NULL insertion
    df = df.where(pd.notnull(df), None)
    
    # Convert listing_date to proper format if it exists and is not None
    if 'listing_date' in df.columns:
        def parse_date(date_str):
            if date_str is None or pd.isna(date_str):
                return None
            try:
                # Try different date formats
                if isinstance(date_str, str):
                    # Handle various Japanese date formats
                    if '年' in date_str and '月' in date_str and '日' in date_str:
                        # Format: YYYY年MM月DD日
                        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                        return datetime.strptime(date_str, '%Y-%m-%d').date()
                    elif '/' in date_str:
                        # Format: YYYY/MM/DD or MM/DD/YYYY
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            if len(parts[0]) == 4:  # YYYY/MM/DD
                                return datetime.strptime(date_str, '%Y/%m/%d').date()
                            else:  # MM/DD/YYYY
                                return datetime.strptime(date_str, '%m/%d/%Y').date()
                    elif '-' in date_str:
                        # Format: YYYY-MM-DD
                        return datetime.strptime(date_str, '%Y-%m-%d').date()
                return None
            except:
                return None
        
        df['listing_date'] = df['listing_date'].apply(parse_date)
    
    # Ensure consolidation_yes_no fits the expected values
    if 'consolidation_yes_no' in df.columns:
        df['consolidation_yes_no'] = df['consolidation_yes_no'].apply(
            lambda x: x if x in ['有', '無'] or x is None else None
        )
    
    # Process keywords column - convert semicolon-separated string to list
    if 'keywords' in df.columns:
        def process_keywords(keywords_str):
            if keywords_str is None or pd.isna(keywords_str) or keywords_str == '':
                return None
            # Split by semicolon and clean up each keyword
            keywords_list = [kw.strip() for kw in str(keywords_str).split(';') if kw.strip()]
            return keywords_list if keywords_list else None
        
        df['keywords'] = df['keywords'].apply(process_keywords)
        print(f"Processed keywords column - sample: {df['keywords'].dropna().iloc[0] if not df['keywords'].dropna().empty else 'No keywords found'}")
    
    # Process aliases column similarly if it exists
    if 'aliases' in df.columns:
        def process_aliases(aliases_str):
            if aliases_str is None or pd.isna(aliases_str) or aliases_str == '':
                return None
            # Split by semicolon and clean up each alias
            aliases_list = [alias.strip() for alias in str(aliases_str).split(';') if alias.strip()]
            return aliases_list if aliases_list else None
        
        df['aliases'] = df['aliases'].apply(process_aliases)
    
    return df

def insert_company_data(conn, df):
    """Insert company data into the database"""
    cursor = conn.cursor()
    
    try:
        # Define the expected database columns in order
        db_columns = [
            'securities_code', 'ticker', 'company_name_japanese', 'company_name_english',
            'company_name_kana', 'sector_japanese', 'sector_english', 'company_address',
            'corporate_number', 'listing_classification', 'consolidation_yes_no',
            'listing_date', 'market_status', 'aliases', 'keywords', 'fiscal_year_end', 
            'edinet_code', 'bloomberg_code'
        ]
        
        # Check which columns exist in both DataFrame and database schema
        available_columns = [col for col in db_columns if col in df.columns]
        print(f"Mapping {len(available_columns)} columns: {available_columns}")
        
        # Prepare the insert query with only available columns
        columns_str = ', '.join(available_columns)
        placeholders = ', '.join(['%s'] * len(available_columns))
        
        insert_query = f"""
        INSERT INTO company_master ({columns_str}) VALUES %s
        ON CONFLICT (securities_code) DO UPDATE SET
        """
        
        # Add update clauses for all columns except securities_code
        update_clauses = []
        for col in available_columns:
            if col != 'securities_code':
                update_clauses.append(f"{col} = EXCLUDED.{col}")
        
        if update_clauses:
            insert_query += ', '.join(update_clauses) + ", updated_at = NOW()"
        else:
            insert_query += "updated_at = NOW()"
        
        # Prepare data tuples with only available columns
        data_tuples = []
        for _, row in df.iterrows():
            data_tuple = tuple(row.get(col) for col in available_columns)
            data_tuples.append(data_tuple)
        
        print(f"Preparing to insert {len(data_tuples)} records...")
        
        # Execute bulk insert with smaller batches to handle potential issues
        batch_size = 500
        total_inserted = 0
        
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            execute_values(
                cursor, insert_query, batch,
                template=None, page_size=batch_size
            )
            total_inserted += len(batch)
            print(f"Processed {total_inserted}/{len(data_tuples)} records...")
        
        conn.commit()
        print(f"Successfully inserted/updated {total_inserted} records")
        
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def main():
    """Main function to populate company_master table"""
    
    # Check if CSV file exists
    csv_file = 'data/japan_company_master_utf8.csv'
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        sys.exit(1)
    
    # Connect to database
    print("Connecting to database...")
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database!")
        sys.exit(1)
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} records from CSV")
        
        # Display first few rows and column info
        print("\nCSV Columns:")
        print(df.columns.tolist())
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Clean data
        df_cleaned = clean_data(df)
        
        # Insert data
        print("\nInserting data into company_master table...")
        insert_company_data(conn, df_cleaned)
        
        print("\nData population completed successfully!")
        
        # Show summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM company_master")
        total_count = cursor.fetchone()[0]
        print(f"Total records in company_master table: {total_count}")
        cursor.close()
        
    except Exception as e:
        print(f"Error during data population: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main() 