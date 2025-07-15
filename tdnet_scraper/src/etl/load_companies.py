#!/usr/bin/env python3
"""
Load companies from Excel file into the database.

Companies table schema:
    id            SERIAL PRIMARY KEY,
    company_code  VARCHAR(5) UNIQUE NOT NULL, -- e.g. '6758'
    name_en       TEXT NOT NULL,
    name_ja       TEXT,
    exchange_id   INTEGER REFERENCES exchanges(id),
    sector_id     INTEGER REFERENCES sectors(id),
    created_at    TIMESTAMP DEFAULT now()
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

from config.config import DB_URL


def load_companies_from_excel(excel_path: str) -> int:
    """
    Load companies from Excel file into the database.
    
    Args:
        excel_path (str): Path to the Excel file containing company data
        
    Returns:
        int: Number of companies successfully loaded
    """
    
    # Read Excel file
    try:
        df = pd.read_excel(excel_path)
        print(f"Read {len(df)} rows from {excel_path}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return 0
    
    # Validate required columns
    required_columns = ['company_code', 'name_en', 'name_ja']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return 0
    
    # Clean the data
    df = df.dropna(subset=['company_code', 'name_en'])  # name_ja can be null
    df['company_code'] = df['company_code'].astype(str).str.strip()
    df['name_en'] = df['name_en'].astype(str).str.strip()
    df['name_ja'] = df['name_ja'].fillna('').astype(str).str.strip()
    
    # Replace empty strings with None for name_ja
    df.loc[df['name_ja'] == '', 'name_ja'] = None
    
    print(f"After cleaning: {len(df)} valid rows")
    
    # Connect to database
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    inserted_count = 0
    duplicate_count = 0
    error_count = 0
    
    try:
        for _, row in df.iterrows():
            try:
                # Insert company using raw SQL to handle conflicts
                result = session.execute(
                    text("""
                        INSERT INTO companies (company_code, name_en, name_ja)
                        VALUES (:company_code, :name_en, :name_ja)
                        ON CONFLICT (company_code) DO NOTHING
                        RETURNING id
                    """),
                    {
                        'company_code': row['company_code'],
                        'name_en': row['name_en'],
                        'name_ja': row['name_ja']
                    }
                )
                
                if result.rowcount > 0:
                    inserted_count += 1
                    if inserted_count % 100 == 0:
                        print(f"Inserted {inserted_count} companies...")
                else:
                    duplicate_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"Error inserting company {row['company_code']}: {e}")
                session.rollback()
                continue
        
        # Commit all changes
        session.commit()
        
    except Exception as e:
        print(f"Database error: {e}")
        session.rollback()
    finally:
        session.close()
    
    print(f"\nLoad complete:")
    print(f"  - Inserted: {inserted_count} companies")
    print(f"  - Duplicates skipped: {duplicate_count} companies")
    print(f"  - Errors: {error_count} companies")
    
    return inserted_count


def main():
    """Main function to load companies."""
    excel_path = r"C:\Users\Alex\Desktop\Internship\tdnet_scraper\data_e.xls"
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return
    
    print(f"Loading companies from {excel_path}...")
    loaded_count = load_companies_from_excel(excel_path)
    print(f"Successfully loaded {loaded_count} companies into the database.")


if __name__ == "__main__":
    main()