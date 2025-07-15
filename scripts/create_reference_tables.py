#!/usr/bin/env python3
"""
TDnet Database Reference Tables Creation Script

This script creates the reference tables (company_master, disclosures, document_chunks)
that will serve as the baseline for database migrations.

Usage:
    python scripts/create_reference_tables.py [--database-url DATABASE_URL]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add the src directory to the Python path to allow imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/create_reference_tables.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment or prompt user"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        db_url = os.getenv('TDNET_DB_URL')
    
    if not db_url:
        # Default for local development
        db_url = "postgresql://postgres:password@localhost/tdnet"
        print(f"No DATABASE_URL found, using default: {db_url}")
    
    return db_url

def read_sql_file():
    """Read the SQL file content"""
    sql_file = Path(__file__).parent / "create_reference_tables.sql"
    
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        return f.read()

def execute_sql_script(db_url, sql_content, logger):
    """Execute the SQL script to create reference tables"""
    try:
        # Connect to database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cursor:
            logger.info("Executing reference tables creation script...")
            
            # Execute the SQL script
            cursor.execute(sql_content)
            
            # Get any notices/messages from the execution
            if conn.notices:
                for notice in conn.notices:
                    logger.info(f"Database notice: {notice.strip()}")
        
        logger.info("✅ Reference tables created successfully!")
        
        # Verify tables were created
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('company_master', 'disclosures', 'document_chunks', 'reports')
                ORDER BY table_name;
            """)
            
            tables = cursor.fetchall()
            logger.info("Created tables:")
            for table in tables:
                logger.info(f"  - {table[0]}")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def check_existing_tables(db_url, logger):
    """Check if tables already exist"""
    try:
        conn = psycopg2.connect(db_url)
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('company_master', 'disclosures', 'document_chunks')
                ORDER BY table_name;
            """)
            
            existing_tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        if existing_tables:
            logger.warning(f"The following tables already exist: {', '.join(existing_tables)}")
            response = input("Do you want to continue? This may cause errors. (y/N): ")
            return response.lower() in ['y', 'yes']
        
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error checking existing tables: {e}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Create TDnet reference tables')
    parser.add_argument(
        '--database-url', 
        help='PostgreSQL database URL (default: from DATABASE_URL env var)'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Skip existing tables check'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("TDnet Reference Tables Creation")
    logger.info("=" * 60)
    
    try:
        # Get database URL
        db_url = args.database_url or get_database_url()
        logger.info(f"Database URL: {db_url.split('@')[1] if '@' in db_url else db_url}")
        
        # Check existing tables unless forced
        if not args.force:
            if not check_existing_tables(db_url, logger):
                logger.info("Operation cancelled by user")
                return
        
        # Read SQL script
        logger.info("Reading SQL script...")
        sql_content = read_sql_file()
        
        # Execute script
        success = execute_sql_script(db_url, sql_content, logger)
        
        if success:
            logger.info("=" * 60)
            logger.info("✅ Reference tables creation completed successfully!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Verify tables using: \\dt in psql")
            logger.info("2. Check table structure: \\d table_name")
            logger.info("3. Begin database migration planning")
            logger.info("4. Populate company_master table with reference data")
        else:
            logger.error("❌ Reference tables creation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 