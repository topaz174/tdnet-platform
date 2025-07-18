#!/usr/bin/env python3
"""
Setup script for TDnet Search scraper.
This script initializes the database and prepares the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scraper.tdnet_search.init_db_search import Base, engine
from config.config_tdnet_search import DB_CONFIG_SEARCH

def create_database():
    """Create the tdnet_search database if it doesn't exist."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        # Connect to PostgreSQL server (without specifying database)
        conn = psycopg2.connect(
            host=DB_CONFIG_SEARCH['host'],
            user=DB_CONFIG_SEARCH['user'],
            password=DB_CONFIG_SEARCH['password'],
            port=DB_CONFIG_SEARCH['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG_SEARCH['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{DB_CONFIG_SEARCH['database']}'...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG_SEARCH['database']}")
            print(f"Database '{DB_CONFIG_SEARCH['database']}' created successfully!")
        else:
            print(f"Database '{DB_CONFIG_SEARCH['database']}' already exists.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating database: {e}")
        print("Please ensure PostgreSQL is running and credentials are correct.")
        return False
    
    return True

def create_tables():
    """Create the database tables."""
    try:
        print("Creating database tables...")
        Base.metadata.create_all(engine)
        print("Database tables created successfully!")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    # Map pip package names to their import names
    package_imports = {
        'requests': 'requests',
        'beautifulsoup4': 'bs4',
        'sqlalchemy': 'sqlalchemy',
        'psycopg2': 'psycopg2',
        'python-dateutil': 'dateutil'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required dependencies are installed.")
    return True

def check_directories_config():
    """Check if directories.json exists and is configured."""
    directories_file = project_root / 'directories.json'
    
    if not directories_file.exists():
        print("directories.json not found. Creating default configuration...")
        
        import json
        default_config = {
            "pdf_directory": str(project_root / "pdfs"),
            "backup_directory": str(project_root / "backups")
        }
        
        with open(directories_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Created directories.json with default settings:")
        print(f"  PDF Directory: {default_config['pdf_directory']}")
        print(f"  Backup Directory: {default_config['backup_directory']}")
    else:
        print("directories.json found.")
    
    return True

def main():
    """Main setup function."""
    print("TDnet Search Scraper Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check directories configuration
    if not check_directories_config():
        return False
    
    # Create database
    if not create_database():
        return False
    
    # Create tables
    if not create_tables():
        return False
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nYou can now run the TDnet Search scraper:")
    print("  Default (historical from 1 month ago):")
    print("    python src/scraper_tdnet_search/tdnet_search_scraper.py")
    print("  Historical from specific date:")
    print("    python src/scraper_tdnet_search/tdnet_search_scraper.py historical 2024-01-01")
    print("  Specific date range:")
    print("    python src/scraper_tdnet_search/tdnet_search_scraper.py range 2024-01-01 2024-01-31")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 