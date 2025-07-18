#!/usr/bin/env python3
"""
Migration 013: Backup path columns before potential removal

This script creates a backup of xbrl_path and pdf_path columns from the 
disclosures table to a zip file in the backup directory before the columns 
are dropped from the main table.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.config import DB_CONFIG
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure psycopg2 and config are available")
    sys.exit(1)


def load_directories_config():
    """Load directories configuration from JSON file."""
    config_file = project_root / "directories.json"
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find directories.json at {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing directories.json: {e}")
        sys.exit(1)


def backup_path_columns():
    """Backup xbrl_path and pdf_path columns to a SQL file."""
    print("Migration 013: Backing up path columns...")
    
    # Load configuration
    directories = load_directories_config()
    backup_dir = Path(directories['backup_directory'])
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"disclosures_path_backup_{timestamp}.sql"
    backup_path = backup_dir / backup_filename
    
    # Connect to database
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query all path data from disclosures table
        query = """
            SELECT 
                id,
                disclosure_date,
                company_code,
                title,
                xbrl_path,
                pdf_path
            FROM disclosures
            WHERE xbrl_path IS NOT NULL OR pdf_path IS NOT NULL
            ORDER BY disclosure_date DESC, id
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("No path data found to backup.")
            return
        
        print(f"Found {len(rows)} records with path data to backup")
        
        # Create SQL backup file
        backup_time = datetime.now().isoformat()
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write(f"-- Migration 013: Path columns backup\n")
            f.write(f"-- Created: {backup_time}\n")
            f.write(f"-- Total records: {len(rows)}\n")
            f.write(f"-- Records with XBRL paths: {sum(1 for row in rows if row['xbrl_path'])}\n")
            f.write(f"-- Records with PDF paths: {sum(1 for row in rows if row['pdf_path'])}\n")
            f.write(f"-- Backup reason: Pre-removal backup of redundant path columns\n")
            f.write(f"\n")
            
            # Create backup table
            f.write(f"-- Create backup table for path data\n")
            f.write(f"CREATE TABLE IF NOT EXISTS disclosures_path_backup (\n")
            f.write(f"    id INTEGER PRIMARY KEY,\n")
            f.write(f"    disclosure_date DATE,\n")
            f.write(f"    company_code VARCHAR(10),\n")
            f.write(f"    title TEXT,\n")
            f.write(f"    xbrl_path TEXT,\n")
            f.write(f"    pdf_path TEXT,\n")
            f.write(f"    backed_up_at TIMESTAMP DEFAULT NOW()\n")
            f.write(f");\n\n")
            
            # Write INSERT statements
            f.write(f"-- Insert path data\n")
            for row in rows:
                # Escape single quotes in SQL
                def escape_sql(value):
                    if value is None:
                        return 'NULL'
                    return "'" + str(value).replace("'", "''") + "'"
                
                disclosure_date = f"'{row['disclosure_date']}'" if row['disclosure_date'] else 'NULL'
                
                f.write(f"INSERT INTO disclosures_path_backup (id, disclosure_date, company_code, title, xbrl_path, pdf_path) VALUES ")
                f.write(f"({row['id']}, {disclosure_date}, {escape_sql(row['company_code'])}, {escape_sql(row['title'])}, {escape_sql(row['xbrl_path'])}, {escape_sql(row['pdf_path'])});\n")
        
        # Display statistics
        xbrl_count = sum(1 for row in rows if row['xbrl_path'])
        pdf_count = sum(1 for row in rows if row['pdf_path'])
        
        print(f"\n✓ Path backup completed successfully:")
        print(f"  - Backup file: {backup_path}")
        print(f"  - Total records backed up: {len(rows)}")
        print(f"  - Records with XBRL paths: {xbrl_count}")
        print(f"  - Records with PDF paths: {pdf_count}")
        print(f"  - File size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        print(f"\nBackup contains:")
        print(f"  - SQL CREATE TABLE statement for disclosures_path_backup")
        print(f"  - SQL INSERT statements for all path data")
        print(f"  - Header comments with backup statistics")
        
        return backup_path
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"Backup error: {e}")
        return None
    finally:
        if conn:
            conn.close()


def main():
    """Main function."""
    print("=" * 60)
    print("TDnet Database Migration 013: Backup Path Columns")
    print("=" * 60)
    
    backup_path = backup_path_columns()
    
    if backup_path:
        print(f"\n✓ Migration 013 completed successfully")
        print(f"✓ Path data safely backed up to: {backup_path}")
        print(f"✓ Ready to proceed with migration 013 (drop path columns)")
    else:
        print(f"\n✗ Migration 013 failed")
        print(f"✗ Do not proceed with dropping columns until backup is successful")
        sys.exit(1)


if __name__ == "__main__":
    main() 