#!/usr/bin/env python3
"""
Validate XBRL paths for earnings reports in the disclosures table.

This script:
1. Finds all disclosures categorized as earnings reports (subcategory contains '決算短信')
2. Checks if their xbrl_path actually points to an existing file/directory
3. For those that don't exist, sets xbrl_path to NULL in the database
4. Outputs titles of disclosures with invalid paths to a text file
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path for config imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.config import DB_CONFIG
except ImportError:
    print("Error: Could not import database configuration.")
    print("Make sure config/config.py exists and contains DB_CONFIG.")
    sys.exit(1)

def check_xbrl_path_exists(xbrl_path: str) -> bool:
    """Check if the XBRL path exists as a file or directory."""
    if not xbrl_path:
        return False
    
    path = Path(xbrl_path)
    
    # Check if path exists as absolute path
    if path.exists():
        return True
    
    # If not absolute, try relative to project root
    relative_path = project_root / xbrl_path
    if relative_path.exists():
        return True
    
    # Try relative to downloads directory (common location)
    downloads_path = project_root / "downloads" / xbrl_path
    if downloads_path.exists():
        return True
    
    return False

def get_earnings_reports_with_xbrl(db_config: dict) -> List[Tuple]:
    """Fetch all earnings reports that have xbrl_path set."""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query for earnings reports with xbrl_path
        query = """
            SELECT id, title, xbrl_path, company_code, company_name, disclosure_date
            FROM disclosures 
            WHERE subcategory LIKE '%決算短信%' 
            AND xbrl_path IS NOT NULL 
            AND xbrl_path != ''
            ORDER BY disclosure_date DESC, company_code
        """
        
        cur.execute(query)
        results = cur.fetchall()
        
        print(f"Found {len(results)} earnings reports with XBRL paths")
        return results
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return []
    finally:
        if conn:
            conn.close()

def update_invalid_xbrl_paths(db_config: dict, invalid_ids: List[int]) -> int:
    """Set xbrl_path to NULL for disclosures with invalid paths."""
    if not invalid_ids:
        return 0
    
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # Update query to set xbrl_path to NULL
        query = """
            UPDATE disclosures 
            SET xbrl_path = NULL 
            WHERE id = ANY(%s)
        """
        
        cur.execute(query, (invalid_ids,))
        updated_count = cur.rowcount
        
        conn.commit()
        print(f"Updated {updated_count} records, setting xbrl_path to NULL")
        return updated_count
        
    except Exception as e:
        print(f"Error updating database: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def main():
    """Main execution function."""
    print("XBRL Path Validation for Earnings Reports")
    print("=" * 50)
    
    # Get all earnings reports with XBRL paths
    earnings_reports = get_earnings_reports_with_xbrl(DB_CONFIG)
    
    if not earnings_reports:
        print("No earnings reports with XBRL paths found.")
        return
    
    # Check which paths are valid
    valid_paths = []
    invalid_paths = []
    invalid_ids = []
    
    print("\nValidating XBRL paths...")
    for i, report in enumerate(earnings_reports, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(earnings_reports)} reports...")
        
        path_exists = check_xbrl_path_exists(report['xbrl_path'])
        
        if path_exists:
            valid_paths.append(report)
        else:
            invalid_paths.append(report)
            invalid_ids.append(report['id'])
    
    # Print summary
    print(f"\nValidation Results:")
    print(f"  Valid XBRL paths: {len(valid_paths)}")
    print(f"  Invalid XBRL paths: {len(invalid_paths)}")
    print(f"  Total checked: {len(earnings_reports)}")
    
    if invalid_paths:
        # Write invalid titles to text file
        output_file = project_root / "invalid_xbrl_earnings_reports.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Earnings Reports with Invalid XBRL Paths\n")
            f.write(f"Generated: {Path(__file__).name}\n")
            f.write(f"Total invalid: {len(invalid_paths)}\n")
            f.write("=" * 80 + "\n\n")
            
            for report in invalid_paths:
                f.write(f"ID: {report['id']}\n")
                f.write(f"Company: {report['company_code']} - {report['company_name']}\n")
                f.write(f"Date: {report['disclosure_date']}\n")
                f.write(f"Title: {report['title']}\n")
                f.write(f"Invalid Path: {report['xbrl_path']}\n")
                f.write("-" * 80 + "\n")
        
        print(f"\nInvalid XBRL paths written to: {output_file}")
        
        # Ask user confirmation before updating database
        response = input(f"\nUpdate database to set {len(invalid_ids)} xbrl_path values to NULL? (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            updated_count = update_invalid_xbrl_paths(DB_CONFIG, invalid_ids)
            print(f"Database updated successfully. {updated_count} records modified.")
        else:
            print("Database update skipped.")
    else:
        print("\nAll XBRL paths are valid! No database updates needed.")

if __name__ == "__main__":
    main() 