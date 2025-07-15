#!/usr/bin/env python3
"""
Extraction Status Manager

A utility for managing extraction status and tracking in the unified pipeline.
Provides commands for viewing status, resetting failed entries, and managing
the extraction tracking system.

Usage:
    python extraction_status_manager.py status          # Show status report
    python extraction_status_manager.py reset-failed    # Reset failed entries to retry
    python extraction_status_manager.py reset-stuck     # Reset stuck processing entries
    python extraction_status_manager.py clear-all       # Clear all extraction status
    python extraction_status_manager.py setup           # Setup tracking columns
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.unified_extraction_pipeline import DatabaseManager, load_database_config
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Please ensure unified_extraction_pipeline.py is available in src/")
    sys.exit(1)

def setup_tracking_columns(db_manager: DatabaseManager) -> bool:
    """Setup tracking columns if they don't exist"""
    
    setup_sql = """
    -- Add extraction tracking columns if they don't exist
    ALTER TABLE disclosures 
    ADD COLUMN IF NOT EXISTS extraction_status VARCHAR(20) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS extraction_method VARCHAR(20),
    ADD COLUMN IF NOT EXISTS extraction_date TIMESTAMP,
    ADD COLUMN IF NOT EXISTS extraction_error TEXT,
    ADD COLUMN IF NOT EXISTS chunks_extracted INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS extraction_duration FLOAT DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS extraction_file_path TEXT,
    ADD COLUMN IF NOT EXISTS extraction_metadata JSONB;

    -- Create indexes for efficient querying
    CREATE INDEX IF NOT EXISTS disclosures_extraction_status_idx ON disclosures(extraction_status);
    CREATE INDEX IF NOT EXISTS disclosures_extraction_date_idx ON disclosures(extraction_date DESC);
    CREATE INDEX IF NOT EXISTS disclosures_status_date_idx ON disclosures(extraction_status, disclosure_date DESC);
    """
    
    try:
        with db_manager.conn.cursor() as cur:
            cur.execute(setup_sql)
            db_manager.conn.commit()
        
        print("✓ Tracking columns and indexes created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error setting up tracking columns: {e}")
        return False

def show_status_report(db_manager: DatabaseManager):
    """Show comprehensive status report"""
    
    try:
        stats = db_manager.get_extraction_statistics()
        
        if not stats:
            print("No statistics available")
            return
        
        print("\n" + "="*60)
        print("EXTRACTION STATUS REPORT")
        print("="*60)
        print(f"Generated: {stats['generated_at']}")
        
        # Overall statistics
        overall = stats['overall']
        print(f"\nOverall Statistics:")
        print(f"  Total disclosures: {overall['total_disclosures']:,}")
        print(f"  Processable files: {overall['processable']:,}")
        print(f"  Has XBRL: {overall['has_xbrl']:,}")
        print(f"  Has PDF: {overall['has_pdf']:,}")
        print(f"  Total chunks extracted: {overall['total_chunks']:,}")
        if overall.get('avg_duration'):
            print(f"  Average processing time: {overall['avg_duration']:.2f}s")
        
        # Status breakdown
        print(f"\nStatus Breakdown:")
        status_breakdown = stats['status_breakdown']
        for status, count in status_breakdown.items():
            percentage = (count / overall['processable'] * 100) if overall['processable'] > 0 else 0
            print(f"  {status:12}: {count:8,} ({percentage:5.1f}%)")
        
        # Method breakdown
        if stats['method_breakdown']:
            print(f"\nMethod Breakdown:")
            for method, data in stats['method_breakdown'].items():
                print(f"  {method:12}: {data['count']:8,} files, {data['total_chunks']:8,} chunks, avg {data['avg_duration']:.2f}s")
        
        # Recent activity
        if stats['recent_activity']:
            print(f"\nRecent Activity (Last 7 Days):")
            for date, count in sorted(stats['recent_activity'].items(), reverse=True):
                print(f"  {date}: {count:,} completed")
        
        # Progress calculation
        completed = overall.get('completed', 0)
        failed = overall.get('failed', 0)
        pending = overall.get('pending', 0)
        processing = overall.get('processing', 0)
        processable = overall.get('processable', 0)
        
        if processable > 0:
            progress = (completed / processable) * 100
            remaining = pending
            print(f"\nProgress Summary:")
            print(f"  Completed: {completed:,} ({progress:.1f}%)")
            print(f"  Failed: {failed:,}")
            print(f"  Pending: {pending:,}")
            if processing > 0:
                print(f"  Currently processing: {processing:,}")
            
            if completed > 0 and overall.get('avg_duration', 0) > 0:
                estimated_time = remaining * overall['avg_duration']
                hours = estimated_time / 3600
                days = hours / 24
                if days > 1:
                    print(f"  Estimated time remaining: {days:.1f} days")
                else:
                    print(f"  Estimated time remaining: {hours:.1f} hours")
        
    except Exception as e:
        print(f"Error generating status report: {e}")

def reset_failed_entries(db_manager: DatabaseManager, dry_run: bool = False) -> int:
    """Reset failed entries to retry status"""
    
    try:
        if dry_run:
            # Count what would be reset
            with db_manager.conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM disclosures 
                    WHERE extraction_status = 'failed'
                """)
                count = cur.fetchone()[0]
                print(f"Would reset {count} failed entries to 'retry' status")
                return count
        else:
            # Actually reset
            with db_manager.conn.cursor() as cur:
                cur.execute("""
                    UPDATE disclosures 
                    SET extraction_status = 'retry', 
                        extraction_error = NULL 
                    WHERE extraction_status = 'failed'
                """)
                count = cur.rowcount
                db_manager.conn.commit()
                print(f"✓ Reset {count} failed entries to 'retry' status")
                return count
                
    except Exception as e:
        print(f"✗ Error resetting failed entries: {e}")
        return 0

def reset_stuck_entries(db_manager: DatabaseManager, dry_run: bool = False) -> int:
    """Reset stuck processing entries"""
    
    try:
        if dry_run:
            # Count what would be reset
            with db_manager.conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM disclosures 
                    WHERE extraction_status = 'processing' 
                    AND (extraction_date IS NULL OR extraction_date < NOW() - INTERVAL '1 hour')
                """)
                count = cur.fetchone()[0]
                print(f"Would reset {count} stuck processing entries")
                return count
        else:
            # Actually reset
            reset_count = db_manager.reset_processing_status()
            print(f"✓ Reset {reset_count} stuck processing entries")
            return reset_count
            
    except Exception as e:
        print(f"✗ Error resetting stuck entries: {e}")
        return 0

def clear_all_status(db_manager: DatabaseManager, confirm: bool = False) -> bool:
    """Clear all extraction status (reset to pending)"""
    
    if not confirm:
        print("This will reset ALL extraction status to 'pending'. Are you sure?")
        response = input("Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return False
    
    try:
        with db_manager.conn.cursor() as cur:
            cur.execute("""
                UPDATE disclosures 
                SET 
                    extraction_status = 'pending',
                    extraction_method = NULL,
                    extraction_date = NULL,
                    extraction_error = NULL,
                    chunks_extracted = 0,
                    extraction_duration = 0.0,
                    extraction_file_path = NULL,
                    extraction_metadata = NULL
                WHERE (xbrl_path IS NOT NULL AND xbrl_path != '') 
                   OR (pdf_path IS NOT NULL AND pdf_path != '')
            """)
            count = cur.rowcount
            db_manager.conn.commit()
            print(f"✓ Reset {count} entries to 'pending' status")
            return True
            
    except Exception as e:
        print(f"✗ Error clearing status: {e}")
        return False

def list_failed_files(db_manager: DatabaseManager, limit: int = 20):
    """List failed files with error messages"""
    
    try:
        with db_manager.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id, 
                    company_code, 
                    company_name, 
                    extraction_error,
                    extraction_date
                FROM disclosures 
                WHERE extraction_status = 'failed'
                ORDER BY extraction_date DESC
                LIMIT %s
            """, (limit,))
            
            failed_files = cur.fetchall()
            
            if not failed_files:
                print("No failed files found")
                return
            
            print(f"\nFailed Files (showing last {len(failed_files)}):")
            print("-" * 80)
            
            for row in failed_files:
                print(f"ID: {row[0]}, Code: {row[1]}, Company: {row[2]}")
                print(f"Error: {row[3]}")
                if row[4]:
                    print(f"Date: {row[4]}")
                print("-" * 80)
                
    except Exception as e:
        print(f"Error listing failed files: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extraction Status Manager')
    parser.add_argument('command', choices=[
        'status', 'reset-failed', 'reset-stuck', 'clear-all', 'setup', 'list-failed'
    ], help='Command to execute')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--limit', type=int, default=20, help='Limit for list commands')
    
    args = parser.parse_args()
    
    # Load database configuration
    try:
        pg_dsn = load_database_config()
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        print("Please ensure .env file exists with database configuration")
        sys.exit(1)
    
    # Connect to database
    db_manager = DatabaseManager(pg_dsn)
    if not db_manager.connect():
        print("Error: Failed to connect to database")
        sys.exit(1)
    
    try:
        if args.command == 'setup':
            print("Setting up extraction tracking columns...")
            success = setup_tracking_columns(db_manager)
            sys.exit(0 if success else 1)
            
        elif args.command == 'status':
            show_status_report(db_manager)
            
        elif args.command == 'reset-failed':
            reset_failed_entries(db_manager, args.dry_run)
            
        elif args.command == 'reset-stuck':
            reset_stuck_entries(db_manager, args.dry_run)
            
        elif args.command == 'clear-all':
            clear_all_status(db_manager, args.confirm)
            
        elif args.command == 'list-failed':
            list_failed_files(db_manager, args.limit)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()