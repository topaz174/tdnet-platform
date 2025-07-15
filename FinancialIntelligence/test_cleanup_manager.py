#!/usr/bin/env python3
"""
Test Cleanup Manager

A utility for safely testing the unified pipeline with database storage
and cleaning up test data afterwards.

Usage:
    python test_cleanup_manager.py test --days 7         # Run test and show cleanup options
    python test_cleanup_manager.py cleanup --recent      # Clean recent chunks (24h)
    python test_cleanup_manager.py cleanup --all         # Clean ALL chunks (dangerous!)
    python test_cleanup_manager.py status                # Show current chunk statistics
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.unified_extraction_pipeline import DatabaseManager, load_database_config
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    sys.exit(1)

def show_chunk_statistics(db_manager: DatabaseManager):
    """Show current chunk statistics"""
    
    try:
        with db_manager.conn.cursor() as cur:
            # Overall chunk statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT disclosure_id) as unique_disclosures,
                    MIN(created_at) as oldest_chunk,
                    MAX(created_at) as newest_chunk
                FROM document_chunks
            """)
            
            stats = cur.fetchone()
            
            print("\n" + "="*60)
            print("DOCUMENT CHUNKS STATISTICS")
            print("="*60)
            
            if stats[0] == 0:
                print("No chunks found in database")
                return
            
            print(f"Total chunks: {stats[0]:,}")
            print(f"Unique disclosures: {stats[1]:,}")
            print(f"Oldest chunk: {stats[2]}")
            print(f"Newest chunk: {stats[3]}")
            
            # Method breakdown
            cur.execute("""
                SELECT 
                    metadata->>'extraction_method' as method,
                    COUNT(*) as chunk_count,
                    COUNT(DISTINCT disclosure_id) as disclosure_count
                FROM document_chunks 
                WHERE metadata->>'extraction_method' IS NOT NULL
                GROUP BY metadata->>'extraction_method'
                ORDER BY chunk_count DESC
            """)
            
            method_stats = cur.fetchall()
            if method_stats:
                print("\nBy Extraction Method:")
                for method, chunk_count, disclosure_count in method_stats:
                    print(f"  {method:12}: {chunk_count:8,} chunks from {disclosure_count:,} disclosures")
            
            # Recent activity (last 24 hours)
            cur.execute("""
                SELECT 
                    COUNT(*) as recent_chunks,
                    COUNT(DISTINCT disclosure_id) as recent_disclosures
                FROM document_chunks 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            
            recent = cur.fetchone()
            if recent[0] > 0:
                print(f"\nRecent Activity (Last 24 Hours):")
                print(f"  New chunks: {recent[0]:,}")
                print(f"  New disclosures: {recent[1]:,}")
                
    except Exception as e:
        print(f"Error getting chunk statistics: {e}")

def cleanup_recent_chunks(db_manager: DatabaseManager, dry_run: bool = False) -> int:
    """Clean up chunks from the last 24 hours"""
    
    try:
        if dry_run:
            # Count what would be deleted
            with db_manager.conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT disclosure_id)
                    FROM document_chunks 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                count, disclosure_count = cur.fetchone()
                print(f"Would delete {count:,} chunks from {disclosure_count:,} disclosures (last 24 hours)")
                return count
        else:
            # Actually delete
            with db_manager.conn.cursor() as cur:
                # Delete chunks
                cur.execute("""
                    DELETE FROM document_chunks 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                deleted_chunks = cur.rowcount
                
                # Reset extraction status for affected disclosures
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
                    WHERE extraction_date >= NOW() - INTERVAL '24 hours'
                """)
                reset_disclosures = cur.rowcount
                
                db_manager.conn.commit()
                
                print(f"✓ Deleted {deleted_chunks:,} chunks")
                print(f"✓ Reset {reset_disclosures:,} disclosure statuses")
                
                return deleted_chunks
                
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        db_manager.conn.rollback()
        return 0

def cleanup_all_chunks(db_manager: DatabaseManager, confirm: bool = False) -> int:
    """Clean up ALL chunks (dangerous!)"""
    
    if not confirm:
        print("⚠️  WARNING: This will delete ALL chunks from the database!")
        response = input("Type 'DELETE ALL CHUNKS' to confirm: ")
        if response != 'DELETE ALL CHUNKS':
            print("Operation cancelled")
            return 0
    
    try:
        with db_manager.conn.cursor() as cur:
            # Count first
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            total_count = cur.fetchone()[0]
            
            if total_count == 0:
                print("No chunks to delete")
                return 0
            
            # Delete all chunks
            cur.execute("DELETE FROM document_chunks")
            deleted_count = cur.rowcount
            
            # Reset all extraction statuses
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
                WHERE extraction_status IS NOT NULL
            """)
            reset_disclosures = cur.rowcount
            
            db_manager.conn.commit()
            
            print(f"✓ Deleted {deleted_count:,} chunks")
            print(f"✓ Reset {reset_disclosures:,} disclosure statuses")
            
            return deleted_count
            
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        db_manager.conn.rollback()
        return 0

def run_test_pipeline(days: int = 7):
    """Run the pipeline in test mode"""
    
    import subprocess
    
    print(f"Running pipeline in test mode (last {days} days)...")
    print("Command: python src/unified_extraction_pipeline.py --test-mode --test-days", days)
    print("-" * 60)
    
    try:
        result = subprocess.run([
            'python', 'src/unified_extraction_pipeline.py', 
            '--test-mode', '--test-days', str(days)
        ], capture_output=False, text=True)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Cleanup Manager for Unified Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run pipeline test')
    test_parser.add_argument('--days', type=int, default=7, help='Number of days to test')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up test data')
    cleanup_group = cleanup_parser.add_mutually_exclusive_group(required=True)
    cleanup_group.add_argument('--recent', action='store_true', help='Clean recent chunks (24 hours)')
    cleanup_group.add_argument('--all', action='store_true', help='Clean ALL chunks (dangerous!)')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    cleanup_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompts')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show chunk statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load database configuration
    try:
        pg_dsn = load_database_config()
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        sys.exit(1)
    
    # Connect to database
    db_manager = DatabaseManager(pg_dsn)
    if not db_manager.connect():
        print("Error: Failed to connect to database")
        sys.exit(1)
    
    try:
        if args.command == 'test':
            # Show statistics before test
            print("BEFORE TEST:")
            show_chunk_statistics(db_manager)
            
            # Run test
            success = run_test_pipeline(args.days)
            
            if success:
                print("\n" + "="*60)
                print("TEST COMPLETED SUCCESSFULLY!")
                print("="*60)
                
                # Show statistics after test
                print("\nAFTER TEST:")
                show_chunk_statistics(db_manager)
                
                print("\n" + "="*60)
                print("CLEANUP OPTIONS:")
                print("="*60)
                print("1. Clean recent chunks (24h):  python test_cleanup_manager.py cleanup --recent")
                print("2. Clean ALL chunks:           python test_cleanup_manager.py cleanup --all")
                print("3. Show current status:        python test_cleanup_manager.py status")
                print("4. Manual cleanup:             psql $PG_DSN -f cleanup_test_chunks.sql")
            else:
                print("\nTest failed. Check the output above for errors.")
                
        elif args.command == 'cleanup':
            if args.recent:
                if args.dry_run:
                    cleanup_recent_chunks(db_manager, dry_run=True)
                else:
                    deleted = cleanup_recent_chunks(db_manager, dry_run=False)
                    if deleted > 0:
                        print("\n✓ Cleanup completed successfully!")
                        print("  Run the pipeline again to re-test with the same data.")
            elif args.all:
                deleted = cleanup_all_chunks(db_manager, args.confirm)
                if deleted > 0:
                    print("\n✓ All chunks deleted!")
                    print("  Database is now clean for fresh testing.")
                    
        elif args.command == 'status':
            show_chunk_statistics(db_manager)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()