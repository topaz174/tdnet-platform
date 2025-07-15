#!/usr/bin/env python3
"""
Quick database cleanup - bypasses heavy imports for fast deletion
"""

import os
import psycopg2
from dotenv import load_dotenv

def quick_clean():
    # Load environment
    load_dotenv()
    
    # Get connection string
    pg_dsn = os.getenv('PG_DSN')
    if not pg_dsn:
        # Build from components
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'tdnet')
        
        # Remove quotes if present
        if db_password.startswith('"') and db_password.endswith('"'):
            db_password = db_password[1:-1]
        elif db_password.startswith("'") and db_password.endswith("'"):
            db_password = db_password[1:-1]
            
        pg_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    print("Connecting to database...")
    
    try:
        # Connect
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Check current chunks
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cur.fetchone()[0]
            print(f"Current chunks in database: {chunk_count:,}")
            
            if chunk_count == 0:
                print("✓ Database already clean!")
                return
            
            print("Deleting all chunks...")
            # Delete chunks
            cur.execute("DELETE FROM document_chunks")
            print(f"✓ Deleted {chunk_count:,} chunks")
            
            print("Resetting extraction status...")
            # Reset extraction status
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
            """)
            reset_count = cur.rowcount
            print(f"✓ Reset {reset_count:,} disclosure statuses")
            
            # Verify cleanup
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            remaining = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(*) 
                FROM disclosures 
                WHERE (xbrl_path IS NOT NULL AND xbrl_path != '') 
                   OR (pdf_path IS NOT NULL AND pdf_path != '')
            """)
            processable = cur.fetchone()[0]
            
            print(f"\n✅ Cleanup completed successfully!")
            print(f"   Remaining chunks: {remaining}")
            print(f"   Processable disclosures: {processable:,}")
            print(f"   Ready for full extraction!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()
    
    return True

if __name__ == "__main__":
    quick_clean()