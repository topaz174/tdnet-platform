#!/usr/bin/env python3
"""
Minimal database cleanup without heavy imports
"""
import os
import psycopg2

def main():
    # Load environment directly
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get connection
    pg_dsn = os.getenv('PG_DSN')
    if not pg_dsn:
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '').strip('\'"')
        db_host = os.getenv('DB_HOST', 'localhost') 
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'tdnet')
        pg_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    print("Connecting to database...")
    
    try:
        conn = psycopg2.connect(pg_dsn)
        print("✓ Connected successfully")
        
        with conn.cursor() as cur:
            # Check current state
            print("\nChecking current state...")
            
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cur.fetchone()[0]
            print(f"Current chunks: {chunk_count:,}")
            
            cur.execute("""
                SELECT 
                    extraction_status,
                    COUNT(*) 
                FROM disclosures 
                WHERE (xbrl_path IS NOT NULL AND xbrl_path != '') 
                   OR (pdf_path IS NOT NULL AND pdf_path != '')
                GROUP BY extraction_status 
                ORDER BY COUNT(*) DESC
            """)
            
            status_counts = cur.fetchall()
            print("Disclosure status breakdown:")
            for status, count in status_counts:
                print(f"  {status or 'NULL'}: {count:,}")
            
            if chunk_count == 0:
                print("\n✓ Chunks table already empty")
            else:
                print(f"\nDeleting {chunk_count:,} chunks...")
                cur.execute("DELETE FROM document_chunks")
                print("✓ All chunks deleted")
            
            print("\nResetting extraction status...")
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
            
            reset_count = cur.rowcount
            print(f"✓ Reset {reset_count:,} disclosure statuses")
            
            conn.commit()
            
            # Verify cleanup
            print("\nVerifying cleanup...")
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            remaining_chunks = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(*) 
                FROM disclosures 
                WHERE extraction_status = 'pending'
                AND ((xbrl_path IS NOT NULL AND xbrl_path != '') 
                     OR (pdf_path IS NOT NULL AND pdf_path != ''))
            """)
            pending_count = cur.fetchone()[0]
            
            print(f"Remaining chunks: {remaining_chunks}")
            print(f"Pending extractions: {pending_count:,}")
            
            if remaining_chunks == 0 and pending_count > 0:
                print("\n🎉 Database cleanup successful!")
                print("Ready for clean extraction")
            else:
                print("\n⚠️ Cleanup may not be complete")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")
    
    return True

if __name__ == "__main__":
    main()