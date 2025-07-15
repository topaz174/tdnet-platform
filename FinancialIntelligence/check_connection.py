#!/usr/bin/env python3
"""
Simple database connection check
"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_connection():
    pg_dsn = os.getenv("PG_DSN")
    
    if not pg_dsn:
        print("❌ PG_DSN environment variable not set")
        return
    
    print(f"🔍 Testing connection with: {pg_dsn}")
    
    # Try with psycopg2 first (synchronous)
    try:
        import psycopg2
        conn = psycopg2.connect(pg_dsn)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"✅ psycopg2 connection successful!")
        print(f"   PostgreSQL version: {version}")
        
        # Quick table check
        cur.execute("SELECT COUNT(*) FROM disclosures;")
        count = cur.fetchone()[0]
        print(f"   Total documents in disclosures table: {count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ psycopg2 connection failed: {e}")
        return False

if __name__ == "__main__":
    check_connection()