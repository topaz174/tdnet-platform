#!/usr/bin/env python3
"""
Debug the asyncpg connection issue
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_asyncpg():
    try:
        import asyncpg
        
        # Try the exact same connection string that works with psycopg2
        pg_dsn = os.getenv("PG_DSN")
        print(f"Testing asyncpg with: {pg_dsn}")
        
        # Try to connect
        conn = await asyncpg.connect(pg_dsn)
        print("✅ asyncpg connection successful!")
        
        # Test a simple query
        result = await conn.fetchval("SELECT COUNT(*) FROM disclosures WHERE embedding IS NOT NULL")
        print(f"Documents with embeddings: {result}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ asyncpg connection failed: {e}")
        
        # Try to fix by parsing the DSN and constructing asyncpg-compatible version
        print("\n🔧 Trying to fix connection...")
        
        if pg_dsn.startswith("postgresql://"):
            # Parse the connection string
            import urllib.parse
            parsed = urllib.parse.urlparse(pg_dsn)
            
            print(f"Parsed connection:")
            print(f"  Host: {parsed.hostname}")
            print(f"  Port: {parsed.port}")
            print(f"  Database: {parsed.path[1:]}")  # Remove leading slash
            print(f"  User: {parsed.username}")
            
            try:
                # Try with explicit parameters
                conn = await asyncpg.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    database=parsed.path[1:],
                    user=parsed.username,
                    password=parsed.password
                )
                print("✅ asyncpg connection successful with explicit parameters!")
                
                result = await conn.fetchval("SELECT COUNT(*) FROM disclosures WHERE embedding IS NOT NULL")
                print(f"Documents with embeddings: {result}")
                
                await conn.close()
                return True
                
            except Exception as e2:
                print(f"❌ Still failed with explicit parameters: {e2}")
        
        return False

if __name__ == "__main__":
    asyncio.run(test_asyncpg())