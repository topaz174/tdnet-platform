#!/usr/bin/env python3
"""
Simple test for .env file loading without heavy imports.
"""

import os
import sys
from pathlib import Path

# Test loading environment variables directly
try:
    from dotenv import load_dotenv
    
    # Load .env file
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded .env file: {env_file.absolute()}")
    else:
        print("✗ .env file not found")
        sys.exit(1)
    
    # Check if we can get database configuration
    pg_dsn = os.getenv('PG_DSN')
    if pg_dsn:
        # Hide password
        import re
        match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', pg_dsn)
        if match:
            user, password, host, port, dbname = match.groups()
            display_dsn = f"postgresql://{user}:***@{host}:{port}/{dbname}"
            print(f"✓ PG_DSN loaded: {display_dsn}")
        else:
            print(f"✓ PG_DSN loaded: {pg_dsn[:50]}...")
    else:
        # Try building from components
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'tdnet')
        
        if db_password:
            # Remove quotes if present
            if db_password.startswith("'") and db_password.endswith("'"):
                db_password = db_password[1:-1]
            elif db_password.startswith('"') and db_password.endswith('"'):
                db_password = db_password[1:-1]
            
            pg_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            print(f"✓ Built PG_DSN from components: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}")
        else:
            print("✗ No database password found")
            sys.exit(1)
    
    print("✅ Database configuration successfully loaded from .env file!")
    
except ImportError:
    print("✗ python-dotenv not available. Install with: pip install python-dotenv")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)