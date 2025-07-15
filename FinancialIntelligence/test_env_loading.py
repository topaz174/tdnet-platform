#!/usr/bin/env python3
"""
Test script to verify .env file loading for database configuration.

This script tests the load_database_config function to ensure it properly
reads from the .env file and constructs the connection string.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.unified_extraction_pipeline import load_database_config
except ImportError as e:
    print(f"Error importing: {e}")
    print("Please ensure unified_extraction_pipeline.py is available in src/")
    sys.exit(1)

def test_env_loading():
    """Test database configuration loading"""
    
    print("Testing Database Configuration Loading")
    print("=" * 50)
    
    # Test 1: Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✓ Found .env file: {env_file.absolute()}")
        
        # Read and display .env contents (hiding sensitive info)
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        print("\n.env file contents:")
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'PASSWORD' in line:
                    key, value = line.split('=', 1)
                    print(f"  {key}=***")
                else:
                    print(f"  {line}")
    else:
        print("✗ .env file not found")
        return False
    
    # Test 2: Try loading configuration
    print("\nTesting configuration loading...")
    try:
        pg_dsn = load_database_config()
        
        # Hide password in output
        if pg_dsn:
            # Extract components for display
            import re
            match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', pg_dsn)
            if match:
                user, password, host, port, dbname = match.groups()
                display_dsn = f"postgresql://{user}:***@{host}:{port}/{dbname}"
                print(f"✓ Successfully loaded: {display_dsn}")
            else:
                print(f"✓ Successfully loaded: {pg_dsn[:50]}...")
        else:
            print("✗ No connection string returned")
            return False
            
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False
    
    # Test 3: Verify individual components (if available)
    print("\nEnvironment variables:")
    env_vars = ['PG_DSN', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME', 'DB_PORT']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                print(f"  {var}: ***")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: (not set)")
    
    print("\n✓ Configuration loading test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_env_loading()
    if not success:
        print("\n❌ Configuration loading test failed!")
        print("\nPlease ensure:")
        print("1. .env file exists in the root directory")
        print("2. .env file contains either PG_DSN or individual DB_* variables")
        print("3. python-dotenv is installed: pip install python-dotenv")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Database configuration is properly set up.")