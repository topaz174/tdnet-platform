#!/usr/bin/env python3
"""
Run all database migrations.

This script executes all SQL migration files in the src/database/migrations directory
in alphabetical order to apply database schema changes.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_URL
from sqlalchemy import create_engine, text


def run_migrations():
    """Run all migration files in alphabetical order."""
    migrations_dir = project_root / "src" / "database" / "migrations"
    
    if not migrations_dir.exists():
        print(f"Error: Migrations directory not found at {migrations_dir}")
        return False
    
    # Get all SQL files in migrations directory (excluding subdirectories)
    migration_files = sorted([f for f in migrations_dir.glob("*.sql") if f.is_file()])
    
    if not migration_files:
        print("No migration files found.")
        return True
    
    print("="*60)
    print("Running Database Migrations")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Found {len(migration_files)} migration files")
    print()
    
    # Create database engine
    engine = create_engine(DB_URL)
    
    success_count = 0
    error_count = 0
    
    for i, migration_file in enumerate(migration_files, 1):
        
        try:
            # Read the SQL file
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Execute the migration
            with engine.connect() as conn:
                # Start transaction
                trans = conn.begin()
                try:
                    # Execute the SQL
                    conn.execute(text(sql_content))
                    # Commit transaction
                    trans.commit()
                    print(f"  ✓ {migration_file.name} completed successfully")
                    success_count += 1
                except Exception as e:
                    # Rollback on error
                    trans.rollback()
                    raise e
            
        except Exception as e:
            print(f"  ✗ Error running {migration_file.name}: {e}")
            error_count += 1
            continue
    
    print()
    print("="*60)
    print("Migrations Summary")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total: {len(migration_files)}")
    print("="*60)
    
    return error_count == 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all database migrations")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    args = parser.parse_args()
    
    if args.dry_run:
        migrations_dir = project_root / "src" / "database" / "migrations"
        migration_files = sorted([f for f in migrations_dir.glob("*.sql") if f.is_file()])
        print("Migration files that would be run:")
        for i, migration_file in enumerate(migration_files, 1):
            print(f"  {i}. {migration_file.name}")
        return
    
    success = run_migrations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 