#!/usr/bin/env python3
"""
Run all database seeds.

This script executes all Python seed files in the src/database/seeds directory
in alphabetical order to populate the database with initial data.
"""

import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_URL
from sqlalchemy import create_engine, text


def run_seeds():
    """Run all seed files in alphabetical order."""
    seeds_dir = project_root / "src" / "database" / "seeds"
    
    if not seeds_dir.exists():
        print(f"Error: Seeds directory not found at {seeds_dir}")
        return False
    
    # Get all Python files in seeds directory
    seed_files = sorted([f for f in seeds_dir.glob("*.py") if f.is_file()])
    
    if not seed_files:
        print("No seed files found.")
        return True
    
    print("="*60)
    print("Running Database Seeds")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Found {len(seed_files)} seed files")
    print()
    
    # Create database engine
    engine = create_engine(DB_URL)
    
    success_count = 0
    error_count = 0
    
    for i, seed_file in enumerate(seed_files, 1):
        print(f"[{i}/{len(seed_files)}] Running {seed_file.name}...")
        
        try:
            # Import and run the seed module
            spec = importlib.util.spec_from_file_location(seed_file.stem, seed_file)
            if spec is None or spec.loader is None:
                print(f"  ✗ Could not load module {seed_file.name}")
                error_count += 1
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module has a main function or specific seed functions
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'run'):
                module.run()
            elif hasattr(module, 'seed_units'):
                module.seed_units()
            elif hasattr(module, 'populate_classification_tables'):
                module.populate_classification_tables()
            elif hasattr(module, 'load_companies_from_csv'):
                module.load_companies_from_csv()
            elif hasattr(module, 'load_concepts'):
                module.load_concepts()
            else:
                print(f"  ⚠️  No recognized function found in {seed_file.name}")
                print(f"     Available functions: {[f for f in dir(module) if not f.startswith('_')]}")
                continue
            
            print(f"  ✓ {seed_file.name} completed successfully")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error running {seed_file.name}: {e}")
            error_count += 1
            continue
    
    print()
    print("="*60)
    print("Seeds Summary")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total: {len(seed_files)}")
    print("="*60)
    
    return error_count == 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all database seeds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    args = parser.parse_args()
    
    if args.dry_run:
        seeds_dir = project_root / "src" / "database" / "seeds"
        seed_files = sorted([f for f in seeds_dir.glob("*.py") if f.is_file()])
        print("Seed files that would be run:")
        for i, seed_file in enumerate(seed_files, 1):
            print(f"  {i}. {seed_file.name}")
        return
    
    success = run_seeds()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 