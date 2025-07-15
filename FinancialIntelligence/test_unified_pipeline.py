#!/usr/bin/env python3
"""
Test script for the unified extraction pipeline

This script demonstrates how to use the unified pipeline and provides
a simple way to test it with a small subset of data.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unified_extraction_pipeline import (
    UnifiedExtractionPipeline, 
    UnifiedProcessingConfig,
    DatabaseManager,
    load_database_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_pipeline_connection():
    """Test basic database connection and data availability"""
    
    # Load database configuration from .env file or environment
    try:
        pg_dsn = load_database_config()
        print(f"Testing connection with auto-loaded config: {pg_dsn[:50]}...")
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        print("Please ensure .env file exists with database configuration")
        return False
    
    # Test database connection
    db_manager = DatabaseManager(pg_dsn)
    if not db_manager.connect():
        print("Failed to connect to database")
        return False
    
    print("✓ Database connection successful")
    
    # Test data availability
    try:
        rows = db_manager.get_disclosure_rows(test_mode=True, test_days=30, max_rows=10)
        print(f"✓ Found {len(rows)} disclosure rows in the last 30 days")
        
        if rows:
            sample_row = rows[0]
            print(f"Sample row: {sample_row.company_name} ({sample_row.company_code}) - {sample_row.disclosure_date}")
            print(f"  XBRL path: {'✓' if sample_row.xbrl_path else '✗'}")
            print(f"  PDF path: {'✓' if sample_row.pdf_path else '✗'}")
        
        db_manager.close()
        return len(rows) > 0
        
    except Exception as e:
        print(f"Error testing data availability: {e}")
        db_manager.close()
        return False

async def test_small_extraction():
    """Test the pipeline with a very small dataset"""
    
    try:
        pg_dsn = load_database_config()
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        return False
    
    print("\n" + "="*60)
    print("TESTING UNIFIED EXTRACTION PIPELINE")
    print("="*60)
    
    # Create test configuration
    config = UnifiedProcessingConfig(
        pg_dsn=pg_dsn,
        max_workers=4,  # Small number for testing
        max_concurrent_files=2,
        test_mode=True,
        test_days=30,  # Look at last 30 days
        max_test_rows=3,  # Process only 3 rows for testing
        output_dir="test_unified_output",
        save_chunks=True,
        enable_progress_monitoring=True
    )
    
    # Initialize pipeline
    pipeline = UnifiedExtractionPipeline(config)
    
    try:
        print("Starting test extraction...")
        start_time = datetime.now()
        
        # Run the pipeline
        summary = await pipeline.run_pipeline()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Test extraction completed in {duration:.2f} seconds")
        
        # Print results
        stats = summary.get('processing_stats', {})
        breakdown = summary.get('extraction_method_breakdown', {})
        
        print(f"\nResults:")
        print(f"  Rows processed: {stats.get('processed_files', 0)}")
        print(f"  Rows failed: {stats.get('failed_files', 0)}")
        print(f"  Total chunks extracted: {stats.get('total_chunks', 0)}")
        print(f"  Performance: {stats.get('files_per_second', 0):.2f} rows/s")
        
        print(f"\nExtraction method breakdown:")
        for method, count in breakdown.items():
            print(f"  {method}: {count}")
        
        # Show failed rows if any
        failed_rows = summary.get('failed_rows', [])
        if failed_rows:
            print(f"\nFailed rows ({len(failed_rows)}):")
            for failed in failed_rows[:3]:  # Show first 3 failures
                print(f"  ID {failed['disclosure_id']}: {failed['error']}")
        
        print(f"\nOutput saved to: {config.output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Test extraction failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("Unified Extraction Pipeline Test Suite")
    print("=" * 50)
    
    # Test 1: Database connection and data availability
    print("\n1. Testing database connection and data availability...")
    if not await test_pipeline_connection():
        print("✗ Database connection test failed")
        return
    
    # Test 2: Small extraction test
    print("\n2. Testing small extraction...")
    if not await test_small_extraction():
        print("✗ Small extraction test failed")
        return
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nTo run the full pipeline:")
    print("python src/unified_extraction_pipeline.py --test-mode --test-days 7")
    print("python src/unified_extraction_pipeline.py --full-pipeline --workers 16")
    print("\nFor more options:")
    print("python src/unified_extraction_pipeline.py --help")

if __name__ == "__main__":
    asyncio.run(main())