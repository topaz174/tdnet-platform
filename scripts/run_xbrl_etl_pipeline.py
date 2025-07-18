#!/usr/bin/env python3
"""
XBRL ETL Pipeline Orchestrator

This script coordinates the loading of XBRL filings and their associated facts
in a sequential manner to ensure data consistency and efficient processing.

The pipeline:
1. Loads one XBRL filing into the xbrl_filings table
2. Immediately loads all facts for that filing
3. Moves to the next filing
4. Continues until all filings are processed

This approach ensures that:
- The database always has complete data for processed filings
- Memory usage is controlled by processing one filing at a time
- Failed filings don't block subsequent processing
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.quantitative.etl.load_xbrl_filings import XBRLFilingLoader
from src.quantitative.etl.load_facts import FactsLoader


class XBRLETLOrchestrator:
    """Orchestrates the XBRL ETL pipeline."""
    
    def __init__(self):
        self.filing_loader = XBRLFilingLoader()
        self.facts_loader = FactsLoader()
        
    def run_pipeline(
        self, 
        company_code: Optional[str] = None,
        batch_size: int = 1
    ) -> dict:
        """
        Run the complete XBRL ETL pipeline.
        
        Args:
            company_code: Optional company code filter
            batch_size: Number of filings to process in each batch (currently always 1)
            
        Returns:
            Dict with processing statistics
        """
        start_time = time.time()
        print("="*80)
        print("XBRL ETL Pipeline Started")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Processing all available unprocessed filings")
        if company_code:
            print(f"Company filter: {company_code}")
        print()
        
        stats = {
            'filings_processed': 0,
            'facts_loaded': 0,
            'filings_failed': 0,
            'facts_failed': 0,
            'start_time': start_time
        }
        
        try:
            # Get list of disclosures to process
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            from config.config import DB_URL
            
            engine = create_engine(DB_URL)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            # Query for unprocessed disclosures
            query = """
                SELECT 
                    d.id as disclosure_id,
                    d.company_code,
                    d.title,
                    d.disclosure_date,
                    d.time,
                    c.id as company_id
                FROM disclosures d
                LEFT JOIN companies c ON (
                    d.company_code = c.ticker OR 
                    (LENGTH(d.company_code) = 5 AND SUBSTRING(d.company_code, 1, 4) = c.ticker)
                )
                WHERE d.has_xbrl = true
                  AND NOT EXISTS (
                      SELECT 1 FROM xbrl_filings xf WHERE xf.disclosure_id = d.id
                  )
            """
            
            if company_code:
                query += " AND d.company_code = :company_code"
            
            query += " ORDER BY d.disclosure_date DESC, d.time DESC"
            
            params = {}
            if company_code:
                params['company_code'] = company_code
                
            disclosures = session.execute(text(query), params).fetchall()
            session.close()
            
            total_disclosures = len(disclosures)
            print(f"Found {total_disclosures} unprocessed XBRL disclosures")
            
            if total_disclosures == 0:
                print("No unprocessed disclosures found. Pipeline complete.")
                return stats
            
            print("\nStarting sequential processing...")
            print("-" * 80)
            
            # Process each disclosure sequentially
            for idx, disclosure in enumerate(disclosures, 1):
                filing_start = time.time()
                
                print(f"\n[{idx}/{total_disclosures}] Processing {disclosure.company_code}: {disclosure.title[:60]}...")
                
                # Step 1: Load the filing
                try:
                    filing_count = self.filing_loader.load_filings(
                        company_code=disclosure.company_code
                    )
                    
                    if filing_count > 0:
                        stats['filings_processed'] += 1
                        
                        # Step 2: Load facts for this filing
                        try:
                            facts_count = self.facts_loader.load_facts(
                                company_code=disclosure.company_code
                            )
                            stats['facts_loaded'] += facts_count
                            
                            filing_time = time.time() - filing_start
                            print(f"  ✓ Filing loaded, {facts_count} facts loaded ({filing_time:.1f}s)")
                            
                        except Exception as e:
                            stats['facts_failed'] += 1
                            print(f"  ✗ Facts loading failed: {e}")
                            
                    else:
                        stats['filings_failed'] += 1
                        print(f"  ✗ Filing loading failed")
                        
                except Exception as e:
                    stats['filings_failed'] += 1
                    print(f"  ✗ Filing loading error: {e}")
                
                # Progress update every 10 filings
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (total_disclosures - idx) / rate if rate > 0 else 0
                    
                    print(f"\n--- Progress Update ---")
                    print(f"Processed: {idx}/{total_disclosures} ({idx/total_disclosures*100:.1f}%)")
                    print(f"Rate: {rate:.2f} filings/sec")
                    print(f"ETA: {eta/3600:.1f} hours")
                    print(f"Success rate: {stats['filings_processed']/idx*100:.1f}%")
                    print("-" * 40)
        
        except Exception as e:
            print(f"Pipeline error: {e}")
            
        # Final statistics
        elapsed_time = time.time() - start_time
        stats['elapsed_time'] = elapsed_time
        
        print("\n" + "="*80)
        print("XBRL ETL Pipeline Completed")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"Filings processed: {stats['filings_processed']}")
        print(f"Facts loaded: {stats['facts_loaded']}")
        print(f"Filings failed: {stats['filings_failed']}")
        print(f"Facts failed: {stats['facts_failed']}")
        
        if stats['filings_processed'] > 0:
            print(f"Average facts per filing: {stats['facts_loaded']/stats['filings_processed']:.0f}")
            print(f"Processing rate: {stats['filings_processed']/elapsed_time*3600:.0f} filings/hour")
        
        print("="*80)
        
        return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run XBRL ETL Pipeline")
    parser.add_argument("--code", help="Restrict to specific company_code", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size (currently unused, always 1)", default=1)
    
    args = parser.parse_args()
    
    orchestrator = XBRLETLOrchestrator()
    stats = orchestrator.run_pipeline(
        company_code=args.company,
        batch_size=args.batch_size
    )
    
    # Exit with error code if there were failures
    if stats['filings_failed'] > 0 or stats['facts_failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main() 