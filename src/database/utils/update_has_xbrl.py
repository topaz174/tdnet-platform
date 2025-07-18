#!/usr/bin/env python3
"""
Update has_xbrl field for all existing disclosures.

This script:
1. Reads all disclosures from the database
2. Derives the expected XBRL path using path_derivation.py
3. Checks if the file actually exists
4. Updates has_xbrl to True if the file exists, False otherwise

This is more reliable than checking xbrl_url or xbrl_path fields since those
can be null even for disclosures that actually have XBRL files.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import unified config and path derivation
from config.config import DB_URL
from src.shared.utils.path_derivation import derive_xbrl_path, check_file_exists

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)


def get_all_disclosures(session):
    """Get all disclosures from the database."""
    query = text("""
        SELECT 
            id,
            company_code,
            disclosure_date,
            time,
            title,
            has_xbrl
        FROM disclosures
        ORDER BY disclosure_date DESC, time DESC
    """)
    
    result = session.execute(query)
    return result.fetchall()


def update_has_xbrl_for_disclosure(session, disclosure_id: int, has_xbrl: bool):
    """Update has_xbrl field for a specific disclosure."""
    try:
        session.execute(
            text("UPDATE disclosures SET has_xbrl = :has_xbrl WHERE id = :id"),
            {'has_xbrl': has_xbrl, 'id': disclosure_id}
        )
        return True
    except Exception as e:
        logger.error(f"Error updating disclosure {disclosure_id}: {e}")
        return False


def update_all_has_xbrl(dry_run: bool = False, limit: Optional[int] = None):
    """
    Update has_xbrl for all disclosures based on file existence.
    
    Args:
        dry_run (bool): If True, only show what would be updated without making changes
        limit (int): Limit number of disclosures to process (for testing)
    """
    session = Session()
    
    try:
        # Get all disclosures
        disclosures = get_all_disclosures(session)
        
        if limit:
            disclosures = disclosures[:limit]
            logger.info(f"Processing limited to {limit} disclosures")
        
        logger.info(f"Processing {len(disclosures)} disclosures...")
        
        updated_count = 0
        unchanged_count = 0
        error_count = 0
        
        for i, disclosure in enumerate(disclosures, 1):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(disclosures)} disclosures...")
            
            try:
                # Derive expected XBRL path
                xbrl_path = derive_xbrl_path(
                    company_code=disclosure.company_code,
                    disclosure_date=disclosure.disclosure_date,
                    disclosure_time=disclosure.time,
                    title=disclosure.title
                )
                
                # Check if file exists
                file_exists = check_file_exists(xbrl_path)
                
                # Determine what has_xbrl should be
                should_have_xbrl = file_exists
                current_has_xbrl = disclosure.has_xbrl
                
                # Update if different
                if should_have_xbrl != current_has_xbrl:
                    if not dry_run:
                        success = update_has_xbrl_for_disclosure(session, disclosure.id, should_have_xbrl)
                        if success:
                            updated_count += 1
                            logger.info(f"Updated disclosure {disclosure.id}: has_xbrl {current_has_xbrl} → {should_have_xbrl}")
                            if file_exists:
                                logger.info(f"  File exists: {xbrl_path}")
                            else:
                                logger.info(f"  File missing: {xbrl_path}")
                        else:
                            error_count += 1
                    else:
                        # Dry run - just log what would be changed
                        logger.info(f"Would update disclosure {disclosure.id}: has_xbrl {current_has_xbrl} → {should_have_xbrl}")
                        if file_exists:
                            logger.info(f"  File exists: {xbrl_path}")
                        else:
                            logger.info(f"  File missing: {xbrl_path}")
                        updated_count += 1
                else:
                    unchanged_count += 1
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing disclosure {disclosure.id}: {e}")
                continue
        
        # Commit changes if not dry run
        if not dry_run:
            session.commit()
            logger.info(f"Committed changes to database")
        
        # Summary
        logger.info(f"\nUpdate Summary:")
        logger.info(f"  Total processed: {len(disclosures)}")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Unchanged: {unchanged_count}")
        logger.info(f"  Errors: {error_count}")
        
        if dry_run:
            logger.info(f"  DRY RUN - No changes were made to the database")
        
        return updated_count, unchanged_count, error_count
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        return 0, 0, 1
    finally:
        session.close()


def get_has_xbrl_statistics():
    """Get statistics about current has_xbrl values."""
    session = Session()
    
    try:
        # Get counts
        total_query = text("SELECT COUNT(*) FROM disclosures")
        has_xbrl_query = text("SELECT COUNT(*) FROM disclosures WHERE has_xbrl = true")
        no_xbrl_query = text("SELECT COUNT(*) FROM disclosures WHERE has_xbrl = false")
        
        total = session.execute(total_query).scalar()
        has_xbrl_count = session.execute(has_xbrl_query).scalar()
        no_xbrl_count = session.execute(no_xbrl_query).scalar()
        
        logger.info(f"Current has_xbrl Statistics:")
        logger.info(f"  Total disclosures: {total}")
        if total:
            logger.info(f"  has_xbrl = true: {has_xbrl_count} ({has_xbrl_count/total*100:.1f}%)")
            logger.info(f"  has_xbrl = false: {no_xbrl_count} ({no_xbrl_count/total*100:.1f}%)")
        else:
            logger.info(f"  has_xbrl = true: {has_xbrl_count}")
            logger.info(f"  has_xbrl = false: {no_xbrl_count}")
        
        return total, has_xbrl_count, no_xbrl_count
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return 0, 0, 0
    finally:
        session.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update has_xbrl field for all disclosures")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be updated without making changes")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of disclosures to process (for testing)")
    parser.add_argument("--stats", action="store_true",
                       help="Show current has_xbrl statistics only")
    
    args = parser.parse_args()
    
    if args.stats:
        get_has_xbrl_statistics()
        return
    
    logger.info("Starting has_xbrl update process...")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made to the database")
    
    # Show current statistics
    get_has_xbrl_statistics()
    
    # Update has_xbrl
    updated, unchanged, errors = update_all_has_xbrl(
        dry_run=args.dry_run,
        limit=args.limit
    )
    
    # Show final statistics
    logger.info("\nFinal Statistics:")
    get_has_xbrl_statistics()
    
    if args.dry_run:
        logger.info("\nTo apply changes, run without --dry-run")


if __name__ == "__main__":
    main() 