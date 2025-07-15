import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_URL
from config.config_tdnet_search import DB_URL_SEARCH
from src.database.utils.init_db import Base as MainBase, Disclosure as MainDisclosure
from src.scraper.tdnet_search.init_db_search import Base as SearchBase, Disclosure as SearchDisclosure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TDnetSearchMigrator:
    def __init__(self):
        # Load directories config
        directories_file = project_root / 'directories.json'
        with open(directories_file, 'r') as f:
            config = json.load(f)
            self.main_pdf_dir = Path(config['pdf_directory'])
            self.main_xbrl_dir = Path(config['xbrls_directory'])
            self.backup_dir = Path(config['backup_directory'])
        
        # Original search directories (for path updates)
        self.search_pdf_dir = Path(config.get('tdnet_search_pdf_directory', 'pdfs_tdnet_search'))
        self.search_xbrl_dir = Path(config.get('tdnet_search_xbrl_directory', 'xbrls_tdnet_search'))
        
        # Database connections
        self.main_engine = create_engine(DB_URL)
        self.search_engine = create_engine(DB_URL_SEARCH)
        
        # Sessions
        MainSession = sessionmaker(bind=self.main_engine)
        SearchSession = sessionmaker(bind=self.search_engine)
        self.main_session = MainSession()
        self.search_session = SearchSession()
    
    def check_xbrl_path_column(self):
        """Check if xbrl_path column exists in main database, add if not"""
        inspector = inspect(self.main_engine)
        columns = [col['name'] for col in inspector.get_columns('disclosures')]
        
        if 'xbrl_path' not in columns:
            logger.info("Adding xbrl_path column to main database...")
            with self.main_engine.connect() as conn:
                conn.execute(text("ALTER TABLE disclosures ADD COLUMN xbrl_path TEXT"))
                conn.commit()
            logger.info("xbrl_path column added successfully")
        else:
            logger.info("xbrl_path column already exists")
    
    def update_file_paths(self, search_disclosure):
        """Update file paths from search format to unified format"""
        updated_pdf_path = search_disclosure.pdf_path
        updated_xbrl_path = search_disclosure.xbrl_path
        
        # Update PDF path
        if search_disclosure.pdf_path:
            old_pdf_path = Path(search_disclosure.pdf_path)
            if old_pdf_path.is_absolute():
                # Keep absolute paths as is
                updated_pdf_path = search_disclosure.pdf_path
            else:
                # Convert relative path from search directory to unified directory
                if str(old_pdf_path).startswith(str(self.search_pdf_dir)):
                    # Path already includes search directory
                    relative_path = old_pdf_path.relative_to(self.search_pdf_dir)
                    updated_pdf_path = str(self.main_pdf_dir / relative_path)
                else:
                    # Path is relative to search directory
                    updated_pdf_path = str(self.main_pdf_dir / old_pdf_path)
        
        # Update XBRL path
        if search_disclosure.xbrl_path:
            old_xbrl_path = Path(search_disclosure.xbrl_path)
            if old_xbrl_path.is_absolute():
                # Keep absolute paths as is
                updated_xbrl_path = search_disclosure.xbrl_path
            else:
                # Convert relative path from search directory to unified directory
                if str(old_xbrl_path).startswith(str(self.search_xbrl_dir)):
                    # Path already includes search directory
                    relative_path = old_xbrl_path.relative_to(self.search_xbrl_dir)
                    updated_xbrl_path = str(self.main_xbrl_dir / relative_path)
                else:
                    # Path is relative to search directory
                    updated_xbrl_path = str(self.main_xbrl_dir / old_xbrl_path)
        
        return updated_pdf_path, updated_xbrl_path
    
    def migrate_data(self):
        """Migrate disclosure records from search database to main database"""
        logger.info("Starting data migration...")
        
        # Get all search disclosures
        search_disclosures = self.search_session.query(SearchDisclosure).all()
        logger.info(f"Found {len(search_disclosures)} records in tdnet_search database")
        
        migrated_count = 0
        skipped_count = 0
        error_count = 0
        
        for search_disclosure in search_disclosures:
            try:
                # Update file paths to point to unified directories
                updated_pdf_path, updated_xbrl_path = self.update_file_paths(search_disclosure)
                
                # Create new disclosure record for main database
                main_disclosure = MainDisclosure(
                    disclosure_date=search_disclosure.disclosure_date,
                    time=search_disclosure.time,
                    company_code=search_disclosure.company_code,
                    company_name=search_disclosure.company_name,
                    title=search_disclosure.title,
                    xbrl_url=None,  # Search database didn't have URLs
                    xbrl_path=updated_xbrl_path,  # Map xbrl_path from search
                    pdf_path=updated_pdf_path,
                    exchange=search_disclosure.exchange,
                    update_history=search_disclosure.update_history,
                    page_number=search_disclosure.page_number,
                    scraped_at=search_disclosure.scraped_at,
                    category=search_disclosure.category,
                    subcategory=search_disclosure.subcategory
                )
                
                # Check for duplicates based on key fields
                existing = self.main_session.query(MainDisclosure).filter_by(
                    disclosure_date=search_disclosure.disclosure_date,
                    company_code=search_disclosure.company_code,
                    title=search_disclosure.title
                ).first()
                
                if existing:
                    logger.debug(f"Duplicate found, skipping: {search_disclosure.company_code} - {search_disclosure.title}")
                    skipped_count += 1
                    continue
                
                # Add to main database
                self.main_session.add(main_disclosure)
                migrated_count += 1
                
                if migrated_count % 1000 == 0:
                    logger.info(f"Migrated {migrated_count} records...")
                    self.main_session.commit()
                
            except Exception as e:
                logger.error(f"Error migrating record {search_disclosure.id}: {e}")
                error_count += 1
                self.main_session.rollback()
        
        # Final commit
        try:
            self.main_session.commit()
            logger.info(f"Data migration completed: {migrated_count} migrated, {skipped_count} skipped, {error_count} errors")
        except Exception as e:
            logger.error(f"Final commit failed: {e}")
            self.main_session.rollback()
            raise
    
    def create_backup(self):
        """Create database backups before migration (databases only, no files)"""
        logger.info("Creating database backups before migration...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup main database
        try:
            from src.database.utils.backup_database import TDnetBackupManager
            main_backup = TDnetBackupManager()
            main_backup.timestamp = f"pre_migration_{timestamp}"
            main_backup.create_database_backup("full")
            logger.info("Main tdnet database backup created")
        except Exception as e:
            logger.warning(f"Failed to backup main tdnet database: {e}")
            logger.info("Continuing migration without main database backup...")
        
        # Backup search database (database only, not files)
        try:
            from src.scraper.tdnet_search.backup_tdnet_search import TDnetSearchBackupManager
            search_backup = TDnetSearchBackupManager()
            search_backup.timestamp = f"pre_migration_{timestamp}"
            search_backup.create_database_backup("full")  # Only database, not full backup with files
            logger.info("TDnet Search database backup created")
        except Exception as e:
            logger.error(f"Failed to backup TDnet Search database: {e}")
        
        logger.info("Database backups completed")
    
    def migrate(self, create_backup=True):
        """Run complete migration process"""
        logger.info("Starting TDnet Search migration to main database")
        try:
            if create_backup:
                self.create_backup()
            
            # 1. Ensure xbrl_path column exists
            self.check_xbrl_path_column()
            
            # 2. Migrate data (file paths will be updated to point to unified directories)
            self.migrate_data()
            
            logger.info("Migration completed successfully!")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            self.main_session.close()
            self.search_session.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate TDnet Search database to main TDnet database")
    parser.add_argument('--no-backup', action='store_true', 
                       help='Skip creating backup before migration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without actually doing it')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run logic
        return
    
    migrator = TDnetSearchMigrator()
    migrator.migrate(create_backup=not args.no_backup)

if __name__ == "__main__":
    main()