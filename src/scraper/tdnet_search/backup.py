import os
import subprocess
import shutil
import json
import zipfile
from datetime import datetime
from pathlib import Path
import logging
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_tdnet_search import DB_CONFIG_SEARCH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TDnetSearchBackupManager:
    def __init__(self):
        # Load directories from config
        directories_file = project_root / 'directories.json'
        with open(directories_file, 'r') as f:
            config = json.load(f)
            self.pdf_directory = Path(config.get('tdnet_search_pdf_directory', 'pdfs_tdnet_search'))
            self.xbrl_directory = Path(config.get('tdnet_search_xbrl_directory', 'xbrls_tdnet_search'))
            self.backup_dir = Path(config['backup_directory'])
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_database_backup(self, backup_type="full"):
        """
        Create PostgreSQL database backup using pg_dump
        
        Args:
            backup_type: "full" (schema + data) or "schema" (structure only)
        """
        backup_filename = f"tdnet_search_db_{backup_type}_{self.timestamp}.sql"
        backup_path = self.backup_dir / backup_filename
        
        # Set PostgreSQL password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = DB_CONFIG_SEARCH['password']
        
        # Build pg_dump command
        cmd = [
            'pg_dump',
            '-h', DB_CONFIG_SEARCH['host'],
            '-U', DB_CONFIG_SEARCH['user'],
            '-d', DB_CONFIG_SEARCH['database'],
            '--clean',  # Include DROP statements
            '--if-exists',  # Use IF EXISTS for DROP statements
            '--create',  # Include CREATE DATABASE statement
            '-f', str(backup_path)
        ]
        
        if backup_type == "schema":
            cmd.append('--schema-only')
        
        try:
            logger.info(f"Creating {backup_type} database backup for tdnet_search...")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            logger.info(f"TDnet Search database backup created: {backup_path}")
            return backup_path
        except subprocess.CalledProcessError as e:
            logger.error(f"TDnet Search database backup failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("pg_dump not found. Please install PostgreSQL client tools.")
            raise
    
    def create_pdf_backup(self, compress=True):
        """
        Create backup of PDF files
        
        Args:
            compress: If True, create compressed archive; if False, copy directory
        """
        if not self.pdf_directory.exists():
            logger.warning(f"PDF directory not found: {self.pdf_directory}")
            return None
        
        if compress:
            backup_filename = f"tdnet_search_pdfs_{self.timestamp}.zip"
            backup_path = self.backup_dir / backup_filename
            
            logger.info("Creating compressed PDF backup for tdnet_search...")
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.pdf_directory.rglob('*.pdf'):
                    # Preserve directory structure in archive
                    arcname = file_path.relative_to(self.pdf_directory.parent)
                    zipf.write(file_path, arcname)
                    
            logger.info(f"TDnet Search PDF backup created: {backup_path}")
        else:
            backup_dirname = f"tdnet_search_pdfs_{self.timestamp}"
            backup_path = self.backup_dir / backup_dirname
            
            logger.info("Creating PDF directory backup for tdnet_search...")
            shutil.copytree(self.pdf_directory, backup_path)
            logger.info(f"TDnet Search PDF backup created: {backup_path}")
        
        return backup_path
    
    def create_xbrl_backup(self, compress=True):
        """
        Create backup of XBRL files
        
        Args:
            compress: If True, create compressed archive; if False, copy directory
        """
        if not self.xbrl_directory.exists():
            logger.warning(f"XBRL directory not found: {self.xbrl_directory}")
            return None
        
        if compress:
            backup_filename = f"tdnet_search_xbrls_{self.timestamp}.zip"
            backup_path = self.backup_dir / backup_filename
            
            logger.info("Creating compressed XBRL backup for tdnet_search...")
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.xbrl_directory.rglob('*.xbrl'):
                    # Preserve directory structure in archive
                    arcname = file_path.relative_to(self.xbrl_directory.parent)
                    zipf.write(file_path, arcname)
                    
            logger.info(f"TDnet Search XBRL backup created: {backup_path}")
        else:
            backup_dirname = f"tdnet_search_xbrls_{self.timestamp}"
            backup_path = self.backup_dir / backup_dirname
            
            logger.info("Creating XBRL directory backup for tdnet_search...")
            shutil.copytree(self.xbrl_directory, backup_path)
            logger.info(f"TDnet Search XBRL backup created: {backup_path}")
        
        return backup_path
    
    def create_full_backup(self, compress_files=True):
        """Create complete backup of database, PDFs, XBRLs, and configuration"""
        logger.info(f"Starting full TDnet Search backup at {datetime.now()}")
        
        backup_summary = {
            'timestamp': self.timestamp,
            'database_backup': None,
            'schema_backup': None,
            'pdf_backup': None,
            'xbrl_backup': None,
            'errors': []
        }
        
        try:
            # Database backups
            backup_summary['database_backup'] = str(self.create_database_backup("full"))
            backup_summary['schema_backup'] = str(self.create_database_backup("schema"))
            
            # PDF backup
            backup_summary['pdf_backup'] = str(self.create_pdf_backup(compress_files))
            
            # XBRL backup
            backup_summary['xbrl_backup'] = str(self.create_xbrl_backup(compress_files))
            
        except Exception as e:
            backup_summary['errors'].append(str(e))
            logger.error(f"TDnet Search backup error: {e}")
        
        # Save backup summary
        summary_file = self.backup_dir / f"tdnet_search_backup_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(backup_summary, f, indent=2)
        
        logger.info(f"Full TDnet Search backup completed. Summary: {summary_file}")
        return backup_summary
    
    def list_backups(self):
        """List all available TDnet Search backups"""
        backups = {
            'database': list(self.backup_dir.glob("tdnet_search_db_*.sql")),
            'pdfs': list(self.backup_dir.glob("tdnet_search_pdfs_*")),
            'xbrls': list(self.backup_dir.glob("tdnet_search_xbrls_*")),
            'summaries': list(self.backup_dir.glob("tdnet_search_backup_summary_*.json"))
        }
        
        for backup_type, files in backups.items():
            logger.info(f"TDnet Search {backup_type.title()} backups:")
            for file in sorted(files, reverse=True):  # Most recent first
                size = file.stat().st_size if file.is_file() else sum(f.stat().st_size for f in file.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                logger.info(f"  {file.name} ({size_mb:.1f} MB)")
        
        return backups
    
    def restore_database(self, backup_file):
        """
        Restore database from backup file
        
        Args:
            backup_file: Path to SQL backup file
        """
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        # Set PostgreSQL password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = DB_CONFIG_SEARCH['password']
        
        # Build psql command
        cmd = [
            'psql',
            '-h', DB_CONFIG_SEARCH['host'],
            '-U', DB_CONFIG_SEARCH['user'],
            '-d', 'postgres',  # Connect to postgres db first
            '-f', str(backup_path)
        ]
        
        try:
            logger.info(f"Restoring TDnet Search database from: {backup_path}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            logger.info("TDnet Search database restore completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"TDnet Search database restore failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("psql not found. Please install PostgreSQL client tools.")
            raise

    def restore_pdfs(self, backup_file):
        """
        Restore PDFs from backup file (zip or directory)
        
        Args:
            backup_file: Path to PDF backup (zip file or directory)
        """
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"PDF backup not found: {backup_file}")
        
        try:
            if backup_path.suffix == '.zip':
                logger.info(f"Extracting TDnet Search PDF backup from: {backup_path}")
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    # Extract to parent directory to preserve structure
                    zipf.extractall(self.pdf_directory.parent)
                logger.info(f"TDnet Search PDFs restored to: {self.pdf_directory}")
            else:
                # Directory backup
                logger.info(f"Copying TDnet Search PDF backup from: {backup_path}")
                if self.pdf_directory.exists():
                    shutil.rmtree(self.pdf_directory)
                shutil.copytree(backup_path, self.pdf_directory)
                logger.info(f"TDnet Search PDFs restored to: {self.pdf_directory}")
                
        except Exception as e:
            logger.error(f"TDnet Search PDF restore failed: {e}")
            raise

    def restore_xbrls(self, backup_file):
        """
        Restore XBRLs from backup file (zip or directory)
        
        Args:
            backup_file: Path to XBRL backup (zip file or directory)
        """
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"XBRL backup not found: {backup_file}")
        
        try:
            if backup_path.suffix == '.zip':
                logger.info(f"Extracting TDnet Search XBRL backup from: {backup_path}")
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    # Extract to parent directory to preserve structure
                    zipf.extractall(self.xbrl_directory.parent)
                logger.info(f"TDnet Search XBRLs restored to: {self.xbrl_directory}")
            else:
                # Directory backup
                logger.info(f"Copying TDnet Search XBRL backup from: {backup_path}")
                if self.xbrl_directory.exists():
                    shutil.rmtree(self.xbrl_directory)
                shutil.copytree(backup_path, self.xbrl_directory)
                logger.info(f"TDnet Search XBRLs restored to: {self.xbrl_directory}")
                
        except Exception as e:
            logger.error(f"TDnet Search XBRL restore failed: {e}")
            raise

    def restore_full_backup(self, timestamp):
        """
        Restore complete backup from timestamp
        
        Args:
            timestamp: Backup timestamp (e.g., "20250527_133852")
        """
        logger.info(f"Starting full TDnet Search restore from timestamp: {timestamp}")
        
        # Find backup files
        db_backup = self.backup_dir / f"tdnet_search_db_full_{timestamp}.sql"
        pdf_backup_zip = self.backup_dir / f"tdnet_search_pdfs_{timestamp}.zip"
        pdf_backup_dir = self.backup_dir / f"tdnet_search_pdfs_{timestamp}"
        xbrl_backup_zip = self.backup_dir / f"tdnet_search_xbrls_{timestamp}.zip"
        xbrl_backup_dir = self.backup_dir / f"tdnet_search_xbrls_{timestamp}"
        
        restore_summary = {
            'timestamp': timestamp,
            'database_restored': False,
            'pdfs_restored': False,
            'xbrls_restored': False,
            'errors': []
        }
        
        try:
            # Restore database
            if db_backup.exists():
                self.restore_database(db_backup)
                restore_summary['database_restored'] = True
            else:
                restore_summary['errors'].append(f"Database backup not found: {db_backup}")
            
            # Restore PDFs (try zip first, then directory)
            if pdf_backup_zip.exists():
                self.restore_pdfs(pdf_backup_zip)
                restore_summary['pdfs_restored'] = True
            elif pdf_backup_dir.exists():
                self.restore_pdfs(pdf_backup_dir)
                restore_summary['pdfs_restored'] = True
            else:
                restore_summary['errors'].append(f"PDF backup not found: {pdf_backup_zip} or {pdf_backup_dir}")
            
            # Restore XBRLs (try zip first, then directory)
            if xbrl_backup_zip.exists():
                self.restore_xbrls(xbrl_backup_zip)
                restore_summary['xbrls_restored'] = True
            elif xbrl_backup_dir.exists():
                self.restore_xbrls(xbrl_backup_dir)
                restore_summary['xbrls_restored'] = True
            else:
                restore_summary['errors'].append(f"XBRL backup not found: {xbrl_backup_zip} or {xbrl_backup_dir}")
        
        except Exception as e:
            restore_summary['errors'].append(str(e))
            logger.error(f"TDnet Search restore error: {e}")
        
        logger.info(f"Full TDnet Search restore completed. Summary: {restore_summary}")
        return restore_summary

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TDnet Search Database Backup Manager")
    parser.add_argument('action', choices=['backup', 'list', 'restore'], 
                       help='Action to perform')
    parser.add_argument('--type', choices=['full', 'database', 'pdfs', 'xbrls'], 
                       default='full', help='Backup/restore type')
    parser.add_argument('--compress', action='store_true', default=True,
                       help='Compress file backups (default: True)')
    parser.add_argument('--file', help='Backup file for restore operation')
    parser.add_argument('--timestamp', help='Backup timestamp for full restore (e.g., 20250527_133852)')
    
    args = parser.parse_args()
    
    backup_manager = TDnetSearchBackupManager()
    
    if args.action == 'backup':
        if args.type == 'full':
            backup_manager.create_full_backup(args.compress)
        elif args.type == 'database':
            backup_manager.create_database_backup("full")
        elif args.type == 'pdfs':
            backup_manager.create_pdf_backup(args.compress)
        elif args.type == 'xbrls':
            backup_manager.create_xbrl_backup(args.compress)
    
    elif args.action == 'list':
        backup_manager.list_backups()
    
    elif args.action == 'restore':
        if args.type == 'full':
            if not args.timestamp:
                logger.error("--timestamp argument required for full restore (e.g., 20250527_133852)")
                return
            backup_manager.restore_full_backup(args.timestamp)
        elif args.type == 'database':
            if not args.file:
                logger.error("--file argument required for database restore")
                return
            backup_manager.restore_database(args.file)
        elif args.type == 'pdfs':
            if not args.file:
                logger.error("--file argument required for PDF restore")
                return
            backup_manager.restore_pdfs(args.file)
        elif args.type == 'xbrls':
            if not args.file:
                logger.error("--file argument required for XBRL restore")
                return
            backup_manager.restore_xbrls(args.file)

if __name__ == "__main__":
    main() 