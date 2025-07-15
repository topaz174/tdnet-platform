import schedule
import time
import logging
from backup_database import TDnetBackupManager
from datetime import datetime, timedelta
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackupScheduler:
    def __init__(self):
        self.backup_manager = TDnetBackupManager()
        self.max_backups = {
            'daily': 7,    # Keep 7 daily backups
            'weekly': 1,   # Keep 4 weekly backups
            'monthly': 12  # Keep 12 monthly backups
        }
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        backup_dir = self.backup_manager.backup_dir
        
        # Get all backup files sorted by date
        db_backups = sorted(backup_dir.glob("tdnet_db_full_*.sql"), reverse=True)
        pdf_backups = sorted(backup_dir.glob("tdnet_pdfs_*.zip"), reverse=True)
        
        # Keep only the most recent backups
        for backup_list in [db_backups, pdf_backups]:
            if len(backup_list) > self.max_backups['daily']:
                for old_backup in backup_list[self.max_backups['daily']:]:
                    try:
                        old_backup.unlink()
                        logger.info(f"Removed old backup: {old_backup.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove {old_backup.name}: {e}")
    
    def daily_backup(self):
        """Perform daily backup"""
        logger.info("Starting daily backup...")
        # self.backup_manager.create_full_backup(compress_pdfs=True)
        self.backup_manager.create_database_backup("full")
        self.cleanup_old_backups()
    
    def weekly_backup(self):
        """Perform weekly backup (more comprehensive)"""
        logger.info("Starting weekly backup...")
        # Create uncompressed PDF backup for faster access
        self.backup_manager.create_pdf_backup(compress=True)
        self.backup_manager.create_database_backup("full")
    
    def run_scheduler(self):
        """Run the backup scheduler"""
        # Schedule daily backups at 2 AM
        schedule.every().day.at("02:00").do(self.daily_backup)
        
        # Schedule weekly backups on Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(self.weekly_backup)
        
        logger.info("Backup scheduler started. Daily: 2:00 AM, Weekly: Sunday 3:00 AM")
        logger.info(f"Backup directory: {self.backup_manager.backup_dir}")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    scheduler = BackupScheduler()
    scheduler.run_scheduler() 