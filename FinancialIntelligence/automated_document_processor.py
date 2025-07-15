#!/usr/bin/env python3
"""
Automated document processing pipeline with monitoring and incremental updates
"""

import asyncio
import schedule
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime, timedelta

class DocumentWatcher(FileSystemEventHandler):
    """Watch for new PDF files and queue them for processing"""
    
    def __init__(self, processor):
        self.processor = processor
        self.pending_files = set()
    
    def on_created(self, event):
        if event.is_file and event.src_path.endswith('.pdf'):
            print(f"New PDF detected: {event.src_path}")
            self.pending_files.add(event.src_path)
    
    def process_pending(self):
        """Process any pending files"""
        if self.pending_files:
            print(f"Processing {len(self.pending_files)} pending files...")
            for pdf_path in list(self.pending_files):
                try:
                    # Add to database and process
                    self.processor.add_new_document(pdf_path)
                    self.pending_files.remove(pdf_path)
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")

class AutomatedProcessor:
    def __init__(self):
        self.processor = EnhancedEmbeddingProcessor()
        
    def daily_maintenance(self):
        """Daily maintenance: process any missed documents"""
        print(f"Starting daily maintenance at {datetime.now()}")
        
        # Process any documents added to DB but not embedded
        self.processor.process_batch(force_reprocess=False)
        
        # Clean up old cache entries, update statistics, etc.
        self.cleanup_and_stats()
    
    def weekly_full_scan(self):
        """Weekly: check for any file changes and reprocess if needed"""
        print(f"Starting weekly full scan at {datetime.now()}")
        
        # This will check content hashes and reprocess changed files
        self.processor.process_batch(force_reprocess=False)
    
    def cleanup_and_stats(self):
        """Cleanup and generate processing statistics"""
        with self.processor.Session() as session:
            # Get processing stats
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_docs,
                    COUNT(dense_embedding) as embedded_docs,
                    COUNT(CASE WHEN dense_embedding IS NULL THEN 1 END) as pending_docs
                FROM disclosures
            """)).first()
            
            print(f"Processing stats: {result.embedded_docs}/{result.total_docs} documents embedded, {result.pending_docs} pending")

def setup_automation():
    """Set up automated processing schedules"""
    processor = AutomatedProcessor()
    
    # Schedule daily processing at 2 AM
    schedule.every().day.at("02:00").do(processor.daily_maintenance)
    
    # Schedule weekly full scan on Sundays at 3 AM  
    schedule.every().sunday.at("03:00").do(processor.weekly_full_scan)
    
    # Set up file watcher for immediate processing
    watcher = DocumentWatcher(processor)
    observer = Observer()
    observer.schedule(watcher, path="/path/to/pdf/directory", recursive=True)
    observer.start()
    
    print("Automated processing pipeline started...")
    
    try:
        while True:
            schedule.run_pending()
            watcher.process_pending()  # Process any new files immediately
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    setup_automation() 