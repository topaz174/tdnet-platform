#!/usr/bin/env python3
"""
Unified Extraction Pipeline for Financial Intelligence

This pipeline connects to the PostgreSQL database and processes all rows in the
disclosures table, using intelligent XBRL vs PDF selection logic with parallel processing.

Key Features:
- Database-driven processing: queries disclosures table directly
- Smart file selection: XBRL qualitative.htm preferred, PDF fallback
- Parallel processing: based on patterns from parallel_pdf_extraction_pipeline.py
- Test mode: process limited number of days/rows for validation
- Progress tracking and error handling

Usage:
    python unified_extraction_pipeline.py --test-days 7 --workers 16
    python unified_extraction_pipeline.py --full-pipeline --workers 32
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import time
import zipfile
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import multiprocessing as mp
import threading
from queue import Queue
import psutil

# Environment variables
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    DOTENV_AVAILABLE = False

# Database
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import uuid

# Import processors from existing pipelines
sys.path.append(os.path.dirname(__file__))
try:
    from parallel_pdf_extraction_pipeline import (
        AsyncPDFProcessor, ParallelProcessingConfig, ProcessingStats
    )
    from xbrl_qualitative_extractor import (
        XBRLProcessor as BaseXBRLProcessor, XBRLChunk, parse_zip_meta, sha256_file
    )
    from pdf_extraction_pipeline import DocumentChunk
except ImportError as e:
    print(f"Error importing existing pipelines: {e}")
    print("Please ensure parallel_pdf_extraction_pipeline.py and xbrl_qualitative_extractor.py are available")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def load_database_config(env_file_path: str = None) -> str:
    """
    Load database configuration from .env file or environment variables.
    
    Args:
        env_file_path: Path to .env file (default: searches parent directories)
    
    Returns:
        PostgreSQL connection string (PG_DSN format)
    """
    # Try to load .env file
    if DOTENV_AVAILABLE:
        if env_file_path:
            env_path = Path(env_file_path)
        else:
            # Search for .env file in current directory and parent directories
            current_dir = Path(__file__).parent
            env_path = None
            for parent in [current_dir] + list(current_dir.parents):
                candidate = parent / '.env'
                if candidate.exists():
                    env_path = candidate
                    break
        
        if env_path and env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.info("No .env file found, using system environment variables")
    else:
        logger.info("python-dotenv not available, using system environment variables only")
    
    # Try to get PG_DSN directly first
    pg_dsn = os.getenv('PG_DSN')
    if pg_dsn and pg_dsn.strip():
        logger.info("Using PG_DSN from environment")
        return pg_dsn.strip()
    
    # Build PG_DSN from individual components
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'tdnet')
    
    # Remove quotes from password if present
    if db_password.startswith("'") and db_password.endswith("'"):
        db_password = db_password[1:-1]
    elif db_password.startswith('"') and db_password.endswith('"'):
        db_password = db_password[1:-1]
    
    if not db_password:
        logger.warning("No database password found in environment variables")
    
    # Construct PG_DSN
    pg_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Built PG_DSN from components: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}")
    
    return pg_dsn

@dataclass
class DatabaseRow:
    """Represents a row from the disclosures table"""
    id: int
    company_code: str
    company_name: str
    disclosure_date: date
    xbrl_path: Optional[str]
    pdf_path: Optional[str]
    title: Optional[str] = None
    category: Optional[str] = None
    # Tracking fields
    extraction_status: str = 'pending'
    extraction_method: Optional[str] = None
    extraction_date: Optional[datetime] = None
    extraction_error: Optional[str] = None
    chunks_extracted: int = 0
    extraction_duration: float = 0.0
    extraction_file_path: Optional[str] = None

@dataclass
class ExtractionResult:
    """Result of processing a single database row"""
    disclosure_id: int
    extraction_method: str  # 'xbrl', 'pdf', 'failed'
    chunks_count: int
    processing_time: float
    file_path: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class UnifiedProcessingConfig:
    """Configuration for unified processing pipeline"""
    # Database settings
    pg_dsn: str
    
    # Processing settings
    max_workers: int = 16
    max_concurrent_files: int = 8
    batch_size: int = 4
    test_mode: bool = False
    test_days: int = 7
    max_test_rows: int = 100
    
    # File processing settings
    prefer_xbrl: bool = True
    require_qualitative_htm: bool = True
    
    # Tracking and resume settings
    resume_mode: bool = False
    retry_failed: bool = False
    max_retries: int = 3
    force_reprocess: bool = False
    skip_completed: bool = True
    update_status: bool = True
    
    # Output settings
    output_dir: str = "unified_output"
    save_chunks: bool = True
    save_to_database: bool = True
    save_to_files: bool = False
    
    # Parallel processing (inherited from PDF pipeline)
    use_gpu_ocr: bool = False
    gpu_batch_size: int = 8
    memory_limit_gb: int = 16
    enable_progress_monitoring: bool = True
    ocr_backend: str = "tesseract"
    smart_ocr: bool = True
    disable_ocr: bool = False

class DatabaseManager:
    """Manages database connections and queries for the unified pipeline"""
    
    def __init__(self, pg_dsn: str):
        self.pg_dsn = pg_dsn
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.pg_dsn)
            self.conn.autocommit = False
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_disclosure_rows(self, test_mode: bool = False, test_days: int = 7, max_rows: int = 100) -> List[DatabaseRow]:
        """
        Get disclosure rows from database with optional test mode filtering
        
        Args:
            test_mode: If True, limit to recent rows for testing
            test_days: Number of recent days to include in test mode
            max_rows: Maximum rows to return in test mode
        """
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        base_query = """
        SELECT 
            id,
            company_code,
            company_name,
            disclosure_date,
            xbrl_path,
            pdf_path,
            title,
            category,
            COALESCE(extraction_status, 'pending') as extraction_status,
            extraction_method,
            extraction_date,
            extraction_error,
            COALESCE(chunks_extracted, 0) as chunks_extracted,
            COALESCE(extraction_duration, 0.0) as extraction_duration,
            extraction_file_path
        FROM disclosures
        WHERE (xbrl_path IS NOT NULL AND xbrl_path != '') 
           OR (pdf_path IS NOT NULL AND pdf_path != '')
        """
        
        if test_mode:
            # Test mode: recent data only
            cutoff_date = datetime.now().date() - timedelta(days=test_days)
            query = base_query + """
            AND disclosure_date >= %s
            ORDER BY disclosure_date DESC
            LIMIT %s
            """
            params = (cutoff_date, max_rows)
            logger.info(f"Test mode: fetching up to {max_rows} rows from last {test_days} days (since {cutoff_date})")
        else:
            # Full mode: all data, ordered by date (newest first)
            query = base_query + " ORDER BY disclosure_date DESC"
            params = ()
            logger.info("Full mode: fetching all available disclosure rows")
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            
            # Convert to DatabaseRow objects
            disclosure_rows = []
            for row in rows:
                disclosure_rows.append(DatabaseRow(
                    id=row['id'],
                    company_code=row['company_code'] or 'UNKNOWN',
                    company_name=row['company_name'] or 'UNKNOWN',
                    disclosure_date=row['disclosure_date'],
                    xbrl_path=row['xbrl_path'],
                    pdf_path=row['pdf_path'],
                    title=row['title'],
                    category=row['category'],
                    extraction_status=row['extraction_status'],
                    extraction_method=row['extraction_method'],
                    extraction_date=row['extraction_date'],
                    extraction_error=row['extraction_error'],
                    chunks_extracted=row['chunks_extracted'],
                    extraction_duration=row['extraction_duration'],
                    extraction_file_path=row['extraction_file_path']
                ))
            
            logger.info(f"Retrieved {len(disclosure_rows)} disclosure rows from database")
            return disclosure_rows
            
        except Exception as e:
            logger.error(f"Error querying disclosure rows: {e}")
            return []
    
    def get_disclosure_rows_with_tracking(self, config: 'UnifiedProcessingConfig') -> List[DatabaseRow]:
        """
        Get disclosure rows with intelligent filtering based on tracking configuration
        
        Args:
            config: Processing configuration with tracking settings
        """
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        # Build filtering conditions based on configuration
        conditions = []
        params = []
        
        # Base condition: must have files
        conditions.append("((xbrl_path IS NOT NULL AND xbrl_path != '') OR (pdf_path IS NOT NULL AND pdf_path != ''))")
        
        # Resume mode: only process unprocessed files
        if config.resume_mode:
            if config.retry_failed:
                # Include failed extractions for retry
                conditions.append("(COALESCE(extraction_status, 'pending') IN ('pending', 'failed', 'retry'))")
            else:
                # Only pending files
                conditions.append("(COALESCE(extraction_status, 'pending') = 'pending')")
        elif config.skip_completed and not config.force_reprocess:
            # Skip completed files unless force reprocess
            conditions.append("(COALESCE(extraction_status, 'pending') != 'completed')")
        
        # Test mode filtering
        if config.test_mode:
            cutoff_date = datetime.now().date() - timedelta(days=config.test_days)
            conditions.append("disclosure_date >= %s")
            params.append(cutoff_date)
        
        # Build query
        base_query = """
        SELECT 
            id,
            company_code,
            company_name,
            disclosure_date,
            xbrl_path,
            pdf_path,
            title,
            category,
            COALESCE(extraction_status, 'pending') as extraction_status,
            extraction_method,
            extraction_date,
            extraction_error,
            COALESCE(chunks_extracted, 0) as chunks_extracted,
            COALESCE(extraction_duration, 0.0) as extraction_duration,
            extraction_file_path
        FROM disclosures
        WHERE """ + " AND ".join(conditions) + """
        ORDER BY disclosure_date DESC
        """
        
        # Apply row limit for test mode
        if config.test_mode:
            base_query += " LIMIT %s"
            params.append(config.max_test_rows)
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(base_query, params)
                rows = cur.fetchall()
            
            # Convert to DatabaseRow objects
            disclosure_rows = []
            for row in rows:
                disclosure_rows.append(DatabaseRow(
                    id=row['id'],
                    company_code=row['company_code'] or 'UNKNOWN',
                    company_name=row['company_name'] or 'UNKNOWN',
                    disclosure_date=row['disclosure_date'],
                    xbrl_path=row['xbrl_path'],
                    pdf_path=row['pdf_path'],
                    title=row['title'],
                    category=row['category'],
                    extraction_status=row['extraction_status'],
                    extraction_method=row['extraction_method'],
                    extraction_date=row['extraction_date'],
                    extraction_error=row['extraction_error'],
                    chunks_extracted=row['chunks_extracted'],
                    extraction_duration=row['extraction_duration'],
                    extraction_file_path=row['extraction_file_path']
                ))
            
            logger.info(f"Retrieved {len(disclosure_rows)} disclosure rows with tracking filters")
            return disclosure_rows
            
        except Exception as e:
            logger.error(f"Error querying disclosure rows with tracking: {e}")
            return []
    
    def update_extraction_status(self, disclosure_id: int, status: str, 
                                method: str = None, file_path: str = None, 
                                error_message: str = None, chunks_count: int = 0, 
                                duration: float = 0.0, metadata: Dict[str, Any] = None) -> bool:
        """
        Update extraction status for a disclosure
        
        Args:
            disclosure_id: ID of the disclosure
            status: New status ('processing', 'completed', 'failed', 'retry')
            method: Extraction method used ('xbrl', 'pdf', 'pdf_fallback')
            file_path: Path to the file that was processed
            error_message: Error message if failed
            chunks_count: Number of chunks extracted
            duration: Processing duration in seconds
            metadata: Additional metadata
        """
        if not self.conn:
            logger.error("Database not connected")
            return False
        
        try:
            update_sql = """
            UPDATE disclosures 
            SET 
                extraction_status = %s,
                extraction_method = COALESCE(%s, extraction_method),
                extraction_date = CASE WHEN %s = 'completed' THEN NOW() ELSE extraction_date END,
                extraction_error = %s,
                chunks_extracted = CASE WHEN %s = 'completed' THEN %s ELSE chunks_extracted END,
                extraction_duration = CASE WHEN %s = 'completed' THEN %s ELSE extraction_duration END,
                extraction_file_path = COALESCE(%s, extraction_file_path),
                extraction_metadata = COALESCE(%s::jsonb, extraction_metadata)
            WHERE id = %s
            """
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self.conn.cursor() as cur:
                cur.execute(update_sql, (
                    status, method, status, error_message, 
                    status, chunks_count, status, duration, 
                    file_path, metadata_json, disclosure_id
                ))
                
                if cur.rowcount == 1:
                    self.conn.commit()
                    logger.debug(f"Updated extraction status for disclosure {disclosure_id}: {status}")
                    return True
                else:
                    logger.warning(f"No rows updated for disclosure {disclosure_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating extraction status for disclosure {disclosure_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        if not self.conn:
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Overall statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_disclosures,
                        COUNT(CASE WHEN xbrl_path IS NOT NULL AND xbrl_path != '' THEN 1 END) as has_xbrl,
                        COUNT(CASE WHEN pdf_path IS NOT NULL AND pdf_path != '' THEN 1 END) as has_pdf,
                        COUNT(CASE WHEN (xbrl_path IS NOT NULL AND xbrl_path != '') 
                                    OR (pdf_path IS NOT NULL AND pdf_path != '') THEN 1 END) as processable,
                        COUNT(CASE WHEN COALESCE(extraction_status, 'pending') = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN COALESCE(extraction_status, 'pending') = 'failed' THEN 1 END) as failed,
                        COUNT(CASE WHEN COALESCE(extraction_status, 'pending') = 'pending' THEN 1 END) as pending,
                        COUNT(CASE WHEN COALESCE(extraction_status, 'pending') = 'processing' THEN 1 END) as processing,
                        SUM(COALESCE(chunks_extracted, 0)) as total_chunks,
                        AVG(COALESCE(extraction_duration, 0)) as avg_duration
                    FROM disclosures
                """)
                overall_stats = dict(cur.fetchone())
                
                # Status breakdown
                cur.execute("""
                    SELECT 
                        COALESCE(extraction_status, 'pending') as status,
                        COUNT(*) as count
                    FROM disclosures 
                    WHERE (xbrl_path IS NOT NULL AND xbrl_path != '') 
                       OR (pdf_path IS NOT NULL AND pdf_path != '')
                    GROUP BY extraction_status 
                    ORDER BY count DESC
                """)
                status_breakdown = {row['status']: row['count'] for row in cur.fetchall()}
                
                # Method breakdown
                cur.execute("""
                    SELECT 
                        extraction_method,
                        COUNT(*) as count,
                        SUM(chunks_extracted) as total_chunks,
                        AVG(extraction_duration) as avg_duration
                    FROM disclosures 
                    WHERE extraction_method IS NOT NULL
                    GROUP BY extraction_method 
                    ORDER BY count DESC
                """)
                method_breakdown = {
                    row['extraction_method']: {
                        'count': row['count'],
                        'total_chunks': row['total_chunks'] or 0,
                        'avg_duration': float(row['avg_duration']) if row['avg_duration'] else 0.0
                    } for row in cur.fetchall()
                }
                
                # Recent activity
                cur.execute("""
                    SELECT 
                        DATE(extraction_date) as date,
                        COUNT(*) as completed_today
                    FROM disclosures 
                    WHERE extraction_date >= CURRENT_DATE - INTERVAL '7 days'
                    AND extraction_status = 'completed'
                    GROUP BY DATE(extraction_date)
                    ORDER BY date DESC
                """)
                recent_activity = {str(row['date']): row['completed_today'] for row in cur.fetchall()}
                
                return {
                    'overall': overall_stats,
                    'status_breakdown': status_breakdown,
                    'method_breakdown': method_breakdown,
                    'recent_activity': recent_activity,
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting extraction statistics: {e}")
            return {}
    
    def mark_processing_start(self, disclosure_id: int) -> bool:
        """Mark a disclosure as currently being processed"""
        return self.update_extraction_status(disclosure_id, 'processing')
    
    def mark_processing_complete(self, disclosure_id: int, method: str, file_path: str, 
                               chunks_count: int, duration: float, metadata: Dict[str, Any] = None) -> bool:
        """Mark a disclosure as successfully processed"""
        return self.update_extraction_status(
            disclosure_id, 'completed', method, file_path, 
            None, chunks_count, duration, metadata
        )
    
    def mark_processing_failed(self, disclosure_id: int, error_message: str, 
                              method: str = None, file_path: str = None) -> bool:
        """Mark a disclosure as failed"""
        return self.update_extraction_status(
            disclosure_id, 'failed', method, file_path, error_message
        )
    
    def reset_processing_status(self, disclosure_ids: List[int] = None) -> int:
        """
        Reset processing status for stuck 'processing' entries
        
        Args:
            disclosure_ids: Specific IDs to reset, or None for all stuck entries
            
        Returns:
            Number of rows reset
        """
        if not self.conn:
            return 0
        
        try:
            if disclosure_ids:
                # Reset specific IDs
                placeholders = ','.join(['%s'] * len(disclosure_ids))
                reset_sql = f"""
                UPDATE disclosures 
                SET extraction_status = 'retry' 
                WHERE id IN ({placeholders}) 
                AND extraction_status = 'processing'
                """
                params = disclosure_ids
            else:
                # Reset all stuck processing entries (older than 1 hour)
                reset_sql = """
                UPDATE disclosures 
                SET extraction_status = 'retry' 
                WHERE extraction_status = 'processing' 
                AND (extraction_date IS NULL OR extraction_date < NOW() - INTERVAL '1 hour')
                """
                params = ()
            
            with self.conn.cursor() as cur:
                cur.execute(reset_sql, params)
                reset_count = cur.rowcount
                self.conn.commit()
                
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} stuck processing entries to 'retry' status")
                
                return reset_count
                
        except Exception as e:
            logger.error(f"Error resetting processing status: {e}")
            self.conn.rollback()
            return 0
    
    def insert_document_chunks(self, chunks: List[Dict[str, Any]], disclosure_id: int) -> bool:
        """
        Insert document chunks into the database
        
        Args:
            chunks: List of chunk dictionaries with all required fields
            disclosure_id: ID of the disclosure these chunks belong to
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn or not chunks:
            return False
        
        try:
            insert_sql = """
            INSERT INTO document_chunks (
                disclosure_id, chunk_index, content, content_type, section_code,
                heading_text, char_length, tokens, vectorize, is_numeric,
                disclosure_hash, source_file, page_number, metadata
            ) VALUES (
                %(disclosure_id)s, %(chunk_index)s, %(content)s, %(content_type)s, %(section_code)s,
                %(heading_text)s, %(char_length)s, %(tokens)s, %(vectorize)s, %(is_numeric)s,
                %(disclosure_hash)s, %(source_file)s, %(page_number)s, %(metadata)s
            )
            ON CONFLICT (disclosure_id, chunk_index) 
            DO UPDATE SET
                content = EXCLUDED.content,
                content_type = EXCLUDED.content_type,
                section_code = EXCLUDED.section_code,
                heading_text = EXCLUDED.heading_text,
                char_length = EXCLUDED.char_length,
                tokens = EXCLUDED.tokens,
                vectorize = EXCLUDED.vectorize,
                is_numeric = EXCLUDED.is_numeric,
                disclosure_hash = EXCLUDED.disclosure_hash,
                source_file = EXCLUDED.source_file,
                page_number = EXCLUDED.page_number,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """
            
            # Prepare chunks for insertion
            insert_data = []
            for chunk in chunks:
                # Handle metadata serialization
                metadata_json = chunk.get('metadata', {})
                if isinstance(metadata_json, dict):
                    metadata_json = json.dumps(metadata_json)
                
                insert_data.append({
                    'disclosure_id': disclosure_id,
                    'chunk_index': chunk.get('chunk_index', 0),
                    'content': chunk.get('content', ''),
                    'content_type': chunk.get('content_type'),
                    'section_code': chunk.get('section_code'),
                    'heading_text': chunk.get('heading_text'),
                    'char_length': chunk.get('char_length', len(chunk.get('content', ''))),
                    'tokens': chunk.get('tokens', 0),
                    'vectorize': chunk.get('vectorize', True),
                    'is_numeric': chunk.get('is_numeric', False),
                    'disclosure_hash': chunk.get('disclosure_hash'),
                    'source_file': chunk.get('source_file'),
                    'page_number': chunk.get('page_number'),  # Will be None for XBRL chunks
                    'metadata': metadata_json
                })
            
            # Execute batch insert
            with self.conn.cursor() as cur:
                execute_batch(cur, insert_sql, insert_data, page_size=100)
                self.conn.commit()
            
            logger.info(f"Successfully inserted {len(chunks)} chunks for disclosure {disclosure_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting document chunks for disclosure {disclosure_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_existing_chunks_count(self, disclosure_id: int) -> int:
        """
        Get count of existing chunks for a disclosure
        
        Args:
            disclosure_id: ID of the disclosure
            
        Returns:
            Number of existing chunks, or -1 on error
        """
        if not self.conn:
            return -1
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM document_chunks 
                    WHERE disclosure_id = %s
                """, (disclosure_id,))
                
                count = cur.fetchone()[0]
                return count
                
        except Exception as e:
            logger.error(f"Error getting existing chunks count for disclosure {disclosure_id}: {e}")
            return -1
    
    def clear_chunks_for_disclosure(self, disclosure_id: int) -> bool:
        """
        Clear all existing chunks for a disclosure (for reprocessing)
        
        Args:
            disclosure_id: ID of the disclosure
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM document_chunks 
                    WHERE disclosure_id = %s
                """, (disclosure_id,))
                
                deleted_count = cur.rowcount
                self.conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleared {deleted_count} existing chunks for disclosure {disclosure_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error clearing chunks for disclosure {disclosure_id}: {e}")
            self.conn.rollback()
            return False

class UnifiedXBRLProcessor(BaseXBRLProcessor):
    """Enhanced XBRL processor for unified pipeline"""
    
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def check_xbrl_qualitative_available(self, xbrl_path: str) -> bool:
        """Check if XBRL zip contains qualitative.htm file"""
        if not os.path.exists(xbrl_path):
            return False
        
        try:
            with zipfile.ZipFile(xbrl_path, 'r') as z:
                q_files = [n for n in z.namelist() if n.lower().endswith("qualitative.htm")]
                return len(q_files) > 0
        except Exception as e:
            logger.debug(f"Error checking XBRL file {xbrl_path}: {e}")
            return False
    
    def process_xbrl_file_unified(self, xbrl_path: str, disclosure_id: int) -> Tuple[List[XBRLChunk], str]:
        """
        Process XBRL file and return chunks with status
        
        Returns:
            Tuple of (chunks, status) where status is 'success' or 'no_qualitative' or 'error'
        """
        if not self.check_xbrl_qualitative_available(xbrl_path):
            return [], 'no_qualitative'
        
        try:
            chunks = self.process_xbrl_file(xbrl_path, disclosure_id)
            return chunks, 'success'
        except Exception as e:
            logger.error(f"Error processing XBRL {xbrl_path}: {e}")
            return [], 'error'

class UnifiedPDFProcessor:
    """Enhanced PDF processor wrapper for unified pipeline"""
    
    def __init__(self, config: UnifiedProcessingConfig):
        # Create config for the existing AsyncPDFProcessor
        pdf_config = ParallelProcessingConfig(
            max_workers=config.max_workers,
            max_concurrent_files=config.max_concurrent_files,
            batch_size=config.batch_size,
            use_gpu_ocr=config.use_gpu_ocr,
            gpu_batch_size=config.gpu_batch_size,
            memory_limit_gb=config.memory_limit_gb,
            ocr_backend=config.ocr_backend,
            smart_ocr=config.smart_ocr,
            disable_ocr=config.disable_ocr
        )
        
        self.processor = AsyncPDFProcessor(pdf_config)
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def initialize(self):
        """Initialize async components"""
        await self.processor.initialize()
    
    async def process_pdf_file_unified(self, pdf_path: str, disclosure_id: int) -> Tuple[List[DocumentChunk], str]:
        """
        Process PDF file and return chunks with status
        
        Returns:
            Tuple of (chunks, status) where status is 'success' or 'error'
        """
        try:
            chunks = await self.processor.process_pdf_file_async(pdf_path, disclosure_id)
            return chunks, 'success'
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return [], 'error'

class FileSelectionStrategy:
    """Implements intelligent file selection logic"""
    
    def __init__(self, config: UnifiedProcessingConfig):
        self.config = config
    
    def select_file_for_processing(self, row: DatabaseRow) -> Tuple[str, str]:
        """
        Select which file to process based on availability and configuration
        
        Args:
            row: Database row with file paths
            
        Returns:
            Tuple of (file_path, method) where method is 'xbrl' or 'pdf' or 'none'
        """
        xbrl_path = row.xbrl_path
        pdf_path = row.pdf_path
        
        # Check XBRL availability and qualitative.htm requirement
        xbrl_available = False
        if xbrl_path and os.path.exists(xbrl_path):
            if self.config.require_qualitative_htm:
                # Check if qualitative.htm exists in the zip
                try:
                    with zipfile.ZipFile(xbrl_path, 'r') as z:
                        q_files = [n for n in z.namelist() if n.lower().endswith("qualitative.htm")]
                        xbrl_available = len(q_files) > 0
                except Exception:
                    xbrl_available = False
            else:
                xbrl_available = True
        
        # Check PDF availability
        pdf_available = pdf_path and os.path.exists(pdf_path)
        
        # Decision logic
        if self.config.prefer_xbrl and xbrl_available:
            return xbrl_path, 'xbrl'
        elif pdf_available:
            return pdf_path, 'pdf'
        elif not self.config.prefer_xbrl and xbrl_available:
            return xbrl_path, 'xbrl'
        else:
            return '', 'none'

class UnifiedExtractionPipeline:
    """Main unified extraction pipeline orchestrator"""
    
    def __init__(self, config: UnifiedProcessingConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.pg_dsn)
        self.xbrl_processor = UnifiedXBRLProcessor(config.output_dir)
        self.pdf_processor = UnifiedPDFProcessor(config)
        self.file_selector = FileSelectionStrategy(config)
        
        # Statistics
        self.stats = ProcessingStats()
        self.start_time = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Progress monitoring
        self.progress_queue = Queue() if config.enable_progress_monitoring else None
        self.monitor_thread = None
        
        logger.info(f"Initialized unified pipeline with {config.max_workers} workers")
        logger.info(f"Output directory: {config.output_dir}")
        if config.test_mode:
            logger.info(f"Test mode: {config.test_days} days, max {config.max_test_rows} rows")
    
    async def process_disclosure_row(self, row: DatabaseRow) -> ExtractionResult:
        """Process a single disclosure row using intelligent file selection"""
        
        start_time = time.time()
        
        # Mark as processing if tracking is enabled
        if self.config.update_status:
            self.db_manager.mark_processing_start(row.id)
        
        # Select file to process
        file_path, method = self.file_selector.select_file_for_processing(row)
        
        if method == 'none':
            error_msg = 'No suitable files available'
            if self.config.update_status:
                self.db_manager.mark_processing_failed(row.id, error_msg)
            
            return ExtractionResult(
                disclosure_id=row.id,
                extraction_method='failed',
                chunks_count=0,
                processing_time=time.time() - start_time,
                file_path='',
                error_message=error_msg,
                metadata={'company_code': row.company_code, 'company_name': row.company_name}
            )
        
        try:
            logger.info(f"Processing {method.upper()}: {row.company_name} ({row.company_code}) - {os.path.basename(file_path)}")
            
            if method == 'xbrl':
                chunks, status = self.xbrl_processor.process_xbrl_file_unified(file_path, row.id)
                if status == 'no_qualitative':
                    # Fallback to PDF if available
                    if row.pdf_path and os.path.exists(row.pdf_path):
                        logger.info(f"XBRL has no qualitative.htm, falling back to PDF: {row.company_name}")
                        chunks, status = await self.pdf_processor.process_pdf_file_unified(row.pdf_path, row.id)
                        method = 'pdf_fallback'
                        file_path = row.pdf_path
                    else:
                        error_msg = 'XBRL has no qualitative.htm and no PDF available'
                        if self.config.update_status:
                            self.db_manager.mark_processing_failed(row.id, error_msg, 'xbrl', file_path)
                        
                        return ExtractionResult(
                            disclosure_id=row.id,
                            extraction_method='failed',
                            chunks_count=0,
                            processing_time=time.time() - start_time,
                            file_path=file_path,
                            error_message=error_msg,
                            metadata={'company_code': row.company_code, 'company_name': row.company_name}
                        )
            else:  # method == 'pdf'
                chunks, status = await self.pdf_processor.process_pdf_file_unified(file_path, row.id)
            
            if status != 'success':
                error_msg = f'{method.upper()} processing failed: {status}'
                if self.config.update_status:
                    self.db_manager.mark_processing_failed(row.id, error_msg, method, file_path)
                
                return ExtractionResult(
                    disclosure_id=row.id,
                    extraction_method='failed',
                    chunks_count=0,
                    processing_time=time.time() - start_time,
                    file_path=file_path,
                    error_message=error_msg,
                    metadata={'company_code': row.company_code, 'company_name': row.company_name}
                )
            
            # Save chunks to database and/or files
            if chunks:
                success = await self._save_chunks(chunks, row, method)
                if not success:
                    logger.warning(f"Failed to save chunks for {row.company_name}")
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed {method.upper()}: {row.company_name} - {len(chunks)} chunks in {processing_time:.2f}s")
            
            # Mark as completed if tracking is enabled
            if self.config.update_status:
                metadata = {
                    'company_code': row.company_code,
                    'company_name': row.company_name,
                    'disclosure_date': str(row.disclosure_date),
                    'title': row.title,
                    'category': row.category
                }
                self.db_manager.mark_processing_complete(
                    row.id, method, file_path, len(chunks), processing_time, metadata
                )
            
            return ExtractionResult(
                disclosure_id=row.id,
                extraction_method=method,
                chunks_count=len(chunks),
                processing_time=processing_time,
                file_path=file_path,
                metadata={
                    'company_code': row.company_code,
                    'company_name': row.company_name,
                    'disclosure_date': str(row.disclosure_date),
                    'title': row.title,
                    'category': row.category
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {method.upper()} for {row.company_name}: {error_msg}")
            
            # Mark as failed if tracking is enabled
            if self.config.update_status:
                self.db_manager.mark_processing_failed(row.id, error_msg, method, file_path)
            
            return ExtractionResult(
                disclosure_id=row.id,
                extraction_method='failed',
                chunks_count=0,
                processing_time=time.time() - start_time,
                file_path=file_path,
                error_message=error_msg,
                metadata={'company_code': row.company_code, 'company_name': row.company_name}
            )
    
    async def _save_chunks(self, chunks: Union[List[XBRLChunk], List[DocumentChunk]], 
                          row: DatabaseRow, method: str) -> bool:
        """Save chunks to database and/or files based on configuration"""
        success = True
        
        # Convert chunks to standardized format for database
        standardized_chunks = self._standardize_chunks(chunks, row, method)
        
        # Save to database
        if self.config.save_to_database and standardized_chunks:
            try:
                # Clear existing chunks if reprocessing
                if self.config.force_reprocess:
                    self.db_manager.clear_chunks_for_disclosure(row.id)
                
                # Insert chunks
                db_success = self.db_manager.insert_document_chunks(standardized_chunks, row.id)
                if not db_success:
                    logger.error(f"Failed to save chunks to database for disclosure {row.id}")
                    success = False
                else:
                    logger.debug(f"Saved {len(standardized_chunks)} chunks to database for disclosure {row.id}")
                    
            except Exception as e:
                logger.error(f"Error saving chunks to database: {e}")
                success = False
        
        # Save to files (for debugging/inspection)
        if self.config.save_to_files and chunks:
            try:
                await self._save_chunks_to_file(chunks, row, method)
            except Exception as e:
                logger.error(f"Error saving chunks to file: {e}")
                # Don't mark as failure if database save succeeded
        
        return success
    
    def _standardize_chunks(self, chunks: Union[List[XBRLChunk], List[DocumentChunk]], 
                           row: DatabaseRow, method: str) -> List[Dict[str, Any]]:
        """Convert chunks to standardized format for database insertion"""
        standardized = []
        
        for i, chunk in enumerate(chunks):
            # Convert chunk to dict if it's a dataclass or object
            if hasattr(chunk, '__dict__'):
                chunk_dict = chunk.__dict__.copy()
            elif hasattr(chunk, '_asdict'):  # namedtuple
                chunk_dict = chunk._asdict()
            else:
                try:
                    chunk_dict = asdict(chunk)
                except (TypeError, AttributeError):
                    # Fallback for unknown chunk types
                    chunk_dict = {
                        'content': str(chunk),
                        'chunk_index': i,
                        'disclosure_id': row.id
                    }
            
            # Ensure required fields are present
            standardized_chunk = {
                'disclosure_id': chunk_dict.get('disclosure_id', row.id),
                'chunk_index': chunk_dict.get('chunk_index', i),
                'content': chunk_dict.get('content', ''),
                'content_type': chunk_dict.get('content_type'),
                'section_code': chunk_dict.get('section_code'),
                'heading_text': chunk_dict.get('heading_text'),
                'char_length': chunk_dict.get('char_length', len(chunk_dict.get('content', ''))),
                'tokens': chunk_dict.get('tokens', 0),
                'vectorize': chunk_dict.get('vectorize', True),
                'is_numeric': chunk_dict.get('is_numeric', False),
                'disclosure_hash': chunk_dict.get('disclosure_hash'),
                'source_file': chunk_dict.get('source_file'),
                'page_number': chunk_dict.get('page_number'),  # Will be None for XBRL
                'metadata': chunk_dict.get('metadata', {})
            }
            
            # Ensure metadata includes extraction info
            if isinstance(standardized_chunk['metadata'], dict):
                standardized_chunk['metadata'].update({
                    'extraction_method': method,
                    'company_code': row.company_code,
                    'company_name': row.company_name,
                    'disclosure_date': str(row.disclosure_date)
                })
            
            standardized.append(standardized_chunk)
        
        return standardized
    
    async def _save_chunks_to_file(self, chunks: Union[List[XBRLChunk], List[DocumentChunk]], 
                                   row: DatabaseRow, method: str):
        """Save chunks to file for inspection/debugging"""
        try:
            filename = f"{row.company_code}_{row.id}_{method}_chunks.json"
            file_path = os.path.join(self.config.output_dir, filename)
            
            # Convert chunks to serializable format
            if chunks and hasattr(chunks[0], '__dict__'):
                chunks_data = [asdict(chunk) for chunk in chunks]
            else:
                chunks_data = [chunk.__dict__ if hasattr(chunk, '__dict__') else str(chunk) for chunk in chunks]
            
            # Include metadata
            output_data = {
                'disclosure_info': {
                    'id': row.id,
                    'company_code': row.company_code,
                    'company_name': row.company_name,
                    'disclosure_date': str(row.disclosure_date),
                    'extraction_method': method
                },
                'chunks': chunks_data,
                'total_chunks': len(chunks),
                'processed_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving chunks to file: {e}")
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete unified extraction pipeline"""
        
        logger.info("Starting unified extraction pipeline...")
        self.start_time = time.time()
        
        # Connect to database
        if not self.db_manager.connect():
            raise RuntimeError("Failed to connect to database")
        
        try:
            # Reset any stuck processing entries first
            if self.config.update_status:
                reset_count = self.db_manager.reset_processing_status()
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} stuck processing entries")
            
            # Get initial statistics
            if self.config.update_status:
                initial_stats = self.db_manager.get_extraction_statistics()
                logger.info(f"Initial status: {initial_stats['status_breakdown']}")
            
            # Get disclosure rows with tracking
            rows = self.db_manager.get_disclosure_rows_with_tracking(self.config)
            
            if not rows:
                logger.warning("No disclosure rows found to process")
                return {'error': 'No rows found'}
            
            self.stats.total_files = len(rows)
            
            # Initialize processors
            await self.pdf_processor.initialize()
            
            # Start progress monitoring
            if self.config.enable_progress_monitoring:
                self.monitor_thread = threading.Thread(target=self._monitor_progress, daemon=True)
                self.monitor_thread.start()
            
            # Process rows with concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
            
            results = []
            failed_results = []
            
            # Create tasks for all rows
            tasks = [
                self._process_single_row_with_semaphore(semaphore, row)
                for row in rows
            ]
            
            # Execute with progress tracking
            logger.info(f"Starting parallel processing of {len(rows)} disclosure rows...")
            
            for completed_task in asyncio.as_completed(tasks):
                try:
                    result = await completed_task
                    results.append(result)
                    
                    if result.extraction_method == 'failed':
                        failed_results.append({
                            'disclosure_id': result.disclosure_id,
                            'error': result.error_message,
                            'metadata': result.metadata
                        })
                        self.stats.failed_files += 1
                    else:
                        self.stats.processed_files += 1
                        self.stats.total_chunks += result.chunks_count
                    
                    # Update progress
                    if self.progress_queue:
                        self.progress_queue.put({
                            'processed': self.stats.processed_files + self.stats.failed_files,
                            'total': self.stats.total_files,
                            'current': f"{result.metadata.get('company_name', 'Unknown')} ({result.extraction_method})"
                        })
                        
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    self.stats.failed_files += 1
            
            # Calculate final statistics
            self.stats.total_processing_time = time.time() - self.start_time
            if self.stats.total_processing_time > 0:
                self.stats.files_per_second = (self.stats.processed_files + self.stats.failed_files) / self.stats.total_processing_time
                self.stats.chunks_per_second = self.stats.total_chunks / self.stats.total_processing_time
            
            # Generate summary
            summary = {
                'processing_stats': asdict(self.stats),
                'total_rows_processed': self.stats.processed_files,
                'total_rows_failed': self.stats.failed_files,
                'total_chunks_extracted': self.stats.total_chunks,
                'failed_rows': failed_results,
                'results': [asdict(result) for result in results],
                'config': asdict(self.config),
                'extraction_method_breakdown': self._calculate_method_breakdown(results)
            }
            
            # Save summary
            await self._save_summary(summary)
            
            logger.info(f"Pipeline completed in {self.stats.total_processing_time:.2f}s")
            logger.info(f"Processed {self.stats.processed_files} rows, {self.stats.total_chunks} chunks")
            logger.info(f"Performance: {self.stats.files_per_second:.2f} rows/s, {self.stats.chunks_per_second:.2f} chunks/s")
            
            return summary
            
        finally:
            self.db_manager.close()
    
    async def _process_single_row_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                                 row: DatabaseRow) -> ExtractionResult:
        """Process a single row with concurrency control"""
        async with semaphore:
            return await self.process_disclosure_row(row)
    
    def _calculate_method_breakdown(self, results: List[ExtractionResult]) -> Dict[str, int]:
        """Calculate breakdown of extraction methods used"""
        breakdown = {}
        for result in results:
            method = result.extraction_method
            breakdown[method] = breakdown.get(method, 0) + 1
        return breakdown
    
    async def _save_summary(self, summary: Dict[str, Any]):
        """Save processing summary"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = os.path.join(
                self.config.output_dir, 
                f"unified_extraction_summary_{timestamp}.json"
            )
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
    
    def _monitor_progress(self):
        """Monitor processing progress in separate thread"""
        if not self.progress_queue:
            return
        
        last_update = time.time()
        
        while True:
            try:
                if not self.progress_queue.empty():
                    progress = self.progress_queue.get_nowait()
                    current_time = time.time()
                    
                    # Update every 5 seconds or when processing completes
                    if (current_time - last_update > 5.0 or 
                        progress['processed'] >= progress['total']):
                        
                        elapsed = current_time - self.start_time
                        rate = progress['processed'] / elapsed if elapsed > 0 else 0
                        remaining = (progress['total'] - progress['processed']) / rate if rate > 0 else 0
                        
                        logger.info(
                            f"Progress: {progress['processed']}/{progress['total']} rows "
                            f"({progress['processed']/progress['total']*100:.1f}%) - "
                            f"Rate: {rate:.2f} rows/s - "
                            f"ETA: {remaining:.0f}s - "
                            f"Current: {progress['current']}"
                        )
                        
                        last_update = current_time
                    
                    # Exit when all files processed
                    if progress['processed'] >= progress['total']:
                        break
                
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Progress monitoring error: {e}")
                break

async def main_async():
    """Async main function"""
    parser = argparse.ArgumentParser(description='Unified Extraction Pipeline for Financial Intelligence')
    
    # Database settings
    parser.add_argument('--pg-dsn', type=str, 
                       default='',
                       help='PostgreSQL connection string (default: auto-loaded from .env file)')
    parser.add_argument('--env-file', type=str,
                       help='Path to .env file (default: searches parent directories)')
    
    # Processing settings
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes')
    parser.add_argument('--concurrent-files', type=int, default=8, help='Max concurrent files')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    
    # Test mode settings
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--test-days', type=int, default=7, help='Number of recent days to process in test mode')
    parser.add_argument('--max-test-rows', type=int, default=100, help='Maximum rows to process in test mode')
    
    # File selection settings
    parser.add_argument('--prefer-pdf', action='store_true', help='Prefer PDF over XBRL (default: prefer XBRL)')
    parser.add_argument('--allow-no-qualitative', action='store_true', 
                       help='Allow XBRL files without qualitative.htm')
    
    # Tracking and resume settings
    parser.add_argument('--resume', action='store_true', help='Resume processing from last checkpoint (only process unprocessed files)')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed extractions (use with --resume)')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocess all files (ignore completed status)')
    parser.add_argument('--no-update-status', action='store_true', help='Do not update extraction status in database')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for failed files')
    parser.add_argument('--status-report', action='store_true', help='Show extraction status report and exit')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='unified_output', help='Output directory')
    parser.add_argument('--no-save-chunks', action='store_true', help='Do not save chunks at all')
    parser.add_argument('--no-database', action='store_true', help='Do not save chunks to database')
    parser.add_argument('--save-files', action='store_true', help='Save chunks to JSON files (in addition to database)')
    
    # Advanced settings
    parser.add_argument('--gpu-ocr', action='store_true', help='Enable GPU-accelerated OCR')
    parser.add_argument('--disable-ocr', action='store_true', help='Disable OCR completely')
    parser.add_argument('--ocr-backend', type=str, default='tesseract', 
                       choices=['tesseract', 'dolphin', 'auto', 'none'],
                       help='OCR backend to use (default: tesseract)')
    parser.add_argument('--memory-limit', type=int, default=16, help='Memory limit in GB')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Pipeline mode
    parser.add_argument('--full-pipeline', action='store_true', 
                       help='Run full pipeline on all data (overrides test mode)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load database configuration
    try:
        if args.pg_dsn:
            # Use provided PG_DSN
            pg_dsn = args.pg_dsn
            logger.info("Using PG_DSN from command line argument")
        else:
            # Load from .env file or environment variables
            pg_dsn = load_database_config(args.env_file)
        
        if not pg_dsn:
            raise ValueError("No database connection string found")
            
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        print("\nPlease ensure one of the following:")
        print("1. Set PG_DSN in .env file: PG_DSN=postgresql://user:password@localhost/tdnet")
        print("2. Set individual variables in .env file: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DB_PORT")
        print("3. Provide --pg-dsn argument: --pg-dsn postgresql://user:password@localhost/tdnet")
        print("4. Set PG_DSN environment variable")
        sys.exit(1)
    
    # Auto-detect optimal worker count if using default
    if args.workers == 16:  # default value
        cpu_count = mp.cpu_count()
        optimal_workers = min(max(cpu_count // 2, cpu_count * 3 // 4), 32)
        args.workers = optimal_workers
        logger.info(f"Auto-detected optimal worker count: {args.workers}")
    
    # Create configuration
    config = UnifiedProcessingConfig(
        pg_dsn=pg_dsn,
        max_workers=args.workers,
        max_concurrent_files=args.concurrent_files,
        batch_size=args.batch_size,
        test_mode=args.test_mode and not args.full_pipeline,
        test_days=args.test_days,
        max_test_rows=args.max_test_rows,
        prefer_xbrl=not args.prefer_pdf,
        require_qualitative_htm=not args.allow_no_qualitative,
        # Tracking and resume settings
        resume_mode=args.resume,
        retry_failed=args.retry_failed,
        max_retries=args.max_retries,
        force_reprocess=args.force_reprocess,
        skip_completed=not args.force_reprocess,
        update_status=not args.no_update_status,
        # Output settings
        output_dir=args.output_dir,
        save_chunks=not args.no_save_chunks,
        save_to_database=not args.no_database and not args.no_save_chunks,
        save_to_files=args.save_files,
        use_gpu_ocr=args.gpu_ocr,
        disable_ocr=args.disable_ocr,
        ocr_backend=args.ocr_backend,
        memory_limit_gb=args.memory_limit
    )
    
    # Handle status report request
    if args.status_report:
        from tabulate import tabulate
        
        # Create a temporary database manager for status report
        db_manager = DatabaseManager(pg_dsn)
        if not db_manager.connect():
            print("Error: Failed to connect to database")
            sys.exit(1)
        
        try:
            stats = db_manager.get_extraction_statistics()
            
            print("\n" + "="*60)
            print("EXTRACTION STATUS REPORT")
            print("="*60)
            
            # Overall statistics
            overall = stats['overall']
            print(f"Total disclosures: {overall['total_disclosures']:,}")
            print(f"Processable files: {overall['processable']:,}")
            print(f"  - Has XBRL: {overall['has_xbrl']:,}")
            print(f"  - Has PDF: {overall['has_pdf']:,}")
            print(f"Total chunks extracted: {overall['total_chunks']:,}")
            print(f"Average processing time: {overall['avg_duration']:.2f}s")
            
            # Status breakdown
            print(f"\nStatus Breakdown:")
            status_data = [[status, count] for status, count in stats['status_breakdown'].items()]
            print(tabulate(status_data, headers=['Status', 'Count'], tablefmt='grid'))
            
            # Method breakdown
            if stats['method_breakdown']:
                print(f"\nMethod Breakdown:")
                method_data = [
                    [method, data['count'], data['total_chunks'], f"{data['avg_duration']:.2f}s"]
                    for method, data in stats['method_breakdown'].items()
                ]
                print(tabulate(method_data, headers=['Method', 'Count', 'Chunks', 'Avg Duration'], tablefmt='grid'))
            
            # Recent activity
            if stats['recent_activity']:
                print(f"\nRecent Activity (Last 7 Days):")
                activity_data = [[date, count] for date, count in stats['recent_activity'].items()]
                print(tabulate(activity_data, headers=['Date', 'Completed'], tablefmt='grid'))
            
            # Progress calculation
            completed = overall.get('completed', 0)
            processable = overall.get('processable', 0)
            if processable > 0:
                progress = (completed / processable) * 100
                remaining = processable - completed
                print(f"\nProgress: {completed:,}/{processable:,} ({progress:.1f}% complete)")
                print(f"Remaining: {remaining:,} files")
                
                if completed > 0 and overall.get('avg_duration', 0) > 0:
                    estimated_time = remaining * overall['avg_duration']
                    hours = estimated_time / 3600
                    print(f"Estimated time remaining: {hours:.1f} hours")
            
        except ImportError:
            print("Warning: tabulate not installed. Install with: pip install tabulate")
            print("Showing basic statistics:")
            print(json.dumps(stats, indent=2, default=str))
        except Exception as e:
            print(f"Error generating status report: {e}")
        
        finally:
            db_manager.close()
        
        return
    
    # Initialize and run pipeline
    pipeline = UnifiedExtractionPipeline(config)
    
    try:
        summary = await pipeline.run_pipeline()
        
        # Print final summary
        print("\n" + "="*60)
        print("UNIFIED EXTRACTION PIPELINE SUMMARY")
        print("="*60)
        stats = summary.get('processing_stats', {})
        breakdown = summary.get('extraction_method_breakdown', {})
        
        print(f"Rows processed: {stats.get('processed_files', 0)}")
        print(f"Rows failed: {stats.get('failed_files', 0)}")
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        print(f"Processing time: {stats.get('total_processing_time', 0):.2f}s")
        print(f"Performance: {stats.get('files_per_second', 0):.2f} rows/s")
        print(f"Results saved to: {args.output_dir}")
        
        print("\nExtraction method breakdown:")
        for method, count in breakdown.items():
            print(f"  {method}: {count}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main_async())

if __name__ == "__main__":
    main()