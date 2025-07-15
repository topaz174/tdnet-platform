#!/usr/bin/env python3
"""
Enhanced Data Extraction Pipeline for Japanese Financial Intelligence Platform

This script processes XBRL files and PDFs to extract structured financial data
and enhanced text content for the hybrid agent system.

Usage:
    python extraction_pipeline.py --setup-schema  # First time setup
    python extraction_pipeline.py --process-xbrl  # Process XBRL files
    python extraction_pipeline.py --process-pdf   # Enhanced PDF processing
    python extraction_pipeline.py --full-pipeline # Complete processing
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

# Core libraries
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import hashlib

# XBRL processing
try:
    from arelle import ModelManager, FileSource, ModelXbrl, XbrlConst
    from arelle.ModelDocument import ModelDocument
    ARELLE_AVAILABLE = True
except ImportError:
    print("Warning: Arelle not installed. XBRL processing will be limited.")
    ARELLE_AVAILABLE = False

# PDF processing
try:
    import pdfplumber
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: pdfplumber not installed. Enhanced PDF processing will be limited.")
    PDF_PROCESSING_AVAILABLE = False

# Text processing
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not installed. Japanese text processing will be limited.")
    MECAB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FinancialMetric:
    ticker: str
    period_end_date: date
    period_type: str
    fiscal_year: int
    fiscal_period: Optional[int]
    metric_name: str
    metric_value_jpy: int
    unit_scale: str
    confidence_score: float = 1.0

@dataclass
class DocumentChunk:
    disclosure_id: int
    chunk_index: int
    content: str
    content_type: str
    page_number: Optional[int]
    metadata: Dict[str, Any]

class DatabaseManager:
    """Manages database connections and schema operations"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(self.connection_string)
        self.conn.autocommit = False
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def setup_schema(self):
        """Create additional tables for enhanced data storage"""
        
        schema_sql = """
        -- Company Master Table
        CREATE TABLE IF NOT EXISTS company_master (
            securities_code VARCHAR(10) PRIMARY KEY,
            ticker VARCHAR(10),
            company_name_japanese TEXT NOT NULL,
            company_name_english TEXT,
            company_name_kana TEXT,
            sector_japanese VARCHAR(100),
            sector_english VARCHAR(100),
            company_address TEXT,
            corporate_number VARCHAR(20),
            listing_classification VARCHAR(50),
            consolidation_yes_no VARCHAR(5),
            listing_date DATE,
            market_status VARCHAR(50),
            aliases TEXT[],
            keywords TEXT[],
            fiscal_year_end VARCHAR(10),
            edinet_code VARCHAR(20),
            bloomberg_code VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Financial Metrics Table
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            disclosure_id INTEGER REFERENCES disclosures(id),
            ticker VARCHAR(10),
            period_end_date DATE,
            period_type VARCHAR(20),
            fiscal_year INTEGER,
            fiscal_period INTEGER,
            metric_name VARCHAR(100),
            metric_value_jpy BIGINT,
            metric_value_original BIGINT,
            original_currency VARCHAR(3) DEFAULT 'JPY',
            unit_scale VARCHAR(20),
            calculation_method TEXT,
            confidence_score FLOAT DEFAULT 1.0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            
            UNIQUE(disclosure_id, metric_name, period_end_date)
        );
        
        -- Financial Metrics Table
        CREATE TABLE IF NOT EXISTS financial_statements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            disclosure_id INTEGER REFERENCES disclosures(id),
            statement_type VARCHAR(50),
            period_end_date DATE,
            period_type VARCHAR(20),
            fiscal_year INTEGER,
            fiscal_period INTEGER,
            currency VARCHAR(3) DEFAULT 'JPY',
            data JSONB,
            line_items JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Document Chunks Table
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            disclosure_id INTEGER REFERENCES disclosures(id),
            chunk_index INTEGER,
            content TEXT,
            content_type VARCHAR(50),
            page_number INTEGER,
            embedding vector(1024),
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Business Events Table
        CREATE TABLE IF NOT EXISTS business_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            disclosure_id INTEGER REFERENCES disclosures(id),
            event_type VARCHAR(100),
            event_date DATE,
            description TEXT,
            impact_assessment TEXT,
            confidence_score FLOAT,
            extraction_method VARCHAR(50),
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS financial_metrics_ticker_date_idx 
        ON financial_metrics(ticker, period_end_date DESC);
        
        CREATE INDEX IF NOT EXISTS financial_metrics_metric_date_idx 
        ON financial_metrics(metric_name, period_end_date DESC);
        
        CREATE INDEX IF NOT EXISTS document_chunks_disclosure_idx 
        ON document_chunks(disclosure_id);
        
        CREATE INDEX IF NOT EXISTS business_events_type_date_idx 
        ON business_events(event_type, event_date DESC);
        
        -- Add vector index for document chunks (if not exists)
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw 
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """
        
        with self.conn.cursor() as cur:
            cur.execute(schema_sql)
            self.conn.commit()
        
        logger.info("Database schema setup completed")
    
    def populate_company_master(self):
        """Populate company_master table from existing disclosures"""
        
        populate_sql = """
        INSERT INTO company_master (ticker, company_name, sector)
        SELECT DISTINCT 
            company_code as ticker,
            company_name,
            COALESCE(category, 'unknown') as sector
        FROM disclosures
        WHERE company_code IS NOT NULL 
        AND company_name IS NOT NULL
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            updated_at = NOW();
        """
        
        with self.conn.cursor() as cur:
            cur.execute(populate_sql)
            affected_rows = cur.rowcount
            self.conn.commit()
        
        logger.info(f"Populated company_master with {affected_rows} companies")

class XBRLProcessor:
    """Processes XBRL files to extract structured financial data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        # Japanese GAAP concept mappings
        self.concept_mappings = {
            # Revenue concepts
            'jpcrp_cor:NetSales': 'revenue',
            'jpcrp_cor:OperatingRevenue': 'revenue',
            'jppfs_cor:NetSales': 'revenue',
            
            # Profit concepts
            'jpcrp_cor:NetIncome': 'net_income',
            'jpcrp_cor:ProfitLoss': 'net_income',
            'jppfs_cor:NetIncome': 'net_income',
            'jpcrp_cor:OperatingIncome': 'operating_income',
            
            # Balance sheet concepts
            'jpcrp_cor:TotalAssets': 'total_assets',
            'jppfs_cor:TotalAssets': 'total_assets',
            'jpcrp_cor:TotalLiabilities': 'total_liabilities',
            'jpcrp_cor:TotalEquity': 'total_equity',
            'jppfs_cor:TotalEquity': 'total_equity',
            
            # Cash flow concepts
            'jpcrp_cor:CashAndCashEquivalents': 'cash_and_equivalents',
            'jpcrp_cor:OperatingCashFlow': 'operating_cash_flow',
        }
    
    def process_xbrl_file(self, xbrl_path: str, disclosure_id: int) -> List[FinancialMetric]:
        """Process a single XBRL file and extract financial metrics"""
        
        if not ARELLE_AVAILABLE:
            logger.warning(f"Arelle not available, skipping XBRL processing for {xbrl_path}")
            return []
        
        try:
            # Load XBRL document
            model_manager = ModelManager.initialize()
            model_xbrl = ModelXbrl.load(model_manager, xbrl_path)
            
            if not model_xbrl:
                logger.error(f"Failed to load XBRL: {xbrl_path}")
                return []
            
            metrics = []
            
            # Extract company information
            ticker = self._extract_ticker(model_xbrl)
            
            # Extract facts (financial data points)
            for fact in model_xbrl.facts:
                if fact.concept and fact.value is not None:
                    metric = self._process_fact(fact, ticker, disclosure_id)
                    if metric:
                        metrics.append(metric)
            
            # Close the model
            model_xbrl.close()
            model_manager.close()
            
            logger.info(f"Extracted {len(metrics)} metrics from {xbrl_path}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing XBRL {xbrl_path}: {e}")
            return []
    
    def _extract_ticker(self, model_xbrl: ModelXbrl) -> str:
        """Extract company ticker from XBRL"""
        # Try to find ticker in various locations
        try:
            # Look for EDINETCode or company identifier
            for fact in model_xbrl.facts:
                if 'EDINETCode' in str(fact.concept):
                    return fact.value
                if 'CompanyCode' in str(fact.concept):
                    return fact.value
            
            # Fallback: extract from filename or context
            return "UNKNOWN"
            
        except Exception:
            return "UNKNOWN"
    
    def _process_fact(self, fact, ticker: str, disclosure_id: int) -> Optional[FinancialMetric]:
        """Process individual XBRL fact into standardized metric"""
        
        try:
            concept_name = str(fact.concept.qname)
            
            # Check if this is a mapped financial concept
            metric_name = self.concept_mappings.get(concept_name)
            if not metric_name:
                return None
            
            # Extract period information
            period_info = self._extract_period_info(fact)
            if not period_info:
                return None
            
            # Extract and convert value
            value = self._extract_numeric_value(fact)
            if value is None:
                return None
            
            return FinancialMetric(
                ticker=ticker,
                period_end_date=period_info['end_date'],
                period_type=period_info['period_type'],
                fiscal_year=period_info['fiscal_year'],
                fiscal_period=period_info.get('fiscal_period'),
                metric_name=metric_name,
                metric_value_jpy=value,
                unit_scale=self._determine_unit_scale(fact),
                confidence_score=0.95  # High confidence for XBRL data
            )
            
        except Exception as e:
            logger.debug(f"Error processing fact {fact}: {e}")
            return None
    
    def _extract_period_info(self, fact) -> Optional[Dict[str, Any]]:
        """Extract period information from XBRL fact"""
        try:
            context = fact.context
            if not context or not context.period:
                return None
            
            period = context.period
            
            if period.isInstant:
                # Point-in-time data (balance sheet)
                end_date = period.instant.date() if period.instant else None
                period_type = 'instant'
            elif period.isStartEnd:
                # Period data (income statement, cash flow)
                end_date = period.endDate.date() if period.endDate else None
                period_type = 'duration'
            else:
                return None
            
            if not end_date:
                return None
            
            # Determine fiscal year and period
            fiscal_year = end_date.year
            fiscal_period = None
            
            # Simple heuristic for quarterly periods
            month = end_date.month
            if month in [3, 6, 9, 12]:
                fiscal_period = {3: 4, 6: 1, 9: 2, 12: 3}.get(month)
            
            return {
                'end_date': end_date,
                'period_type': period_type,
                'fiscal_year': fiscal_year,
                'fiscal_period': fiscal_period
            }
            
        except Exception:
            return None
    
    def _extract_numeric_value(self, fact) -> Optional[int]:
        """Extract and convert numeric value to JPY"""
        try:
            value = fact.value
            if isinstance(value, str):
                value = float(value.replace(',', ''))
            elif not isinstance(value, (int, float)):
                return None
            
            # Convert to integer JPY (assuming most values are already in JPY)
            return int(value)
            
        except (ValueError, TypeError):
            return None
    
    def _determine_unit_scale(self, fact) -> str:
        """Determine the unit scale of the value"""
        try:
            if fact.unit and fact.unit.measures:
                unit_str = str(fact.unit.measures[0])
                if 'thousands' in unit_str.lower():
                    return 'thousands'
                elif 'millions' in unit_str.lower():
                    return 'millions'
                elif 'billions' in unit_str.lower():
                    return 'billions'
            
            # Default assumption for Japanese financials
            return 'millions'
            
        except Exception:
            return 'actual'
    
    def store_metrics(self, metrics: List[FinancialMetric]):
        """Store extracted financial metrics in database"""
        
        if not metrics:
            return
        
        insert_sql = """
        INSERT INTO financial_metrics (
            disclosure_id, ticker, period_end_date, period_type,
            fiscal_year, fiscal_period, metric_name, metric_value_jpy,
            unit_scale, confidence_score
        ) VALUES (
            %(disclosure_id)s, %(ticker)s, %(period_end_date)s, %(period_type)s,
            %(fiscal_year)s, %(fiscal_period)s, %(metric_name)s, %(metric_value_jpy)s,
            %(unit_scale)s, %(confidence_score)s
        ) ON CONFLICT (disclosure_id, metric_name, period_end_date) 
        DO UPDATE SET
            metric_value_jpy = EXCLUDED.metric_value_jpy,
            confidence_score = EXCLUDED.confidence_score,
            created_at = NOW()
        """
        
        # Convert metrics to dict format
        metric_dicts = []
        for metric in metrics:
            # Get disclosure_id from the first metric (they should all be the same)
            disclosure_id = getattr(metric, 'disclosure_id', None)
            
            metric_dict = {
                'disclosure_id': disclosure_id,
                'ticker': metric.ticker,
                'period_end_date': metric.period_end_date,
                'period_type': metric.period_type,
                'fiscal_year': metric.fiscal_year,
                'fiscal_period': metric.fiscal_period,
                'metric_name': metric.metric_name,
                'metric_value_jpy': metric.metric_value_jpy,
                'unit_scale': metric.unit_scale,
                'confidence_score': metric.confidence_score
            }
            metric_dicts.append(metric_dict)
        
        with self.db_manager.conn.cursor() as cur:
            execute_batch(cur, insert_sql, metric_dicts, page_size=100)
            self.db_manager.conn.commit()
        
        logger.info(f"Stored {len(metrics)} financial metrics")

class PDFProcessor:
    """Enhanced PDF processing for better text chunking and content categorization"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.mecab = MeCab.Tagger() if MECAB_AVAILABLE else None
    
    def process_pdf_file(self, pdf_path: str, disclosure_id: int) -> List[DocumentChunk]:
        """Process PDF file with enhanced chunking and categorization"""
        
        if not PDF_PROCESSING_AVAILABLE:
            logger.warning(f"pdfplumber not available, skipping PDF processing for {pdf_path}")
            return []
        
        try:
            chunks = []
            chunk_index = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Clean and normalize text
                    text = self._clean_text(text)
                    
                    # Categorize content type
                    content_type = self._categorize_content(text, page_num)
                    
                    # Chunk text semantically
                    page_chunks = self._chunk_text_semantically(text, max_chunk_size=500)
                    
                    for chunk_text in page_chunks:
                        if len(chunk_text.strip()) < 50:  # Skip very short chunks
                            continue
                        
                        chunk = DocumentChunk(
                            disclosure_id=disclosure_id,
                            chunk_index=chunk_index,
                            content=chunk_text,
                            content_type=content_type,
                            page_number=page_num,
                            metadata={
                                'page_number': page_num,
                                'chunk_length': len(chunk_text),
                                'language': 'ja'
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            
            # Apply deduplication
            chunks = self._deduplicate_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Japanese text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\-=]+\s*$', '', text, flags=re.MULTILINE)
        
        # Normalize Japanese characters if needed
        # (Add more sophisticated normalization as needed)
        
        return text.strip()
    
    def _categorize_content(self, text: str, page_num: int) -> str:
        """Categorize content type based on text patterns"""
        
        text_lower = text.lower()
        
        # Financial summary indicators
        financial_keywords = ['売上', '利益', '資産', '負債', '財務', '損益', '貸借']
        if any(keyword in text for keyword in financial_keywords):
            return 'financial_summary'
        
        # Management discussion indicators
        management_keywords = ['経営', '戦略', '方針', '見通し', '課題', '対処']
        if any(keyword in text for keyword in management_keywords):
            return 'management_discussion'
        
        # Risk factors
        risk_keywords = ['リスク', '懸念', '不確実', '影響', '対策']
        if any(keyword in text for keyword in risk_keywords):
            return 'risk_factors'
        
        # Business overview
        business_keywords = ['事業', '業務', '製品', 'サービス', '市場']
        if any(keyword in text for keyword in business_keywords):
            return 'business_overview'
        
        # Notes and supplementary
        if page_num > 10:  # Typically later pages
            return 'notes'
        
        return 'general'
    
    def _chunk_text_semantically(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Chunk text semantically using sentence boundaries"""
        
        # Japanese sentence boundary patterns
        sentence_endings = ['。', '！', '？', '．']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.95) -> List[DocumentChunk]:
        """Remove duplicate and near-duplicate chunks based on content"""
        if not chunks:
            return chunks
        
        # Step 1: Remove exact duplicates using hash
        seen_content = set()
        exact_deduplicated = []
        exact_removed = 0
        
        for chunk in chunks:
            content_hash = hash(chunk.content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                exact_deduplicated.append(chunk)
            else:
                exact_removed += 1
                logger.debug(f"Removed exact duplicate chunk {chunk.chunk_index} (page {chunk.page_number}): \"{chunk.content[:50]}...\"")
        
        if exact_removed > 0:
            logger.info(f"Exact deduplication: Removed {exact_removed} duplicate chunks")
        
        # Step 2: Remove near-duplicates using similarity comparison
        # Only do this if we have a reasonable number of chunks (to avoid performance issues)
        if len(exact_deduplicated) <= 100:
            import difflib
            
            final_chunks = []
            near_removed = 0
            
            for i, chunk in enumerate(exact_deduplicated):
                is_duplicate = False
                
                # Compare with already accepted chunks
                for accepted_chunk in final_chunks:
                    # Skip if lengths are very different (quick filter)
                    len_ratio = min(len(chunk.content), len(accepted_chunk.content)) / max(len(chunk.content), len(accepted_chunk.content))
                    if len_ratio < 0.7:
                        continue
                    
                    # Calculate similarity
                    similarity = difflib.SequenceMatcher(None, chunk.content, accepted_chunk.content).ratio()
                    
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        near_removed += 1
                        logger.debug(f"Removed near-duplicate chunk {chunk.chunk_index} (page {chunk.page_number}, similarity: {similarity:.3f}): \"{chunk.content[:50]}...\"")
                        break
                
                if not is_duplicate:
                    final_chunks.append(chunk)
            
            if near_removed > 0:
                logger.info(f"Near-duplicate deduplication: Removed {near_removed} similar chunks (threshold: {similarity_threshold})")
        else:
            final_chunks = exact_deduplicated
            logger.info("Skipping near-duplicate detection for large chunk count (performance)")
        
        # Re-index the remaining chunks to maintain sequential order
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
        
        total_removed = exact_removed + (near_removed if 'near_removed' in locals() else 0)
        if total_removed > 0:
            logger.info(f"Total deduplication: Removed {total_removed} chunks, kept {len(final_chunks)} unique chunks")
        else:
            logger.info("Deduplication: No duplicate chunks found")
        
        return final_chunks
    
    def store_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks in database"""
        
        if not chunks:
            return
        
        insert_sql = """
        INSERT INTO document_chunks (
            disclosure_id, chunk_index, content, content_type,
            page_number, metadata
        ) VALUES (
            %(disclosure_id)s, %(chunk_index)s, %(content)s, %(content_type)s,
            %(page_number)s, %(metadata)s
        )
        """
        
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                'disclosure_id': chunk.disclosure_id,
                'chunk_index': chunk.chunk_index,
                'content': chunk.content,
                'content_type': chunk.content_type,
                'page_number': chunk.page_number,
                'metadata': json.dumps(chunk.metadata)
            }
            chunk_dicts.append(chunk_dict)
        
        with self.db_manager.conn.cursor() as cur:
            execute_batch(cur, insert_sql, chunk_dicts, page_size=100)
            self.db_manager.conn.commit()
        
        logger.info(f"Stored {len(chunks)} document chunks")

class ExtractionPipeline:
    """Main extraction pipeline orchestrator"""
    
    def __init__(self, connection_string: str):
        self.db_manager = DatabaseManager(connection_string)
        self.xbrl_processor = XBRLProcessor(self.db_manager)
        self.pdf_processor = PDFProcessor(self.db_manager)
    
    def setup_database(self):
        """Setup database schema and initial data"""
        logger.info("Setting up database schema...")
        self.db_manager.connect()
        self.db_manager.setup_schema()
        self.db_manager.populate_company_master()
        self.db_manager.close()
    
    def process_all_xbrl_files(self, batch_size: int = 100):
        """Process all XBRL files in the disclosures table"""
        logger.info("Starting XBRL processing...")
        
        self.db_manager.connect()
        
        try:
            # Get all disclosures with XBRL paths
            with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, company_code, company_name, xbrl_path
                    FROM disclosures 
                    WHERE xbrl_path IS NOT NULL 
                    AND xbrl_path != ''
                    AND id NOT IN (
                        SELECT DISTINCT disclosure_id 
                        FROM financial_metrics 
                        WHERE disclosure_id IS NOT NULL
                    )
                    ORDER BY disclosure_date DESC
                    LIMIT %s
                """, (batch_size,))
                
                disclosures = cur.fetchall()
            
            logger.info(f"Processing {len(disclosures)} XBRL files...")
            
            for disclosure in disclosures:
                try:
                    xbrl_path = disclosure['xbrl_path']
                    if not os.path.exists(xbrl_path):
                        logger.warning(f"XBRL file not found: {xbrl_path}")
                        continue
                    
                    logger.info(f"Processing XBRL: {disclosure['company_name']} - {xbrl_path}")
                    
                    metrics = self.xbrl_processor.process_xbrl_file(xbrl_path, disclosure['id'])
                    
                    # Add disclosure_id to metrics
                    for metric in metrics:
                        metric.disclosure_id = disclosure['id']
                    
                    self.xbrl_processor.store_metrics(metrics)
                    
                except Exception as e:
                    logger.error(f"Error processing disclosure {disclosure['id']}: {e}")
                    continue
            
        finally:
            self.db_manager.close()
    
    def process_all_pdf_files(self, batch_size: int = 50):
        """Process all PDF files with enhanced chunking"""
        logger.info("Starting enhanced PDF processing...")
        
        self.db_manager.connect()
        
        try:
            # Get all disclosures with PDF paths
            with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, company_code, company_name, pdf_path
                    FROM disclosures 
                    WHERE pdf_path IS NOT NULL 
                    AND pdf_path != ''
                    AND id NOT IN (
                        SELECT DISTINCT disclosure_id 
                        FROM document_chunks 
                        WHERE disclosure_id IS NOT NULL
                    )
                    ORDER BY disclosure_date DESC
                    LIMIT %s
                """, (batch_size,))
                
                disclosures = cur.fetchall()
            
            logger.info(f"Processing {len(disclosures)} PDF files...")
            
            for disclosure in disclosures:
                try:
                    pdf_path = disclosure['pdf_path']
                    if not os.path.exists(pdf_path):
                        logger.warning(f"PDF file not found: {pdf_path}")
                        continue
                    
                    logger.info(f"Processing PDF: {disclosure['company_name']} - {pdf_path}")
                    
                    chunks = self.pdf_processor.process_pdf_file(pdf_path, disclosure['id'])
                    self.pdf_processor.store_chunks(chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing disclosure {disclosure['id']}: {e}")
                    continue
            
        finally:
            self.db_manager.close()
    
    def run_full_pipeline(self):
        """Run the complete extraction pipeline"""
        logger.info("Starting full extraction pipeline...")
        
        # Setup database schema
        self.setup_database()
        
        # Process XBRL files
        self.process_all_xbrl_files()
        
        # Process PDF files
        self.process_all_pdf_files()
        
        logger.info("Full extraction pipeline completed!")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Data Extraction Pipeline')
    parser.add_argument('--connection-string', 
                       default='postgresql://user:password@localhost/dbname',
                       help='PostgreSQL connection string')
    parser.add_argument('--setup-schema', action='store_true',
                       help='Setup database schema only')
    parser.add_argument('--process-xbrl', action='store_true',
                       help='Process XBRL files only')
    parser.add_argument('--process-pdf', action='store_true',
                       help='Process PDF files only')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Validate connection string
    if 'user:password@localhost/dbname' in args.connection_string:
        print("Error: Please provide a valid PostgreSQL connection string")
        print("Example: postgresql://username:password@localhost/your_database")
        sys.exit(1)
    
    pipeline = ExtractionPipeline(args.connection_string)
    
    try:
        if args.setup_schema:
            pipeline.setup_database()
        elif args.process_xbrl:
            pipeline.process_all_xbrl_files(args.batch_size)
        elif args.process_pdf:
            pipeline.process_all_pdf_files(args.batch_size)
        elif args.full_pipeline:
            pipeline.run_full_pipeline()
        else:
            print("Please specify an action: --setup-schema, --process-xbrl, --process-pdf, or --full-pipeline")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()