#!/usr/bin/env python3
"""
Re-embedding Script: Convert to Native 1024 Dimensions
=====================================

This script re-embeds all documents using the native 1024-dimensional vectors
from intfloat/multilingual-e5-large, eliminating the inefficient 1536-dim padding.

Benefits:
- Reduces storage by ~35MB (512 dims × 4 bytes × 17,259 docs)
- Improves query performance (1024 vs 1536 operations)
- Eliminates zero-padding complexity
- May improve retrieval accuracy

Usage:
    python re_embed_1024_dimensions.py [--batch-size 50] [--start-id 0]
"""

import os
import time
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import pdfplumber
from pypdf import PdfReader
import logging
from dotenv import load_dotenv

import warnings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('re_embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

class DocumentReEmbedder:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.pg_dsn = os.getenv('PG_DSN')
        if not self.pg_dsn:
            raise ValueError("PG_DSN environment variable not found")
        
        logger.info(f"Initialized with batch size: {batch_size}")
        logger.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF with fallback methods"""
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            return None
        
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    try:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    except Exception:
                        continue
                if pages:
                    return "\n".join(pages)
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        
        try:
            # Fallback to pypdf
            reader = PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                except Exception:
                    continue
            if pages:
                return "\n".join(pages)
        except Exception as e:
            logger.debug(f"pypdf failed for {pdf_path}: {e}")
        
        return None
    
    def prepare_text_for_embedding(self, text: str, title: str = "") -> str:
        """Prepare text for embedding (same as original logic)"""
        if not text:
            return title
        
        # Clean and truncate text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Combine title and text
        full_text = f"{title} {text}" if title else text
        
        # Truncate to reasonable length (model context limit)
        max_length = 3000  # Conservative limit
        if len(full_text) > max_length:
            full_text = full_text[:max_length]
        
        return full_text
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate native 1024-dimensional embedding"""
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(1024, dtype=np.float32)
        
        # Generate embedding with normalization
        embedding = self.model.encode(
            [text], 
            normalize_embeddings=True, 
            show_progress_bar=False
        )[0]
        
        # Ensure it's exactly 1024 dimensions (should be, but safety check)
        if len(embedding) != 1024:
            logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected 1024")
            if len(embedding) > 1024:
                embedding = embedding[:1024]
            else:
                padded = np.zeros(1024, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
        
        return embedding.astype(np.float32)
    
    def get_documents_to_process(self, start_id: int = 0) -> List[Tuple[int, str, str]]:
        """Get list of documents that need re-embedding"""
        with psycopg2.connect(self.pg_dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, pdf_path, title
                    FROM disclosures 
                    WHERE id >= %s
                      AND pdf_path IS NOT NULL 
                      AND pdf_path != ''
                    ORDER BY id
                """, (start_id,))
                
                return [(row['id'], row['pdf_path'], row['title'] or '') for row in cur.fetchall()]
    
    def update_embeddings_batch(self, updates: List[Tuple[np.ndarray, int]]) -> None:
        """Update embeddings in database batch"""
        with psycopg2.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                # Prepare batch update
                update_data = []
                for embedding, doc_id in updates:
                    # Convert to list for PostgreSQL vector type
                    emb_list = embedding.tolist()
                    update_data.append((emb_list, doc_id))
                
                # Batch update - use embedding_1024 column for safety
                cur.executemany("""
                    UPDATE disclosures 
                    SET embedding_1024 = %s::vector(1024)
                    WHERE id = %s
                """, update_data)
                
                conn.commit()
                logger.info(f"Updated {len(updates)} embeddings in embedding_1024 column")
    
    def re_embed_all(self, start_id: int = 0) -> None:
        """Re-embed all documents with 1024-dimensional vectors"""
        logger.info("Starting re-embedding process...")
        
        # Get documents to process
        documents = self.get_documents_to_process(start_id)
        total_docs = len(documents)
        logger.info(f"Found {total_docs} documents to re-embed (starting from ID {start_id})")
        
        if total_docs == 0:
            logger.info("No documents to process")
            return
        
        # Process in batches
        batch_updates = []
        processed = 0
        errors = 0
        
        with tqdm(total=total_docs, desc="Re-embedding") as pbar:
            for doc_id, pdf_path, title in documents:
                try:
                    # Extract text
                    text = self.extract_text_from_pdf(pdf_path)
                    if text is None:
                        logger.warning(f"Could not extract text from {pdf_path}")
                        text = title  # Fallback to title
                    
                    # Prepare text for embedding
                    prepared_text = self.prepare_text_for_embedding(text, title)
                    
                    # Generate 1024-dim embedding
                    embedding = self.generate_embedding(prepared_text)
                    
                    # Add to batch
                    batch_updates.append((embedding, doc_id))
                    
                    # Update database when batch is full
                    if len(batch_updates) >= self.batch_size:
                        self.update_embeddings_batch(batch_updates)
                        batch_updates = []
                    
                    processed += 1
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing document {doc_id}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed,
                    'errors': errors,
                    'current_id': doc_id
                })
        
        # Process remaining batch
        if batch_updates:
            self.update_embeddings_batch(batch_updates)
        
        logger.info(f"Re-embedding complete! Processed: {processed}, Errors: {errors}")
    
    def verify_dimensions(self) -> None:
        """Verify that all embeddings now have 1024 dimensions"""
        with psycopg2.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                # Check both embedding columns
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(embedding) as docs_with_1536_embeddings,
                        COUNT(embedding_1024) as docs_with_1024_embeddings
                    FROM disclosures;
                """)
                
                result = cur.fetchone()
                total, old_emb, new_emb = result
                
                logger.info("Embedding status after migration:")
                logger.info(f"  Total documents: {total}")
                logger.info(f"  Documents with 1536-dim embeddings: {old_emb}")
                logger.info(f"  Documents with 1024-dim embeddings: {new_emb}")
                
                if new_emb > 0:
                    # Sample a few to verify they're actually 1024 dimensions
                    cur.execute("""
                        SELECT embedding_1024::text 
                        FROM disclosures 
                        WHERE embedding_1024 IS NOT NULL 
                        LIMIT 3;
                    """)
                    
                    samples = cur.fetchall()
                    logger.info("Sample 1024-dim embedding verification:")
                    for i, (emb_text,) in enumerate(samples):
                        if emb_text.startswith('[') and emb_text.endswith(']'):
                            elements = emb_text[1:-1].split(',')
                            dim = len(elements)
                            logger.info(f"  Sample {i+1}: {dim} dimensions ✓" if dim == 1024 else f"  Sample {i+1}: {dim} dimensions ✗")
                        else:
                            logger.info(f"  Sample {i+1}: Unexpected format")


def main():
    parser = argparse.ArgumentParser(description='Re-embed documents with 1024 dimensions')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Batch size for database updates (default: 50)')
    parser.add_argument('--start-id', type=int, default=0,
                       help='Start from this document ID (for resuming, default: 0)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify current dimensions, do not re-embed')
    
    args = parser.parse_args()
    
    embedder = DocumentReEmbedder(batch_size=args.batch_size)
    
    if args.verify_only:
        embedder.verify_dimensions()
    else:
        start_time = time.time()
        embedder.re_embed_all(start_id=args.start_id)
        end_time = time.time()
        
        logger.info(f"Total time: {end_time - start_time:.2f} seconds")
        
        # Verify results
        embedder.verify_dimensions()


if __name__ == "__main__":
    main()