#!/usr/bin/env python3
"""
ULTRA-ROBUST: Fixed embedding script that won't crash

Key fixes for segmentation faults:
1. Memory monitoring and cleanup
2. Much lighter ColBERT processing (pseudo-ColBERT features)
3. Individual document processing with full error isolation
4. Process tracking and resumption
5. Safe model loading
"""

import os, time, hashlib, json, re, gc, psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Only essential imports to reduce memory footprint
from sentence_transformers import SentenceTransformer
import torch
import pdfplumber
from pypdf import PdfReader

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-large"
BATCH_SIZE = 5  # Very small batch size for safety
MAX_CONTEXT_LENGTH = 800
CHUNK_SIZE = 3000

# Memory safety thresholds
MAX_MEMORY_PERCENT = 85  # Stop if memory usage exceeds this
CLEANUP_INTERVAL = 3  # Clean memory every N documents

class UltraRobustEmbeddingProcessor:
    def __init__(self):
        print("Initializing ultra-robust embedding processor...")
        self.dense_model = None
        self.engine = create_engine(os.environ["PG_DSN"])
        self.Session = sessionmaker(bind=self.engine)
        self.processed_count = 0
        
        # Load only dense model (much safer than ColBERT)
        self._load_dense_model()
        print("Dense model loaded successfully!")
    
    def _load_dense_model(self):
        """Load dense model with error handling"""
        try:
            self.dense_model = SentenceTransformer(DENSE_MODEL)
            self.dense_model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading dense model: {e}")
            raise
    
    def _check_memory(self) -> bool:
        """Check if memory usage is too high"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MAX_MEMORY_PERCENT:
            print(f"Memory usage too high: {memory_percent:.1f}%")
            return False
        return True
    
    def _cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_text_safe(self, pdf_path: str) -> str:
        """Extract text with comprehensive error handling"""
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
        except Exception:
            pass
        
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
        except Exception:
            pass
        
        return ""
    
    def generate_dense_embedding_safe(self, text: str) -> Optional[np.ndarray]:
        """Generate dense embedding with error handling"""
        try:
            # Limit text length to prevent memory issues
            text = text[:6000] if len(text) > 6000 else text
            
            # Check memory before processing
            if not self._check_memory():
                self._cleanup_memory()
                if not self._check_memory():
                    return None
            
            # Generate embedding
            embedding = self.dense_model.encode([text], normalize_embeddings=True)[0]
            
            # Cleanup after generation
            self._cleanup_memory()
            
            return embedding
            
        except Exception as e:
            print(f"Error generating dense embedding: {e}")
            self._cleanup_memory()
            return None
    
    def generate_simple_colbert_features(self, text: str) -> List[Dict[str, float]]:
        """Generate simple pseudo-ColBERT features instead of full embeddings"""
        try:
            # Much simpler approach: just extract key features
            text = text[:2000]  # Limit text size
            
            # Simple token-level features
            words = text.split()[:50]  # Limit to 50 words max
            features = []
            
            for i, word in enumerate(words):
                # Simple features per "token"
                feature = {
                    "position": i / len(words),  # Relative position
                    "length": min(len(word) / 20.0, 1.0),  # Normalized length
                    "is_number": 1.0 if re.search(r'\d', word) else 0.0,
                    "is_financial": 1.0 if any(term in word.lower() for term in ['円', '億', '兆', '%', '利益', '売上']) else 0.0
                }
                features.append(feature)
            
            return features[:20]  # Limit to 20 features max
            
        except Exception as e:
            print(f"Error generating ColBERT features: {e}")
            return []
    
    def extract_financial_entities_safe(self, text: str) -> Dict[str, Any]:
        """Extract financial entities safely"""
        try:
            entities = {
                "amounts": [],
                "percentages": [],
                "financial_terms": []
            }
            
            # Limit text processing
            text = text[:3000]  # Process only first 3000 chars
            
            # Simple patterns
            amounts = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:円|億円|兆円)', text)
            entities["amounts"] = amounts[:3]  # Limit to 3 amounts
            
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            entities["percentages"] = percentages[:3]  # Limit to 3 percentages
            
            # Simple financial terms
            financial_terms = ["売上", "営業利益", "純利益", "配当", "決算"]
            found_terms = [term for term in financial_terms if term in text]
            entities["financial_terms"] = found_terms[:5]  # Limit to 5 terms
            
            return entities
            
        except Exception as e:
            print(f"Error extracting financial entities: {e}")
            return {"amounts": [], "percentages": [], "financial_terms": []}
    
    def generate_reasoning_context_safe(self, text: str) -> str:
        """Generate reasoning context safely"""
        try:
            # Take first few meaningful lines
            lines = text.split('\n')
            context_lines = []
            
            for line in lines[:5]:  # Even fewer lines
                if line.strip() and len(line.strip()) > 15:
                    context_lines.append(line.strip())
                    if len(' '.join(context_lines)) > 300:  # Much shorter
                        break
            
            return ' '.join(context_lines)[:MAX_CONTEXT_LENGTH]
            
        except Exception as e:
            print(f"Error generating reasoning context: {e}")
            return ""
    
    def process_single_document_safe(self, doc_id: int, pdf_path: str) -> bool:
        """Process single document with comprehensive error handling"""
        try:
            # Memory check before processing
            if not self._check_memory():
                print(f"Skipping doc {doc_id} due to high memory usage")
                return False
            
            # Extract text safely
            text = self.extract_text_safe(pdf_path)
            if not text or not text.strip():
                print(f"No text extracted from: {os.path.basename(pdf_path)}")
                return False
            
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Generate embeddings safely
            dense_embedding = self.generate_dense_embedding_safe(text)
            if dense_embedding is None:
                print(f"Failed to generate dense embedding for doc {doc_id}")
                return False
            
            # Generate simple ColBERT features
            colbert_features = self.generate_simple_colbert_features(text)
            
            # Extract entities and context safely
            financial_entities = self.extract_financial_entities_safe(text)
            reasoning_context = self.generate_reasoning_context_safe(text)
            
            # Processing metadata
            processing_metadata = {
                "processed_at": datetime.now().isoformat(),
                "model_versions": {
                    "dense": DENSE_MODEL,
                    "colbert": "simple_pseudo_colbert"
                },
                "text_length": len(text),
                "embedding_dimensions": {
                    "dense": len(dense_embedding),
                    "colbert_tokens": len(colbert_features)
                }
            }
            
            # Database update with individual transaction
            try:
                with self.Session() as session:
                    session.execute(
                        sql_text("""
                            UPDATE disclosures SET 
                                dense_embedding = :dense_embedding,
                                colbert_doc_embeddings = :colbert_doc_embeddings,
                                reasoning_context = :reasoning_context,
                                financial_entities = :financial_entities,
                                content_hash = :content_hash,
                                processing_metadata = :processing_metadata
                            WHERE id = :id
                        """),
                        {
                            "id": doc_id,
                            "dense_embedding": dense_embedding.tolist(),
                            "colbert_doc_embeddings": json.dumps(colbert_features),
                            "reasoning_context": reasoning_context,
                            "financial_entities": json.dumps(financial_entities),
                            "content_hash": content_hash,
                            "processing_metadata": json.dumps(processing_metadata)
                        }
                    )
                    session.commit()
                    
                    # Force cleanup after each successful document
                    self.processed_count += 1
                    if self.processed_count % CLEANUP_INTERVAL == 0:
                        self._cleanup_memory()
                    
                    return True
                    
            except Exception as e:
                print(f"Database error for doc {doc_id}: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            self._cleanup_memory()
            return False
    
    def needs_processing_safe(self, session, doc_id: int, pdf_path: str) -> bool:
        """Safely check if document needs processing"""
        try:
            result = session.execute(
                sql_text("SELECT content_hash FROM disclosures WHERE id = :id"),
                {"id": doc_id}
            ).first()
            
            if not result or not result[0]:  # No hash means not processed
                return True
            
            # Check if file has changed
            if os.path.exists(pdf_path):
                current_text = self.extract_text_safe(pdf_path)
                if not current_text.strip():
                    return False
                current_hash = hashlib.md5(current_text.encode()).hexdigest()
                return current_hash != result[0]
            
            return False
        except Exception:
            return True  # If we can't check, assume it needs processing
    
    def process_batch_safe(self, start_id: int = None, max_docs: int = None):
        """Process documents with safety checks and progress tracking"""
        processed = 0
        embedded = 0
        skipped = 0
        failed_ids = set()
        
        print(f"Starting safe batch processing...")
        print(f"Memory usage at start: {psutil.virtual_memory().percent:.1f}%")
        
        while True:
            # Get next batch with safety limit
            with self.Session() as session:
                try:
                    base_query = """
                        SELECT id, pdf_path FROM disclosures 
                        WHERE dense_embedding IS NULL 
                           OR content_hash IS NULL
                    """
                    
                    if start_id:
                        base_query += f" AND id >= {start_id}"
                    
                    if failed_ids:
                        failed_list = list(failed_ids)
                        placeholders = ','.join([f':id{i}' for i in range(len(failed_list))])
                        query = f"{base_query} AND id NOT IN ({placeholders}) ORDER BY id LIMIT :lim"
                        params = {"lim": BATCH_SIZE}
                        for i, failed_id in enumerate(failed_list):
                            params[f'id{i}'] = failed_id
                    else:
                        query = f"{base_query} ORDER BY id LIMIT :lim"
                        params = {"lim": BATCH_SIZE}
                    
                    rows = session.execute(sql_text(query), params).all()
                    
                except Exception as e:
                    print(f"Error querying documents: {e}")
                    break
            
            if not rows:
                print("No more documents to process!")
                break
            
            # Check if we've hit max docs limit
            if max_docs and processed >= max_docs:
                print(f"Reached max documents limit: {max_docs}")
                break
            
            print(f"Processing batch of {len(rows)} documents...")
            
            for doc_id, pdf_path in tqdm(rows, desc="Processing documents"):
                processed += 1
                
                # Check if file exists
                if not os.path.exists(pdf_path):
                    print(f"File not found: {pdf_path}")
                    skipped += 1
                    continue
                
                # Memory check before each document
                if not self._check_memory():
                    print(f"Memory usage too high, stopping at document {doc_id}")
                    break
                
                # Process document
                success = self.process_single_document_safe(doc_id, pdf_path)
                
                if success:
                    embedded += 1
                    if embedded % 10 == 0:
                        print(f"Progress: {embedded} embedded, {skipped} skipped")
                else:
                    skipped += 1
                    failed_ids.add(doc_id)
                
                # Memory cleanup every few documents
                if processed % CLEANUP_INTERVAL == 0:
                    self._cleanup_memory()
                    memory_percent = psutil.virtual_memory().percent
                    print(f"Memory usage: {memory_percent:.1f}%")
        
        print(f"\nCompleted batch:")
        print(f"Total processed: {processed}")
        print(f"Successfully embedded: {embedded}")
        print(f"Skipped: {skipped}")
        if failed_ids:
            print(f"Failed documents: {len(failed_ids)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultra-robust document processing")
    parser.add_argument("--sample", type=int, help="Process only N documents for testing")
    parser.add_argument("--start-id", type=int, help="Start processing from this document ID")
    args = parser.parse_args()
    
    processor = UltraRobustEmbeddingProcessor()
    
    # Test with single document first if --sample 1
    if args.sample == 1:
        print("Testing with a single document...")
        with processor.Session() as session:
            row = session.execute(
                sql_text("SELECT id, pdf_path FROM disclosures ORDER BY id LIMIT 1")
            ).first()
            if row:
                print(f"Testing with document {row[0]}: {row[1]}")
                success = processor.process_single_document_safe(row[0], row[1])
                print(f"Test result: {'SUCCESS' if success else 'FAILED'}")
                return
    
    t0 = time.time()
    processor.process_batch_safe(start_id=args.start_id, max_docs=args.sample)
    print(f"Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main() 