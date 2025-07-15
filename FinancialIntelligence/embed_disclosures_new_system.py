#!/usr/bin/env python3
"""
Enhanced embedding script for the new retrieval system
Populates all new columns: dense_embedding, colbert_doc_embeddings, 
reasoning_context, financial_entities, content_hash, processing_metadata

Key improvements:
1. Uses local models (no API costs)
2. Processes multiple embedding types
3. Extracts financial entities
4. Incremental processing with change detection
5. Batch processing for efficiency
"""

import os, time, hashlib, json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# New models for local processing
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import pdfplumber
from pypdf import PdfReader

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-large"
COLBERT_MODEL = "lightonai/Reason-ModernColBERT"
BATCH_SIZE = 50
MAX_CONTEXT_LENGTH = 1000
CHUNK_SIZE = 4000

class EnhancedEmbeddingProcessor:
    def __init__(self):
        print("Loading models...")
        self.dense_model = SentenceTransformer(DENSE_MODEL)
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL)
        self.colbert_model = AutoModel.from_pretrained(COLBERT_MODEL)
        self.colbert_model.eval()
        print("Models loaded successfully!")
        
        # Database connection
        self.engine = create_engine(os.environ["PG_DSN"])
        self.Session = sessionmaker(bind=self.engine)
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with same fallback logic as original script"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            pass
        try:
            reader = PdfReader(pdf_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""
    
    def generate_dense_embedding(self, text: str) -> np.ndarray:
        """Generate dense embedding using multilingual-e5-large"""
        return self.dense_model.encode([text], normalize_embeddings=True)[0]
    
    def generate_colbert_embeddings(self, text: str) -> List[List[float]]:
        """Generate ColBERT token-level embeddings"""
        inputs = self.colbert_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.colbert_model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
            
        # Normalize embeddings
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings.cpu().numpy().tolist()
    
    def extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """Extract financial entities and metrics from text"""
        entities = {
            "amounts": [],
            "percentages": [],
            "dates": [],
            "financial_terms": [],
            "companies_mentioned": []
        }
        
        # Japanese currency patterns
        yen_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:円|億円|兆円)',
            r'(\d+(?:\.\d+)?)\s*(?:億|兆)',
        ]
        
        for pattern in yen_patterns:
            amounts = re.findall(pattern, text)
            entities["amounts"].extend(amounts)
        
        # Percentage patterns
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(pct_pattern, text)
        entities["percentages"] = percentages[:10]  # Limit to first 10
        
        # Financial terms (Japanese)
        financial_terms = [
            "売上", "営業利益", "純利益", "配当", "株主総会", "決算", "業績予想",
            "設備投資", "自己株式", "株式分割", "増配", "減配", "業績修正",
            "四半期", "通期", "前年同期", "売上高", "営業利益率"
        ]
        
        found_terms = [term for term in financial_terms if term in text]
        entities["financial_terms"] = found_terms[:15]  # Limit to avoid huge JSON
        
        # Company code patterns (4-5 digits)
        company_codes = re.findall(r'\b\d{4,5}\b', text)
        entities["companies_mentioned"] = list(set(company_codes))[:20]
        
        return entities
    
    def generate_reasoning_context(self, text: str) -> str:
        """Extract key context for reasoning-based queries"""
        # Take first paragraph + any lines with numbers/percentages
        lines = text.split('\n')
        context_lines = []
        
        # First few meaningful lines
        for line in lines[:10]:
            if line.strip() and len(line.strip()) > 20:
                context_lines.append(line.strip())
                if len(' '.join(context_lines)) > 500:
                    break
        
        # Add lines with financial data
        for line in lines:
            if any(char in line for char in ['%', '円', '億', '兆']) and len(line.strip()) > 10:
                context_lines.append(line.strip())
                if len(' '.join(context_lines)) > MAX_CONTEXT_LENGTH:
                    break
        
        return ' '.join(context_lines)[:MAX_CONTEXT_LENGTH]
    
    def process_document(self, doc_id: int, pdf_path: str) -> Dict[str, Any]:
        """Process a single document and return all new column data"""
        # Extract text
        text = self.extract_text(pdf_path)
        if not text.strip():
            return None
        
        # Generate content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Generate embeddings
        dense_embedding = self.generate_dense_embedding(text)
        colbert_embeddings = self.generate_colbert_embeddings(text[:8000])  # Limit for efficiency
        
        # Extract entities and context
        financial_entities = self.extract_financial_entities(text)
        reasoning_context = self.generate_reasoning_context(text)
        
        # Processing metadata
        processing_metadata = {
            "processed_at": datetime.now().isoformat(),
            "model_versions": {
                "dense": DENSE_MODEL,
                "colbert": COLBERT_MODEL
            },
            "text_length": len(text),
            "embedding_dimensions": {
                "dense": len(dense_embedding),
                "colbert_tokens": len(colbert_embeddings)
            }
        }
        
        return {
            "dense_embedding": dense_embedding.tolist(),
            "colbert_doc_embeddings": colbert_embeddings,
            "reasoning_context": reasoning_context,
            "financial_entities": json.dumps(financial_entities),
            "content_hash": content_hash,
            "processing_metadata": json.dumps(processing_metadata)
        }
    
    def needs_processing(self, session, doc_id: int, pdf_path: str) -> bool:
        """Check if document needs processing (new or changed)"""
        # Check if we have existing data
        result = session.execute(
            text("SELECT content_hash, processing_metadata FROM disclosures WHERE id = :id"),
            {"id": doc_id}
        ).first()
        
        if not result or not result[0]:  # No hash means not processed
            return True
        
        # Check if file has changed
        if os.path.exists(pdf_path):
            current_text = self.extract_text(pdf_path)
            current_hash = hashlib.md5(current_text.encode()).hexdigest()
            return current_hash != result[0]
        
        return False
    
    def process_batch(self, force_reprocess: bool = False):
        """Process documents in batches"""
        processed = 0
        embedded = 0
        skipped = 0
        failed_ids = set()
        
        while True:
            with self.Session() as session:
                # Query for documents needing processing
                if force_reprocess:
                    # Reprocess everything
                    base_query = "SELECT id, pdf_path FROM disclosures"
                else:
                    # Only process new or changed documents
                    base_query = """
                        SELECT id, pdf_path FROM disclosures 
                        WHERE dense_embedding IS NULL 
                           OR content_hash IS NULL
                    """
                
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
                
                rows = session.execute(text(query), params).all()
                
                if not rows:
                    print("No more documents to process!")
                    break
                
                print(f"Processing batch of {len(rows)} documents...")
                
                for doc_id, pdf_path in tqdm(rows, desc=f"Processing (embedded: {embedded}, skipped: {skipped})"):
                    processed += 1
                    
                    # Check if file exists
                    if not os.path.exists(pdf_path):
                        print(f"File not found: {pdf_path}")
                        skipped += 1
                        failed_ids.add(doc_id)
                        continue
                    
                    # Check if processing needed (unless forcing)
                    if not force_reprocess and not self.needs_processing(session, doc_id, pdf_path):
                        print(f"Document {doc_id} already up to date, skipping")
                        skipped += 1
                        continue
                    
                    try:
                        # Process document
                        result = self.process_document(doc_id, pdf_path)
                        if not result:
                            print(f"No text extracted from: {os.path.basename(pdf_path)}")
                            skipped += 1
                            failed_ids.add(doc_id)
                            continue
                        
                        # Update database
                        session.execute(
                            text("""
                                UPDATE disclosures SET 
                                    dense_embedding = :dense_embedding,
                                    colbert_doc_embeddings = :colbert_doc_embeddings,
                                    reasoning_context = :reasoning_context,
                                    financial_entities = :financial_entities,
                                    content_hash = :content_hash,
                                    processing_metadata = :processing_metadata
                                WHERE id = :id
                            """),
                            {"id": doc_id, **result}
                        )
                        embedded += 1
                        
                        if embedded == 1:
                            print(f"First document processed - Dense embedding dims: {len(result['dense_embedding'])}")
                        
                    except Exception as e:
                        print(f"Error processing document {doc_id}: {e}")
                        skipped += 1
                        failed_ids.add(doc_id)
                
                session.commit()
                print(f"Batch completed. Total embedded: {embedded}, Total skipped: {skipped}")
        
        print(f"\nFinal summary:")
        print(f"Total processed: {processed}")
        print(f"Successfully embedded: {embedded}")
        print(f"Skipped: {skipped}")
        if failed_ids:
            print(f"Failed IDs: {len(failed_ids)} documents")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process documents for new retrieval system")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--sample", type=int, help="Process only N documents for testing")
    args = parser.parse_args()
    
    processor = EnhancedEmbeddingProcessor()
    
    t0 = time.time()
    processor.process_batch(force_reprocess=args.force)
    print(f"Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main() 