#!/usr/bin/env python3
"""
GPU-Optimized Single-Process Embedding Script
============================================

Single-process version to avoid multiprocessing library conflicts while still
using GPU acceleration. Includes robust error handling and memory management.

Key features:
1. GPU acceleration with proper CUDA initialization
2. Single process to avoid shared library conflicts
3. Robust error handling with document-level recovery
4. Memory management and cleanup
5. Uses intfloat/multilingual-e5-large for native 1024-dim embeddings

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname
"""

import os, time, sys, gc
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import psutil

import warnings, logging
warnings.filterwarnings("ignore",
        message=r"CropBox missing from /Page, defaulting to MediaBox")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()              

# Configuration
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Native 1024 dimensions
VECTOR_SIZE = 1024         # ✅ Compatible with current database
CHUNK_CHARS = 3000         # Chunk size for long texts
BATCH_LIMIT = 50           # Documents per batch
MAX_RETRIES = 2            # Maximum retries per document

print(f"🔧 EMBEDDING CONFIGURATION (GPU Single-Process):")
print(f"   Model: {EMBED_MODEL}")
print(f"   Vector Size: {VECTOR_SIZE} dimensions")
print(f"   Chunk Size: {CHUNK_CHARS} characters")
print(f"   Batch Size: {BATCH_LIMIT} documents")

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def initialize_gpu_model():
    """Initialize the embedding model with GPU support."""
    try:
        # Import torch and sentence-transformers
        import torch
        from sentence_transformers import SentenceTransformer
        
        print("🚀 Initializing GPU-accelerated embedding model...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Set memory management
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
            print("   🖥️  GPU not available, using CPU")
        
        # Initialize model
        model = SentenceTransformer(EMBED_MODEL, device=device)
        model_dim = model.get_sentence_embedding_dimension()
        
        print(f"   ✅ Model loaded: {model_dim} dimensions on {device}")
        
        if model_dim != VECTOR_SIZE:
            raise ValueError(f"Model dimension ({model_dim}) != expected ({VECTOR_SIZE})")
        
        return model, device
        
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        raise

def analyze_pdf_risk(pdf_path: str) -> dict:
    """Analyze PDF to detect potential processing risks."""
    risk_factors = {
        'file_size_mb': 0,
        'is_large': False,
        'has_japanese': False,
        'extension_ok': True,
        'risk_score': 0
    }
    
    try:
        # File size analysis
        file_size = os.path.getsize(pdf_path)
        risk_factors['file_size_mb'] = file_size / (1024 * 1024)
        risk_factors['is_large'] = file_size > 20 * 1024 * 1024  # 20MB threshold
        
        # Filename analysis for Japanese characters
        filename = os.path.basename(pdf_path)
        risk_factors['has_japanese'] = any(ord(char) > 127 for char in filename)
        
        # Extension check
        risk_factors['extension_ok'] = pdf_path.lower().endswith('.pdf')
        
        # Calculate risk score
        score = 0
        if risk_factors['is_large']: score += 3
        if risk_factors['has_japanese']: score += 1
        if not risk_factors['extension_ok']: score += 2
        if risk_factors['file_size_mb'] > 50: score += 5
        
        risk_factors['risk_score'] = score
        
    except Exception as e:
        risk_factors['risk_score'] = 10  # Max risk if we can't analyze
    
    return risk_factors

def extract_text_robust(pdf_path: str, risk_info: dict) -> str:
    """Extract text with robust error handling."""
    # Skip extremely large files to prevent crashes
    if risk_info['file_size_mb'] > 100:  # 100MB absolute limit
        return ""
    
    text_parts = []
    
    try:
        # Import PDF libraries with error handling
        import pdfplumber
        from pypdf import PdfReader
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                # Limit pages based on file size
                if risk_info['file_size_mb'] > 50:
                    max_pages = 5  # Very large files: only first 5 pages
                elif risk_info['file_size_mb'] > 20:
                    max_pages = 10  # Large files: first 10 pages
                else:
                    max_pages = min(20, page_count)  # Normal files: up to 20 pages
                
                for page in pdf.pages[:max_pages]:
                    try:
                        page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                        
                        # Extract limited tables for smaller files
                        if risk_info['file_size_mb'] < 30:
                            tables = page.extract_tables()
                            for table in tables[:3]:  # Max 3 tables per page
                                if table and len(table) < 50:  # Skip very large tables
                                    for row in table[:20]:  # Max 20 rows per table
                                        if row and any(cell for cell in row if cell):
                                            table_text = " | ".join(str(cell) if cell else "" for cell in row)
                                            text_parts.append(table_text)
                    except Exception:
                        continue
                        
        except Exception:
            # Fallback to pypdf if pdfplumber fails
            try:
                reader = PdfReader(pdf_path)
                max_pages = min(10, len(reader.pages))  # Conservative with pypdf
                
                for page in reader.pages[:max_pages]:
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                    except Exception:
                        continue
            except Exception:
                pass
        
    except Exception as e:
        print(f"   ⚠️  PDF library import failed: {e}")
    
    return "\n".join(text_parts) if text_parts else ""

def prepare_text_for_embedding(text: str, title: str = "") -> str:
    """Prepare text for embedding."""
    if not text:
        return title
    
    # Clean and normalize text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Combine title and text
    full_text = f"{title} {text}" if title else text
    
    # Truncate to model context limit
    max_length = 3000  # Conservative limit for multilingual-e5-large
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
    
    return full_text

def process_document(doc_id, pdf_path, title, model, device):
    """Process a single document with robust error handling."""
    try:
        # Analyze risk
        risk_info = analyze_pdf_risk(pdf_path)
        
        # Extract text
        text = extract_text_robust(pdf_path, risk_info)
        
        # Use title as fallback if no text extracted
        if not text.strip():
            text = title or ""
            
        if not text.strip():
            return {
                'id': doc_id,
                'success': False,
                'error': 'No text extracted',
                'embedding': None
            }
        
        # Prepare text for embedding
        prepared_text = prepare_text_for_embedding(text, title or "")
        
        # Generate embedding with chunking if needed
        if len(prepared_text) > CHUNK_CHARS:
            chunks = [prepared_text[i:i+CHUNK_CHARS] for i in range(0, len(prepared_text), CHUNK_CHARS)]
            chunk_embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16
            )
            doc_vec = np.mean(chunk_embeddings, axis=0).astype(np.float32)
        else:
            doc_vec = model.encode(
                [prepared_text],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].astype(np.float32)
        
        # Verify embedding dimensions
        if len(doc_vec) != VECTOR_SIZE:
            return {
                'id': doc_id,
                'success': False,
                'error': f'Embedding dimension mismatch: {len(doc_vec)} != {VECTOR_SIZE}',
                'embedding': None
            }
        
        # Clear GPU cache periodically
        if device == 'cuda':
            import torch
            torch.cuda.empty_cache()
        
        return {
            'id': doc_id,
            'success': True,
            'error': None,
            'embedding': doc_vec.tolist(),
            'risk_info': risk_info
        }
        
    except Exception as e:
        # Clean up GPU memory on error
        if device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        return {
            'id': doc_id,
            'success': False,
            'error': str(e),
            'embedding': None
        }

def main():
    """Main processing function."""
    print("🔧 Initializing GPU single-process embedding system...")
    
    # Initialize GPU model
    try:
        model, device = initialize_gpu_model()
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Monitor initial memory usage
    initial_memory = get_memory_usage()
    print(f"   💾 Initial memory usage: {initial_memory:.1f}MB")
    
    # Database connection
    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)

    processed = 0
    embedded = 0
    skipped = 0
    errors = 0
    failed_ids = set()
    risky_files = []

    print(f"\n📊 Starting embedding process...")
    print(f"   Target: Documents without embeddings")
    print(f"   Device: {device.upper()}")

    while True:  # Process until no more rows
        with Session() as session:
            # Build query to exclude failed IDs and NULL paths
            base_query = """SELECT id, pdf_path, title
                           FROM disclosures
                           WHERE embedding IS NULL AND pdf_path IS NOT NULL"""
            
            if failed_ids:
                failed_list = list(failed_ids)
                placeholders = ','.join([':id' + str(i) for i in range(len(failed_list))])
                query = f"{base_query} AND id NOT IN ({placeholders}) ORDER BY id LIMIT :lim"
                params = {"lim": BATCH_LIMIT}
                for i, failed_id in enumerate(failed_list):
                    params[f'id{i}'] = failed_id
            else:
                query = f"{base_query} ORDER BY id LIMIT :lim"
                params = {"lim": BATCH_LIMIT}
            
            rows = session.execute(text(query), params).all()

            if not rows:
                print("✅ No more rows to process!")
                break

            print(f"\n📦 Processing batch of {len(rows)} documents...")
            
            # Process documents
            for doc_id, pdf_path, title in tqdm(rows, desc=f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})"):
                processed += 1
                
                print(f"\n🔍 Processing ID {doc_id}: {os.path.basename(pdf_path) if pdf_path else 'No path'}")
                
                # Quick validation
                if pdf_path is None:
                    print(f"   📁 No file path for document ID {doc_id}")
                    skipped += 1
                    failed_ids.add(doc_id)
                    continue
                
                if not os.path.exists(pdf_path):
                    print(f"   📁 File not found: {pdf_path}")
                    skipped += 1
                    failed_ids.add(doc_id)
                    continue

                # Process document with retries
                success = False
                for attempt in range(MAX_RETRIES + 1):
                    if attempt > 0:
                        print(f"   🔄 Retry {attempt}/{MAX_RETRIES}")
                    
                    result = process_document(doc_id, pdf_path, title, model, device)
                    
                    if result['success']:
                        # Update database
                        session.execute(
                            text("UPDATE disclosures SET embedding = :v WHERE id = :i"),
                            {"v": result['embedding'], "i": doc_id}
                        )
                        
                        embedded += 1
                        success = True
                        
                        # Track risky files
                        if result.get('risk_info', {}).get('risk_score', 0) >= 3:
                            risky_files.append({
                                'id': doc_id,
                                'path': pdf_path,
                                'risk_info': result['risk_info']
                            })
                        
                        # Show first embedding stats
                        if embedded == 1:
                            emb_array = np.array(result['embedding'])
                            print(f"   🔍 First embedding check:")
                            print(f"       Dimension: {len(emb_array)} (expected: {VECTOR_SIZE})")
                            print(f"       Type: {emb_array.dtype}")
                            print(f"       Range: [{emb_array.min():.3f}, {emb_array.max():.3f}]")
                        
                        print(f"   ✅ Successfully embedded document {doc_id}")
                        break
                    else:
                        print(f"   ❌ Processing failed: {result['error']}")
                        if attempt == MAX_RETRIES:
                            errors += 1
                            failed_ids.add(doc_id)
                
                # Force garbage collection after each document
                gc.collect()
                
            session.commit()
            
            # Memory monitoring after batch
            current_memory = get_memory_usage()
            print(f"   ✅ Batch completed. Embedded: {embedded}, Errors: {errors}, Skipped: {skipped}")
            print(f"   💾 Memory usage: {current_memory:.1f}MB")
            
            # Force garbage collection after each batch
            gc.collect()
            if device == 'cuda':
                import torch
                torch.cuda.empty_cache()

    print(f"\n🎉 EMBEDDING PROCESS COMPLETE!")
    print(f"   📊 Total processed: {processed}")
    print(f"   ✅ Successfully embedded: {embedded}")
    print(f"   ❌ Errors: {errors}")
    print(f"   ⏭️  Skipped (no file/path): {skipped}")
    
    if failed_ids:
        print(f"   ❌ Failed IDs: {len(failed_ids)} documents")
        if len(failed_ids) <= 10:
            print(f"      Failed IDs: {sorted(list(failed_ids))}")
        else:
            print(f"      Sample failed IDs: {sorted(list(failed_ids))[:10]}...")
    
    if risky_files:
        print(f"\n⚠️  HIGH-RISK FILE ANALYSIS ({len(risky_files)} files):")
        for rf in risky_files[:10]:  # Show first 10
            print(f"   ID {rf['id']}: {os.path.basename(rf['path'])} "
                  f"({rf['risk_info']['file_size_mb']:.1f}MB, score: {rf['risk_info']['risk_score']})")
        if len(risky_files) > 10:
            print(f"   ... and {len(risky_files) - 10} more risky files")

    # Verify final state
    with Session() as session:
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as with_embeddings
            FROM disclosures
        """)).fetchone()
        
        total, with_emb = result
        print(f"\n📈 FINAL DATABASE STATE:")
        print(f"   Total documents: {total}")
        print(f"   Documents with embeddings: {with_emb}")
        print(f"   Coverage: {with_emb/total*100:.1f}%")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total runtime: {elapsed:.1f} seconds")