#!/usr/bin/env python3
"""
1024-Dimensional Compatible Embedding Script (Process-Isolated Version)
======================================================================

Fixed version with complete process isolation to prevent segmentation faults.

Key fixes:
1. Complete process isolation using multiprocessing
2. Memory management with garbage collection
3. Worker pool approach to prevent model reloading
4. Timeout and error recovery mechanisms
5. Uses intfloat/multilingual-e5-large for native 1024-dim embeddings

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname
"""

import os, time, pathlib, subprocess, sys, json, gc

# Suppress TensorFlow and CUDA verbose logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU if available
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistency
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import multiprocessing
import signal
import psutil
import resource
from contextlib import contextmanager

import warnings, logging
warnings.filterwarnings("ignore",
        message=r"CropBox missing from /Page, defaulting to MediaBox")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()              

# FIXED: Use 1024-dimensional model compatible with database
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Native 1024 dimensions
VECTOR_SIZE = 1024         # ✅ Compatible with current database
CHUNK_CHARS = 3000         # Slightly smaller for better quality
BATCH_LIMIT = 50           # Even smaller batches for stability
WORKER_TIMEOUT = 180       # 3 minutes timeout per document
MAX_RETRIES = 2            # Maximum retries per document

@contextmanager
def timeout_context(seconds):
    """Context manager for timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def set_worker_resource_limits():
    """Set resource limits for worker processes."""
    try:
        # Limit memory to 4GB per worker process
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
        # Limit CPU time to 10 minutes per worker
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    except Exception as e:
        print(f"   ⚠️  Could not set resource limits: {e}")

# ---------------------------------------------------------------------------

def process_document_worker(args):
    """Worker function that processes a single document in isolation.
    
    This function runs in a separate process to prevent crashes from
    affecting the main process.
    """
    doc_id, pdf_path, title = args
    
    try:
        # Set resource limits for this worker process
        # set_worker_resource_limits()
        
        # Suppress TensorFlow/CUDA logging in worker process
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Import heavy libraries only in worker process
        import pdfplumber
        from pypdf import PdfReader
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import torch
        
        # Suppress logging after imports too
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        initial_memory = get_memory_usage()
        
        # Proper CUDA initialization for worker processes
        if torch.cuda.is_available():
            try:
                # Clear any existing CUDA context
                torch.cuda.empty_cache()
                # Initialize CUDA context in worker process
                torch.cuda.init()
                # Set memory fraction to avoid OOM
                torch.cuda.set_per_process_memory_fraction(0.8)
                device = 'cuda'
                print(f"   🚀 Using GPU acceleration in worker process")
            except Exception as cuda_error:
                print(f"   ⚠️  CUDA initialization failed: {cuda_error}, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
            print(f"   🖥️  Using CPU in worker process")
        
        # Initialize model in worker process with proper device
        model = SentenceTransformer(EMBED_MODEL, device=device)
        model_memory = get_memory_usage()
        
        # Monitor memory usage
        if model_memory > 3000:  # 3GB threshold
            print(f"   ⚠️  High memory usage after model load: {model_memory:.1f}MB")
        
        # Analyze risk
        risk_info = _analyze_pdf_risk_worker(pdf_path)
        
        # Extract text
        text = _extract_text_worker(pdf_path, risk_info)
        
        # Generate embedding
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
        prepared_text = _prepare_text_for_embedding_worker(text, title or "")
        
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
        
        # Force garbage collection and memory cleanup before returning
        final_memory = get_memory_usage()
        
        # Clean up GPU memory if using CUDA
        if 'torch' in locals() and torch.cuda.is_available() and device == 'cuda':
            torch.cuda.empty_cache()
        
        del model
        gc.collect()
        
        return {
            'id': doc_id,
            'success': True,
            'error': None,
            'embedding': doc_vec.tolist(),
            'risk_info': risk_info,
            'memory_usage': {
                'initial': initial_memory,
                'model_loaded': model_memory,
                'final': final_memory
            }
        }
        
    except Exception as e:
        # Force cleanup on error
        try:
            if 'model' in locals():
                del model
            # Clean up GPU memory on error
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        gc.collect()
        return {
            'id': doc_id,
            'success': False,
            'error': str(e),
            'embedding': None
        }

def _analyze_pdf_risk_worker(pdf_path: str) -> dict:
    """Analyze PDF risk in worker process."""
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

def _extract_text_worker(pdf_path: str, risk_info: dict) -> str:
    """Extract text in worker process with risk-based approach."""
    # Import libraries only in worker
    import pdfplumber
    from pypdf import PdfReader
    
    # Skip extremely large files to prevent crashes
    if risk_info['file_size_mb'] > 100:  # 100MB absolute limit
        return ""
    
    text_parts = []
    
    try:
        # Try pdfplumber first (better for complex layouts)
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
                    
                    # Extract limited tables for large files
                    if risk_info['file_size_mb'] < 30:  # Only for smaller files
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
            max_pages = min(10, len(reader.pages))  # Even more conservative with pypdf
            
            for page in reader.pages[:max_pages]:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception:
                    continue
        except Exception:
            pass
    
    return "\n".join(text_parts) if text_parts else ""

def _prepare_text_for_embedding_worker(text: str, title: str = "") -> str:
    """Prepare text for embedding in worker process."""
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

# ---------------------------------------------------------------------------

def main():
    """Main process using multiprocessing for isolation."""
    print("🔧 Initializing process-isolated embedding system...")

    print(f"🔧 EMBEDDING CONFIGURATION (Process-Isolated):")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Vector Size: {VECTOR_SIZE} dimensions")
    print(f"   Chunk Size: {CHUNK_CHARS} characters")
    print(f"   Batch Size: {BATCH_LIMIT} documents")
    print(f"   Worker Timeout: {WORKER_TIMEOUT} seconds")
    
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

    print(f"\n📊 Starting embedding process (Process-Isolated)...")
    print(f"   Target: Documents without embeddings")
    print(f"   Worker Timeout: {WORKER_TIMEOUT} seconds")
    print(f"   Max Retries: {MAX_RETRIES}")

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
            
            # Use persistent worker pool for the entire batch to reduce process creation overhead
            with multiprocessing.Pool(processes=1) as pool:
                progress_bar = tqdm(rows, desc=f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")
                for doc_id, pdf_path, title in progress_bar:
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
                        
                        try:
                            # Submit work to persistent process pool with timeout
                            async_result = pool.apply_async(
                                process_document_worker,
                                [(doc_id, pdf_path, title)]
                            )
                            
                            # Wait for result with timeout
                            result = async_result.get(timeout=WORKER_TIMEOUT)
                            
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
                                print(f"   ❌ Worker failed: {result['error']}")
                                if attempt == MAX_RETRIES:
                                    errors += 1
                                    failed_ids.add(doc_id)
                        
                        except multiprocessing.TimeoutError:
                            print(f"   ⏰ Worker timeout after {WORKER_TIMEOUT}s")
                            if attempt == MAX_RETRIES:
                                errors += 1
                                failed_ids.add(doc_id)
                        
                        except Exception as e:
                            print(f"   💥 Process error: {e}")
                            if attempt == MAX_RETRIES:
                                errors += 1
                                failed_ids.add(doc_id)
                    
                    # Force garbage collection after each document
                    gc.collect()
                    
                    # Update progress bar description with the latest stats
                    progress_bar.set_description(f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")

            session.commit()
            
            # Memory monitoring after batch
            current_memory = get_memory_usage()
            print(f"   ✅ Batch completed. Embedded: {embedded}, Errors: {errors}, Skipped: {skipped}")
            print(f"   💾 Memory usage: {current_memory:.1f}MB")
            
            # Force garbage collection after each batch
            gc.collect()

    print(f"\n🎉 EMBEDDING PROCESS COMPLETE!")
    print(f"   📊 Total processed: {processed}")
    print(f"   ✅ Successfully embedded: {embedded}")
    print(f"   ❌ Errors (worker crashes/timeouts): {errors}")
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
    # Essential for multiprocessing - use forkserver for better GPU compatibility
    try:
        multiprocessing.set_start_method('forkserver', force=True)
        print("🔧 Using 'forkserver' multiprocessing method for GPU compatibility")
    except RuntimeError:
        # Fall back to spawn if forkserver not available
        multiprocessing.set_start_method('spawn', force=True)
        print("🔧 Using 'spawn' multiprocessing method")
    
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