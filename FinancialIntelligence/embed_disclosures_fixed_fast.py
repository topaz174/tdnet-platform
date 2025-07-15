#!/usr/bin/env python3
"""
1024-Dimensional Compatible Embedding Script (Fast Worker Pool Version)
======================================================================

Optimized for speed using a persistent worker pool to avoid model reloading.

Key optimizations:
1. Persistent worker pool (`multiprocessing.Pool`)
2. Model is loaded only ONCE per worker process
3. `imap_unordered` for efficient, parallel task distribution
4. Automatic worker recycling (`maxtasksperchild`) for stability
5. Uses intfloat/multilingual-e5-large for native 1024-dim embeddings

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname
"""

import os, time, pathlib, subprocess, sys, json, gc
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import multiprocessing
import psutil
import resource

import warnings, logging
warnings.filterwarnings("ignore",
        message=r"CropBox missing from /Page, defaulting to MediaBox")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()              

# CONFIGURATION
EMBED_MODEL = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024
CHUNK_CHARS = 3000
BATCH_LIMIT = 200          # Larger batches are more efficient with a worker pool
MAX_TASKS_PER_CHILD = 100  # Recycle workers to manage memory
# NUM_WORKERS = multiprocessing.cpu_count() # THIS IS TOO AGGRESSIVE for memory-heavy tasks.
# Each worker loads a large model. The number of workers should be limited by available
# GPU VRAM and system RAM, not the number of CPU cores.
# Start with a small number and increase carefully. For a single GPU, 2 is a safe starting point.
NUM_WORKERS = 2 # Use a conservative number of workers to avoid memory exhaustion.

# ---------------------------------------------------------------------------

class Worker:
    """A worker process that initializes the model once and processes documents."""
    
    def __init__(self):
        """Initialize the worker, loading the model into memory."""
        # Import heavy libraries only in worker process
        from sentence_transformers import SentenceTransformer
        import torch

        print(f"   🔧 Worker (PID: {os.getpid()}) initializing...")
        
        # Set resource limits for this worker process
        self._set_resource_limits()
        
        # Determine device (GPU or CPU)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.init()
                self.device = 'cuda'
                print(f"   🚀 Worker (PID: {os.getpid()}) using GPU.")
            except Exception as cuda_error:
                print(f"   ⚠️  Worker (PID: {os.getpid()}) CUDA init failed: {cuda_error}, using CPU.")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            print(f"   🖥️  Worker (PID: {os.getpid()}) using CPU.")
            
        # Load the model
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        print(f"   ✅ Worker (PID: {os.getpid()}) model loaded.")

    def _set_resource_limits(self):
        """Set resource limits for the worker process."""
        try:
            # Limit memory to 8GB per worker process (more generous for pool)
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
        except Exception as e:
            print(f"   ⚠️  Worker (PID: {os.getpid()}) could not set resource limits: {e}")

    def process(self, args):
        """Processes a single document."""
        doc_id, pdf_path, title = args
        
        # Quick validation
        if pdf_path is None or not os.path.exists(pdf_path):
            return {'id': doc_id, 'success': False, 'error': 'File not found or path is null', 'embedding': None}

        try:
            # Analyze risk
            risk_info = self._analyze_pdf_risk(pdf_path)
            
            # Extract text
            text = self._extract_text(pdf_path, risk_info)
            
            # Generate embedding
            if not text.strip():
                text = title or ""
                
            if not text.strip():
                return {'id': doc_id, 'success': False, 'error': 'No text extracted', 'embedding': None}
            
            # Prepare text for embedding
            prepared_text = self._prepare_text_for_embedding(text, title or "")
            
            # Generate embedding with chunking if needed
            if len(prepared_text) > CHUNK_CHARS:
                chunks = [prepared_text[i:i+CHUNK_CHARS] for i in range(0, len(prepared_text), CHUNK_CHARS)]
                chunk_embeddings = self.model.encode(
                    chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=16
                )
                doc_vec = np.mean(chunk_embeddings, axis=0).astype(np.float32)
            else:
                doc_vec = self.model.encode(
                    [prepared_text], normalize_embeddings=True, show_progress_bar=False
                )[0].astype(np.float32)
            
            if len(doc_vec) != VECTOR_SIZE:
                return {'id': doc_id, 'success': False, 'error': f'Embedding dim mismatch', 'embedding': None}

            return {
                'id': doc_id,
                'success': True,
                'error': None,
                'embedding': doc_vec.tolist(),
                'risk_info': risk_info,
            }
            
        except Exception as e:
            return {'id': doc_id, 'success': False, 'error': str(e), 'embedding': None}

    def _analyze_pdf_risk(self, pdf_path: str) -> dict:
        risk_factors = {'file_size_mb': 0, 'is_large': False, 'risk_score': 0}
        try:
            file_size = os.path.getsize(pdf_path)
            risk_factors['file_size_mb'] = file_size / (1024 * 1024)
            risk_factors['is_large'] = risk_factors['file_size_mb'] > 20
            score = 0
            if risk_factors['is_large']: score += 3
            if risk_factors['file_size_mb'] > 50: score += 5
            risk_factors['risk_score'] = score
        except:
            risk_factors['risk_score'] = 10
        return risk_factors

    def _extract_text(self, pdf_path: str, risk_info: dict) -> str:
        import pdfplumber
        if risk_info.get('file_size_mb', 0) > 100: return ""
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                max_pages = 10 if risk_info.get('is_large') else 20
                for page in pdf.pages[:max_pages]:
                    try:
                        page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                        if page_text and page_text.strip(): text_parts.append(page_text)
                    except: continue
        except: pass
        return "\n".join(text_parts)

    def _prepare_text_for_embedding(self, text: str, title: str = "") -> str:
        if not text: return title
        text = ' '.join(text.replace('\n', ' ').split())
        full_text = f"{title} {text}" if title else text
        return full_text[:CHUNK_CHARS * 5] # Generous truncation limit

# This function needs to be at the top level to be pickleable by multiprocessing
def init_worker():
    """Initializer for each worker in the pool."""
    global worker
    worker = Worker()

def process_document_proxy(args):
    """Proxy function to call the process method on the global worker."""
    return worker.process(args)

# ---------------------------------------------------------------------------

def main():
    """Main process using a persistent worker pool."""
    print("🚀 Initializing fast embedding system with worker pool...")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Vector Size: {VECTOR_SIZE} dimensions")
    print(f"   Chunk Size: {CHUNK_CHARS} characters")
    print(f"   Batch Size: {BATCH_LIMIT} documents")
    print(f"   Number of Workers: {NUM_WORKERS}")
    print(f"   Worker Recycle Rate: {MAX_TASKS_PER_CHILD} tasks/worker")

    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)
    
    processed, embedded, errors, skipped = 0, 0, 0, 0
    failed_ids = set()

    print(f"\n📊 Starting embedding process...")

    # Create the worker pool
    # `maxtasksperchild` is crucial for stability, recycling workers
    pool = multiprocessing.Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        maxtasksperchild=MAX_TASKS_PER_CHILD
    )

    while True:
        with Session() as session:
            # Build query to fetch a batch of documents
            base_query = "SELECT id, pdf_path, title FROM disclosures WHERE embedding IS NULL AND pdf_path IS NOT NULL"
            if failed_ids:
                placeholders = ','.join([':id' + str(i) for i in range(len(failed_ids))])
                query = f"{base_query} AND id NOT IN ({placeholders}) ORDER BY id LIMIT :lim"
                params = {"lim": BATCH_LIMIT, **{f'id{i}': fid for i, fid in enumerate(failed_ids)}}
            else:
                query = f"{base_query} ORDER BY id LIMIT :lim"
                params = {"lim": BATCH_LIMIT}
            
            rows = session.execute(text(query), params).all()

            if not rows:
                print("✅ No more rows to process!")
                break
            
            print(f"\n📦 Processing batch of {len(rows)} documents with {NUM_WORKERS} workers...")
            
            # Use imap_unordered for efficient processing
            results_iterator = pool.imap_unordered(process_document_proxy, rows)
            
            progress_bar = tqdm(results_iterator, total=len(rows), desc=f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")
            
            for result in progress_bar:
                processed += 1
                doc_id = result['id']

                if result['success']:
                    try:
                        session.execute(
                            text("UPDATE disclosures SET embedding = :v WHERE id = :i"),
                            {"v": result['embedding'], "i": doc_id}
                        )
                        embedded += 1
                    except Exception as db_e:
                        errors += 1
                        failed_ids.add(doc_id)
                        print(f"   ❌ DB Error for ID {doc_id}: {db_e}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    if "File not found" in error_msg:
                        skipped += 1
                    else:
                        errors += 1
                    failed_ids.add(doc_id)
                    # Optionally log the specific error
                    # print(f"   ❌ Worker Error for ID {doc_id}: {error_msg}")

                progress_bar.set_description(f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")

            session.commit()
            gc.collect()

    pool.close()
    pool.join()
    
    print(f"\n🎉 EMBEDDING PROCESS COMPLETE!")
    print(f"   📊 Total processed: {processed}")
    print(f"   ✅ Successfully embedded: {embedded}")
    print(f"   ❌ Errors (worker or DB): {errors}")
    print(f"   ⏭️  Skipped (file not found): {skipped}")
    
    # Final database state verification
    with Session() as session:
        count = session.execute(text("SELECT COUNT(embedding) FROM disclosures")).scalar_one()
        total = session.execute(text("SELECT COUNT(*) FROM disclosures")).scalar_one()
        print(f"\n📈 FINAL DATABASE STATE: {count}/{total} documents with embeddings ({count/total*100:.1f}%)")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        multiprocessing.set_start_method('spawn', force=True)
    
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