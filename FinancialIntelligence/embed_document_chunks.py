#!/usr/bin/env python3
"""
Document Chunks Embedding Script
===============================

Creates 1024-dimensional embeddings for the 'content' column in the document_chunks table.
Based on embed_disclosures_fixed.py but adapted for processing text chunks instead of PDF extraction.

Key Features:
- Process-isolated embedding generation to prevent crashes
- Resume capability - tracks already embedded chunks
- Uses intfloat/multilingual-e5-large for native 1024-dim embeddings
- Batch processing with progress tracking
- Memory management and error recovery

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname

Usage:
    python embed_document_chunks.py                    # Process all unembedded chunks
    python embed_document_chunks.py --batch-size 100   # Custom batch size
    python embed_document_chunks.py --test --limit 50  # Test mode with limited chunks
"""

import os, time, sys, json, gc
import argparse
from typing import List, Dict, Any, Optional

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
from contextlib import contextmanager

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()              

# Configuration - matches existing system
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Native 1024 dimensions
VECTOR_SIZE = 1024         # ✅ Compatible with document_chunks.embedding column
BATCH_LIMIT = 100          # Process chunks in batches
WORKER_TIMEOUT = 60        # 1 minute timeout per chunk (faster than documents)
MAX_RETRIES = 2            # Maximum retries per chunk
MAX_CONTENT_LENGTH = 3000  # Maximum content length for embedding

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

# ---------------------------------------------------------------------------

def process_chunk_worker(args):
    """Worker function that processes a single document chunk in isolation.
    
    This function runs in a separate process to prevent crashes from
    affecting the main process.
    """
    chunk_id, content, chunk_index, disclosure_id = args
    
    try:
        # Suppress TensorFlow/CUDA logging in worker process
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Import heavy libraries only in worker process
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
                torch.cuda.set_per_process_memory_fraction(0.6)  # More conservative for chunks
                device = 'cuda'
                print(f"   🚀 Using GPU acceleration for chunk {chunk_id}")
            except Exception as cuda_error:
                print(f"   ⚠️  CUDA initialization failed: {cuda_error}, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
            print(f"   🖥️  Using CPU for chunk {chunk_id}")
        
        # Initialize model in worker process with proper device
        model = SentenceTransformer(EMBED_MODEL, device=device)
        model_memory = get_memory_usage()
        
        # Prepare content for embedding
        prepared_content = _prepare_content_for_embedding_worker(content)
        
        if not prepared_content.strip():
            return {
                'id': chunk_id,
                'success': False,
                'error': 'No content to embed',
                'embedding': None
            }
        
        # Generate embedding
        doc_vec = model.encode(
            [prepared_content],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].astype(np.float32)
        
        # Verify embedding dimensions
        if len(doc_vec) != VECTOR_SIZE:
            return {
                'id': chunk_id,
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
            'id': chunk_id,
            'success': True,
            'error': None,
            'embedding': doc_vec.tolist(),
            'chunk_info': {
                'disclosure_id': disclosure_id,
                'chunk_index': chunk_index,
                'content_length': len(content)
            },
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
            'id': chunk_id,
            'success': False,
            'error': str(e),
            'embedding': None
        }

def _prepare_content_for_embedding_worker(content: str) -> str:
    """Prepare chunk content for embedding in worker process."""
    if not content:
        return ""
    
    # Clean and normalize text
    content = content.replace('\n', ' ').replace('\r', ' ')
    content = ' '.join(content.split())  # Normalize whitespace
    
    # Truncate to model context limit
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH]
    
    return content

# ---------------------------------------------------------------------------

def get_embedding_progress(session) -> Dict[str, int]:
    """Get current embedding progress statistics."""
    result = session.execute(text("""
        SELECT 
            COUNT(*) as total_chunks,
            COUNT(embedding) as embedded_chunks,
            COUNT(*) - COUNT(embedding) as pending_chunks
        FROM document_chunks
        WHERE vectorize = true  -- Only count chunks marked for vectorization
    """)).fetchone()
    
    return {
        'total': result.total_chunks,
        'embedded': result.embedded_chunks,
        'pending': result.pending_chunks
    }

def get_next_batch(session, batch_size: int, failed_ids: set = None) -> List[tuple]:
    """Get next batch of chunks to embed."""
    # Build query to exclude failed IDs and get only unembedded chunks
    base_query = """
        SELECT id, content, chunk_index, disclosure_id
        FROM document_chunks
        WHERE embedding IS NULL 
          AND vectorize = true 
          AND content IS NOT NULL 
          AND char_length(content) > 10  -- Skip very short chunks
    """
    
    if failed_ids:
        failed_list = list(failed_ids)
        placeholders = ','.join([':id' + str(i) for i in range(len(failed_list))])
        query = f"{base_query} AND id NOT IN ({placeholders}) ORDER BY disclosure_id, chunk_index LIMIT :lim"
        params = {"lim": batch_size}
        for i, failed_id in enumerate(failed_list):
            params[f'id{i}'] = failed_id
    else:
        query = f"{base_query} ORDER BY disclosure_id, chunk_index LIMIT :lim"
        params = {"lim": batch_size}
    
    return session.execute(text(query), params).all()

def main():
    """Main process using multiprocessing for isolation."""
    parser = argparse.ArgumentParser(description='Embed document chunks with 1024-dim vectors')
    parser.add_argument('--batch-size', type=int, default=BATCH_LIMIT, 
                       help=f'Batch size for processing (default: {BATCH_LIMIT})')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: process limited chunks')
    parser.add_argument('--limit', type=int, default=50,
                       help='Limit chunks in test mode (default: 50)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    
    args = parser.parse_args()
    
    print("🔧 Initializing document chunks embedding system...")
    print(f"🔧 EMBEDDING CONFIGURATION:")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Vector Size: {VECTOR_SIZE} dimensions")
    print(f"   Batch Size: {args.batch_size} chunks")
    print(f"   Worker Timeout: {WORKER_TIMEOUT} seconds")
    print(f"   Max Content Length: {MAX_CONTENT_LENGTH} characters")
    print(f"   Test Mode: {args.test}")
    if args.test:
        print(f"   Test Limit: {args.limit} chunks")
    
    # Monitor initial memory usage
    initial_memory = get_memory_usage()
    print(f"   💾 Initial memory usage: {initial_memory:.1f}MB")
    
    # Database connection
    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)

    # Get initial statistics
    with Session() as session:
        progress = get_embedding_progress(session)
        print(f"\n📊 INITIAL STATE:")
        print(f"   Total chunks (vectorize=true): {progress['total']}")
        print(f"   Already embedded: {progress['embedded']}")
        print(f"   Pending chunks: {progress['pending']}")
        
        if progress['pending'] == 0:
            print("✅ All chunks are already embedded!")
            return

    processed = 0
    embedded = 0
    skipped = 0
    errors = 0
    failed_ids = set()
    start_time = time.time()

    print(f"\n📊 Starting embedding process...")
    print(f"   Target: {progress['pending']} unembedded chunks")
    print(f"   Worker Timeout: {WORKER_TIMEOUT} seconds")
    print(f"   Max Retries: {MAX_RETRIES}")

    while True:  # Process until no more chunks
        with Session() as session:
            # Get next batch
            if args.test and processed >= args.limit:
                print(f"✅ Test mode limit reached: {args.limit} chunks")
                break
                
            batch_size = min(args.batch_size, args.limit - processed) if args.test else args.batch_size
            rows = get_next_batch(session, batch_size, failed_ids)

            if not rows:
                print("✅ No more chunks to process!")
                break

            print(f"\n📦 Processing batch of {len(rows)} chunks...")
            
            # Use persistent worker pool for the entire batch
            with multiprocessing.Pool(processes=args.workers) as pool:
                progress_bar = tqdm(rows, desc=f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")
                for chunk_id, content, chunk_index, disclosure_id in progress_bar:
                    processed += 1
                    
                    print(f"\n🔍 Processing chunk ID {chunk_id} (disclosure {disclosure_id}, chunk {chunk_index})")
                    
                    # Quick validation
                    if not content or len(content.strip()) <= 10:
                        print(f"   📝 Content too short for chunk ID {chunk_id}")
                        skipped += 1
                        failed_ids.add(chunk_id)
                        continue

                    # Process chunk with retries
                    success = False
                    for attempt in range(MAX_RETRIES + 1):
                        if attempt > 0:
                            print(f"   🔄 Retry {attempt}/{MAX_RETRIES}")
                        
                        try:
                            # Submit work to persistent process pool with timeout
                            async_result = pool.apply_async(
                                process_chunk_worker,
                                [(chunk_id, content, chunk_index, disclosure_id)]
                            )
                            
                            # Wait for result with timeout
                            result = async_result.get(timeout=WORKER_TIMEOUT)
                            
                            if result['success']:
                                # Update database
                                session.execute(
                                    text("UPDATE document_chunks SET embedding = :v WHERE id = :i"),
                                    {"v": result['embedding'], "i": chunk_id}
                                )
                                
                                embedded += 1
                                success = True
                                
                                # Show first embedding stats
                                if embedded == 1:
                                    emb_array = np.array(result['embedding'])
                                    print(f"   🔍 First embedding check:")
                                    print(f"       Dimension: {len(emb_array)} (expected: {VECTOR_SIZE})")
                                    print(f"       Type: {emb_array.dtype}")
                                    print(f"       Range: [{emb_array.min():.3f}, {emb_array.max():.3f}]")
                                
                                print(f"   ✅ Successfully embedded chunk {chunk_id}")
                                break
                            else:
                                print(f"   ❌ Worker failed: {result['error']}")
                                if attempt == MAX_RETRIES:
                                    errors += 1
                                    failed_ids.add(chunk_id)
                        
                        except multiprocessing.TimeoutError:
                            print(f"   ⏰ Worker timeout after {WORKER_TIMEOUT}s")
                            if attempt == MAX_RETRIES:
                                errors += 1
                                failed_ids.add(chunk_id)
                        
                        except Exception as e:
                            print(f"   💥 Process error: {e}")
                            if attempt == MAX_RETRIES:
                                errors += 1
                                failed_ids.add(chunk_id)
                    
                    # Force garbage collection after each chunk
                    gc.collect()
                    
                    # Update progress bar description with the latest stats
                    progress_bar.set_description(f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})")

            session.commit()
            
            # Memory monitoring after batch
            current_memory = get_memory_usage()
            elapsed_time = time.time() - start_time
            chunks_per_sec = processed / elapsed_time if elapsed_time > 0 else 0
            
            print(f"   ✅ Batch completed. Embedded: {embedded}, Errors: {errors}, Skipped: {skipped}")
            print(f"   💾 Memory usage: {current_memory:.1f}MB")
            print(f"   ⏱️  Speed: {chunks_per_sec:.1f} chunks/sec")
            
            # Force garbage collection after each batch
            gc.collect()

    print(f"\n🎉 EMBEDDING PROCESS COMPLETE!")
    print(f"   📊 Total processed: {processed}")
    print(f"   ✅ Successfully embedded: {embedded}")
    print(f"   ❌ Errors (worker crashes/timeouts): {errors}")
    print(f"   ⏭️  Skipped (no content/too short): {skipped}")
    
    if failed_ids:
        print(f"   ❌ Failed chunk IDs: {len(failed_ids)} chunks")
        if len(failed_ids) <= 10:
            print(f"      Failed IDs: {sorted(list(failed_ids))}")
        else:
            print(f"      Sample failed IDs: {sorted(list(failed_ids))[:10]}...")

    # Verify final state
    with Session() as session:
        final_progress = get_embedding_progress(session)
        print(f"\n📈 FINAL DATABASE STATE:")
        print(f"   Total chunks (vectorize=true): {final_progress['total']}")
        print(f"   Chunks with embeddings: {final_progress['embedded']}")
        print(f"   Remaining chunks: {final_progress['pending']}")
        if final_progress['total'] > 0:
            print(f"   Coverage: {final_progress['embedded']/final_progress['total']*100:.1f}%")

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