#!/usr/bin/env python3
"""
Fast Document Chunks Embedding Script
====================================

Optimized version for speed with hybrid approach:
- Model loaded once per large batch (10,000 chunks) instead of per chunk
- Aggressive memory management and cleanup
- Frequent database commits (every 100 chunks)
- Error recovery without crashing entire process
- Expected 10-20x speed improvement over process-isolated version

TRADE-OFFS:
- Faster processing but higher memory usage
- Risk of losing current batch on crash (but can restart and resume)
- Less isolation but with automatic recovery mechanisms

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname

Usage:
    python embed_document_chunks_fast.py                    # Process all unembedded chunks
    python embed_document_chunks_fast.py --batch-size 1000  # Custom batch size
    python embed_document_chunks_fast.py --test --limit 100 # Test mode
"""

import os, time, sys, json, gc
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Suppress TensorFlow and CUDA verbose logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import psutil

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()

# Configuration - optimized for speed
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Native 1024 dimensions
VECTOR_SIZE = 1024
MEGA_BATCH_SIZE = 10000        # Load model once per 10K chunks
COMMIT_BATCH_SIZE = 100        # Commit to DB every 100 chunks
MAX_CONTENT_LENGTH = 3000      # Maximum content length for embedding
RESTART_EVERY_N_BATCHES = 50   # Restart model every 500K chunks to prevent memory leaks

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def get_gpu_memory_usage():
    """Get GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            cached = torch.cuda.memory_reserved() / 1024 / 1024      # MB
            return allocated, cached
    except:
        pass
    return 0, 0

class FastEmbeddingProcessor:
    """Fast embedding processor that reuses model for batches."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
        self.chunks_processed_with_current_model = 0
        
    def initialize_model(self):
        """Initialize the embedding model."""
        print("🚀 Initializing embedding model...")
        
        # Import heavy libraries
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.init()
                torch.cuda.set_per_process_memory_fraction(0.8)
                self.device = 'cuda'
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   🎮 Using GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
            except Exception as e:
                print(f"   ⚠️  CUDA initialization failed: {e}, using CPU")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            print(f"   🖥️  Using CPU")
        
        # Load model
        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.model_loaded = True
        self.chunks_processed_with_current_model = 0
        
        # Show memory usage after model load
        memory_usage = get_memory_usage()
        gpu_alloc, gpu_cached = get_gpu_memory_usage()
        print(f"   💾 Memory usage: {memory_usage:.1f}MB RAM")
        if gpu_alloc > 0:
            print(f"   🎮 GPU memory: {gpu_alloc:.1f}MB allocated, {gpu_cached:.1f}MB cached")
    
    def cleanup_model(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            
        # Aggressive cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        gc.collect()
        print("   🧹 Model cleaned up and memory freed")
    
    def should_restart_model(self) -> bool:
        """Check if we should restart the model to prevent memory leaks."""
        return self.chunks_processed_with_current_model >= (MEGA_BATCH_SIZE * RESTART_EVERY_N_BATCHES)
    
    def embed_chunks(self, chunks_data: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Embed a batch of chunks efficiently.
        
        Args:
            chunks_data: List of (chunk_id, content, chunk_index, disclosure_id) tuples
        
        Returns:
            List of results with embeddings
        """
        if not self.model_loaded or self.should_restart_model():
            if self.model_loaded:
                print(f"   🔄 Restarting model after {self.chunks_processed_with_current_model} chunks...")
                self.cleanup_model()
            self.initialize_model()
        
        results = []
        failed_chunks = []
        
        # Prepare content for embedding
        chunk_ids = []
        contents = []
        chunk_metadata = []
        
        for chunk_id, content, chunk_index, disclosure_id in chunks_data:
            prepared_content = self._prepare_content_for_embedding(content)
            
            if not prepared_content.strip():
                results.append({
                    'id': chunk_id,
                    'success': False,
                    'error': 'No content to embed',
                    'embedding': None
                })
                continue
            
            chunk_ids.append(chunk_id)
            contents.append(prepared_content)
            chunk_metadata.append({
                'chunk_index': chunk_index,
                'disclosure_id': disclosure_id,
                'original_length': len(content)
            })
        
        if not contents:
            return results
        
        try:
            # Batch encode all contents at once - much more efficient
            print(f"   🔢 Encoding {len(contents)} chunks in batch...")
            embeddings = self.model.encode(
                contents,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,  # Internal batch size for the model
                convert_to_tensor=False,
                convert_to_numpy=True
            )
            
            # Verify embedding dimensions and create results
            for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                if len(embedding) != VECTOR_SIZE:
                    results.append({
                        'id': chunk_id,
                        'success': False,
                        'error': f'Embedding dimension mismatch: {len(embedding)} != {VECTOR_SIZE}',
                        'embedding': None
                    })
                else:
                    results.append({
                        'id': chunk_id,
                        'success': True,
                        'error': None,
                        'embedding': embedding.astype(np.float32).tolist(),
                        'metadata': chunk_metadata[i]
                    })
            
            self.chunks_processed_with_current_model += len(contents)
            
        except Exception as e:
            print(f"   ❌ Batch embedding failed: {e}")
            # Return individual failures for all chunks in this batch
            for chunk_id in chunk_ids:
                results.append({
                    'id': chunk_id,
                    'success': False,
                    'error': f'Batch embedding error: {str(e)}',
                    'embedding': None
                })
        
        # Aggressive memory cleanup after each batch
        if self.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        gc.collect()
        
        return results
    
    def _prepare_content_for_embedding(self, content: str) -> str:
        """Prepare chunk content for embedding."""
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
        WHERE vectorize = true
    """)).fetchone()
    
    return {
        'total': result.total_chunks,
        'embedded': result.embedded_chunks,
        'pending': result.pending_chunks
    }

def get_next_mega_batch(session, mega_batch_size: int, failed_ids: set = None) -> List[tuple]:
    """Get next mega batch of chunks to embed."""
    base_query = """
        SELECT id, content, chunk_index, disclosure_id
        FROM document_chunks
        WHERE embedding IS NULL 
          AND vectorize = true 
          AND content IS NOT NULL 
          AND char_length(content) > 10
    """
    
    if failed_ids:
        failed_list = list(failed_ids)
        placeholders = ','.join([':id' + str(i) for i in range(len(failed_list))])
        query = f"{base_query} AND id NOT IN ({placeholders}) ORDER BY disclosure_id, chunk_index LIMIT :lim"
        params = {"lim": mega_batch_size}
        for i, failed_id in enumerate(failed_list):
            params[f'id{i}'] = failed_id
    else:
        query = f"{base_query} ORDER BY disclosure_id, chunk_index LIMIT :lim"
        params = {"lim": mega_batch_size}
    
    return session.execute(text(query), params).all()

def update_embeddings_batch(session, results: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Update embeddings in database for successful results."""
    successful_updates = 0
    failed_updates = 0
    
    # Prepare batch update for successful embeddings
    successful_results = [r for r in results if r['success'] and r['embedding']]
    
    if successful_results:
        try:
            # Use batch update for efficiency
            update_data = [
                {'chunk_id': r['id'], 'embedding': r['embedding']}
                for r in successful_results
            ]
            
            session.execute(text("""
                UPDATE document_chunks 
                SET embedding = :embedding 
                WHERE id = :chunk_id
            """), [{'chunk_id': d['chunk_id'], 'embedding': d['embedding']} for d in update_data])
            
            successful_updates = len(update_data)
            
        except Exception as e:
            print(f"   ❌ Batch database update failed: {e}")
            failed_updates = len(successful_results)
    
    failed_updates += len([r for r in results if not r['success']])
    
    return successful_updates, failed_updates

def main():
    """Main processing function with optimized batch approach."""
    parser = argparse.ArgumentParser(description='Fast embed document chunks with hybrid approach')
    parser.add_argument('--mega-batch-size', type=int, default=MEGA_BATCH_SIZE,
                       help=f'Mega batch size for model reuse (default: {MEGA_BATCH_SIZE})')
    parser.add_argument('--commit-batch-size', type=int, default=COMMIT_BATCH_SIZE,
                       help=f'Database commit batch size (default: {COMMIT_BATCH_SIZE})')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process limited chunks')
    parser.add_argument('--limit', type=int, default=500,
                       help='Limit chunks in test mode (default: 500)')
    
    args = parser.parse_args()
    
    print("🚀 Initializing FAST document chunks embedding system...")
    print(f"🔧 FAST EMBEDDING CONFIGURATION:")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Vector Size: {VECTOR_SIZE} dimensions")
    print(f"   Mega Batch Size: {args.mega_batch_size} chunks (model reload frequency)")
    print(f"   Commit Batch Size: {args.commit_batch_size} chunks (DB commit frequency)")
    print(f"   Max Content Length: {MAX_CONTENT_LENGTH} characters")
    print(f"   Model Restart Every: {RESTART_EVERY_N_BATCHES} mega batches")
    print(f"   Test Mode: {args.test}")
    if args.test:
        print(f"   Test Limit: {args.limit} chunks")
    
    # Initialize processor
    processor = FastEmbeddingProcessor()
    
    # Monitor initial memory
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

    # Processing counters
    total_processed = 0
    total_embedded = 0
    total_skipped = 0
    total_errors = 0
    failed_ids = set()
    start_time = time.time()

    print(f"\n🚀 Starting FAST embedding process...")
    print(f"   Target: {progress['pending']} unembedded chunks")
    print(f"   Expected speed: 10-20x faster than process-isolated version")

    try:
        while True:  # Process until no more chunks
            with Session() as session:
                # Check test mode limit
                if args.test and total_processed >= args.limit:
                    print(f"✅ Test mode limit reached: {args.limit} chunks")
                    break
                
                # Get next mega batch
                current_mega_batch_size = min(args.mega_batch_size, args.limit - total_processed) if args.test else args.mega_batch_size
                mega_batch = get_next_mega_batch(session, current_mega_batch_size, failed_ids)
                
                if not mega_batch:
                    print("✅ No more chunks to process!")
                    break
                
                print(f"\n📦 Processing MEGA BATCH of {len(mega_batch)} chunks...")
                mega_batch_start = time.time()
                
                # Process mega batch in smaller commit batches
                mega_batch_embedded = 0
                mega_batch_errors = 0
                
                for i in range(0, len(mega_batch), args.commit_batch_size):
                    commit_batch = mega_batch[i:i + args.commit_batch_size]
                    
                    print(f"   🔄 Processing commit batch {i//args.commit_batch_size + 1}/{(len(mega_batch)-1)//args.commit_batch_size + 1} ({len(commit_batch)} chunks)")
                    
                    # Process this commit batch
                    try:
                        results = processor.embed_chunks(commit_batch)
                        
                        # Update database
                        batch_embedded, batch_errors = update_embeddings_batch(session, results)
                        
                        # Commit after each batch for safety
                        session.commit()
                        
                        # Update counters
                        total_processed += len(commit_batch)
                        total_embedded += batch_embedded
                        total_errors += batch_errors
                        mega_batch_embedded += batch_embedded
                        mega_batch_errors += batch_errors
                        
                        # Track failed IDs
                        for result in results:
                            if not result['success']:
                                failed_ids.add(result['id'])
                        
                        # Show progress
                        elapsed = time.time() - start_time
                        chunks_per_sec = total_processed / elapsed if elapsed > 0 else 0
                        
                        print(f"     ✅ Batch complete: {batch_embedded} embedded, {batch_errors} errors")
                        print(f"     📊 Total progress: {total_embedded:,} embedded ({chunks_per_sec:.1f} chunks/sec)")
                        
                        # Memory monitoring
                        current_memory = get_memory_usage()
                        gpu_alloc, gpu_cached = get_gpu_memory_usage()
                        if gpu_alloc > 0:
                            print(f"     💾 Memory: {current_memory:.1f}MB RAM, {gpu_alloc:.1f}MB GPU")
                        else:
                            print(f"     💾 Memory: {current_memory:.1f}MB RAM")
                    
                    except Exception as e:
                        print(f"   ❌ Commit batch failed: {e}")
                        # Mark all chunks in this batch as failed
                        for chunk_id, _, _, _ in commit_batch:
                            failed_ids.add(chunk_id)
                        total_errors += len(commit_batch)
                        total_processed += len(commit_batch)
                
                # Mega batch summary
                mega_batch_time = time.time() - mega_batch_start
                mega_batch_speed = len(mega_batch) / mega_batch_time if mega_batch_time > 0 else 0
                
                print(f"   🎉 MEGA BATCH COMPLETE in {mega_batch_time:.1f}s:")
                print(f"      📊 {mega_batch_embedded} embedded, {mega_batch_errors} errors")
                print(f"      ⚡ Speed: {mega_batch_speed:.1f} chunks/sec for this mega batch")
                
                # Force cleanup after mega batch
                gc.collect()

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            processor.cleanup_model()
        except:
            pass

    # Final statistics
    total_time = time.time() - start_time
    avg_speed = total_processed / total_time if total_time > 0 else 0
    
    print(f"\n🎉 FAST EMBEDDING PROCESS COMPLETE!")
    print(f"   📊 Total processed: {total_processed:,}")
    print(f"   ✅ Successfully embedded: {total_embedded:,}")
    print(f"   ❌ Errors: {total_errors:,}")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds")
    print(f"   ⚡ Average speed: {avg_speed:.1f} chunks/second")
    
    if failed_ids:
        print(f"   ❌ Failed chunk IDs: {len(failed_ids)} chunks")
        print(f"      (These can be retried by running the script again)")

    # Final progress check
    with Session() as session:
        final_progress = get_embedding_progress(session)
        print(f"\n📈 FINAL DATABASE STATE:")
        print(f"   Total chunks (vectorize=true): {final_progress['total']:,}")
        print(f"   Chunks with embeddings: {final_progress['embedded']:,}")
        print(f"   Remaining chunks: {final_progress['pending']:,}")
        if final_progress['total'] > 0:
            print(f"   Coverage: {final_progress['embedded']/final_progress['total']*100:.1f}%")

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