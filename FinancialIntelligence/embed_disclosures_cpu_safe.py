#!/usr/bin/env python3
"""
CPU-Safe Embedding Script with Robust Error Handling
==================================================

CPU-only version designed to eliminate segmentation faults from GPU library conflicts.
Focuses on stability and reliability over speed.

Key features:
1. CPU-only processing to avoid GPU driver conflicts
2. Conservative memory management
3. Robust PDF processing with multiple fallbacks
4. Incremental processing with recovery capabilities
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
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------

load_dotenv()              

# Configuration
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Native 1024 dimensions
VECTOR_SIZE = 1024         # ✅ Compatible with current database
CHUNK_CHARS = 2000         # Smaller chunks for stability
BATCH_LIMIT = 20           # Smaller batches for CPU processing
MAX_RETRIES = 1            # Conservative retry count
MEMORY_LIMIT_MB = 4000     # Memory usage limit

print(f"🔧 EMBEDDING CONFIGURATION (CPU-Safe):")
print(f"   Model: {EMBED_MODEL}")
print(f"   Vector Size: {VECTOR_SIZE} dimensions")
print(f"   Chunk Size: {CHUNK_CHARS} characters")
print(f"   Batch Size: {BATCH_LIMIT} documents")
print(f"   Memory Limit: {MEMORY_LIMIT_MB}MB")

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def initialize_cpu_model():
    """Initialize the embedding model with CPU-only support."""
    try:
        # Force CPU usage by setting environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "4"
        
        from sentence_transformers import SentenceTransformer
        
        print("🚀 Initializing CPU-only embedding model...")
        
        # Initialize model with explicit CPU device
        model = SentenceTransformer(EMBED_MODEL, device='cpu')
        model_dim = model.get_sentence_embedding_dimension()
        
        print(f"   ✅ Model loaded: {model_dim} dimensions on CPU")
        
        if model_dim != VECTOR_SIZE:
            raise ValueError(f"Model dimension ({model_dim}) != expected ({VECTOR_SIZE})")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        raise

def extract_text_conservative(pdf_path: str) -> str:
    """Extract text with maximum safety and fallbacks."""
    # Skip very large files immediately
    try:
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            print(f"   ⚠️  Skipping large file: {file_size_mb:.1f}MB")
            return ""
    except:
        return ""
    
    text_parts = []
    
    # Try multiple extraction methods in order of safety
    methods = [
        ("pypdf", extract_with_pypdf),
        ("pdfplumber", extract_with_pdfplumber)
    ]
    
    for method_name, extract_func in methods:
        try:
            text = extract_func(pdf_path)
            if text and text.strip():
                print(f"   ✅ Extracted text using {method_name}")
                return text
        except Exception as e:
            print(f"   ⚠️  {method_name} failed: {str(e)[:100]}")
            continue
    
    return ""

def extract_with_pypdf(pdf_path: str) -> str:
    """Extract using pypdf (most stable)."""
    from pypdf import PdfReader
    
    text_parts = []
    reader = PdfReader(pdf_path)
    
    # Limit to first 10 pages for safety
    max_pages = min(10, len(reader.pages))
    
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception:
            continue
            
        # Break early if we have enough text
        if len(" ".join(text_parts)) > 5000:
            break
    
    return "\n".join(text_parts)

def extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract using pdfplumber (more features but less stable)."""
    import pdfplumber
    
    text_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Limit to first 5 pages with pdfplumber
        max_pages = min(5, len(pdf.pages))
        
        for page in pdf.pages[:max_pages]:
            try:
                # Simple text extraction only
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception:
                continue
                
            # Break early if we have enough text
            if len(" ".join(text_parts)) > 3000:
                break
    
    return "\n".join(text_parts)

def prepare_text_safe(text: str, title: str = "") -> str:
    """Prepare text for embedding with conservative limits."""
    if not text:
        return title or ""
    
    # Clean and normalize text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Combine title and text
    full_text = f"{title} {text}" if title else text
    
    # Conservative truncation
    max_length = 2000  # Smaller limit for CPU processing
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
    
    return full_text

def process_document_safe(doc_id, pdf_path, title, model):
    """Process a single document with maximum safety."""
    try:
        # Check memory usage before processing
        current_memory = get_memory_usage()
        if current_memory > MEMORY_LIMIT_MB:
            gc.collect()  # Force garbage collection
            current_memory = get_memory_usage()
            if current_memory > MEMORY_LIMIT_MB:
                return {
                    'id': doc_id,
                    'success': False,
                    'error': f'Memory limit exceeded: {current_memory:.1f}MB',
                    'embedding': None
                }
        
        # Extract text conservatively
        text = extract_text_conservative(pdf_path)
        
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
        prepared_text = prepare_text_safe(text, title or "")
        
        # Generate embedding (no chunking to keep it simple)
        doc_vec = model.encode(
            [prepared_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1  # Process one at a time
        )[0].astype(np.float32)
        
        # Verify embedding dimensions
        if len(doc_vec) != VECTOR_SIZE:
            return {
                'id': doc_id,
                'success': False,
                'error': f'Embedding dimension mismatch: {len(doc_vec)} != {VECTOR_SIZE}',
                'embedding': None
            }
        
        return {
            'id': doc_id,
            'success': True,
            'error': None,
            'embedding': doc_vec.tolist()
        }
        
    except Exception as e:
        return {
            'id': doc_id,
            'success': False,
            'error': str(e),
            'embedding': None
        }

def main():
    """Main processing function."""
    print("🔧 Initializing CPU-safe embedding system...")
    
    # Initialize CPU model
    try:
        model = initialize_cpu_model()
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

    print(f"\n📊 Starting embedding process...")
    print(f"   Target: Documents without embeddings")
    print(f"   Device: CPU")

    while True:  # Process until no more rows
        with Session() as session:
            # Get next batch
            query = """SELECT id, pdf_path, title
                       FROM disclosures
                       WHERE embedding IS NULL AND pdf_path IS NOT NULL
                       ORDER BY id LIMIT :lim"""
            
            if failed_ids:
                failed_list = list(failed_ids)
                placeholders = ','.join([':id' + str(i) for i in range(len(failed_list))])
                query = f"""SELECT id, pdf_path, title
                           FROM disclosures
                           WHERE embedding IS NULL AND pdf_path IS NOT NULL
                           AND id NOT IN ({placeholders})
                           ORDER BY id LIMIT :lim"""
                params = {"lim": BATCH_LIMIT}
                for i, failed_id in enumerate(failed_list):
                    params[f'id{i}'] = failed_id
            else:
                params = {"lim": BATCH_LIMIT}
            
            rows = session.execute(text(query), params).all()

            if not rows:
                print("✅ No more rows to process!")
                break

            print(f"\n📦 Processing batch of {len(rows)} documents...")
            
            # Process documents one by one
            for doc_id, pdf_path, title in tqdm(rows, desc=f"Embedding (✅ {embedded}, ❌ {errors}, ⏭️ {skipped})"):
                processed += 1
                
                print(f"🔍 Processing ID {doc_id}: {os.path.basename(pdf_path) if pdf_path else 'No path'}")
                
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

                # Process document
                result = process_document_safe(doc_id, pdf_path, title, model)
                
                if result['success']:
                    # Update database
                    session.execute(
                        text("UPDATE disclosures SET embedding = :v WHERE id = :i"),
                        {"v": result['embedding'], "i": doc_id}
                    )
                    
                    embedded += 1
                    
                    # Show first embedding stats
                    if embedded == 1:
                        emb_array = np.array(result['embedding'])
                        print(f"   🔍 First embedding check:")
                        print(f"       Dimension: {len(emb_array)} (expected: {VECTOR_SIZE})")
                        print(f"       Type: {emb_array.dtype}")
                        print(f"       Range: [{emb_array.min():.3f}, {emb_array.max():.3f}]")
                    
                    print(f"   ✅ Successfully embedded document {doc_id}")
                else:
                    print(f"   ❌ Processing failed: {result['error']}")
                    errors += 1
                    failed_ids.add(doc_id)
                
                # Force garbage collection after each document
                gc.collect()
                
                # Monitor memory usage
                current_memory = get_memory_usage()
                if current_memory > MEMORY_LIMIT_MB * 0.8:  # 80% warning threshold
                    print(f"   ⚠️  High memory usage: {current_memory:.1f}MB")
                    gc.collect()
                
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
    print(f"   ❌ Errors: {errors}")
    print(f"   ⏭️  Skipped (no file/path): {skipped}")
    
    if failed_ids:
        print(f"   ❌ Failed IDs: {len(failed_ids)} documents")
        if len(failed_ids) <= 20:
            print(f"      Failed IDs: {sorted(list(failed_ids))}")

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