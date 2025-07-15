#!/usr/bin/env python3
"""
Improved version of embed_disclosures.py with better error handling and progress tracking.

Environment variables (set in shell or .env):
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname
  OPENAI_API_KEY   your OpenAI key
"""

import os, time, pathlib
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pdfplumber                       
from pypdf import PdfReader             
from pdf2image import convert_from_path 
import pytesseract                      
from openai import OpenAI               

# ---------------------------------------------------------------------------

load_dotenv()              
EMBED_MODEL = "text-embedding-3-large"
VECTOR_SIZE = 1536         
CHUNK_CHARS = 4000         
BATCH_LIMIT = 500          

# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Try text extractor, then OCR if needed."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if text.strip():  # Return if we got text
                return text
    except Exception as e:
        print(f"  pdfplumber failed: {e}")
    
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if text.strip():  # Return if we got text
            return text
    except Exception as e:
        print(f"  pypdf failed: {e}")
    
    # OCR fallback
    try:
        print(f"  Trying OCR for {os.path.basename(pdf_path)}...")
        images = convert_from_path(pdf_path, dpi=300)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return text
    except Exception as e:
        print(f"  OCR failed: {e}")
        return ""

def chunk(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i : i + size]

def embed_texts(texts, client: OpenAI):
    """Return a list of np.float32 vectors (1536 dims)."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=VECTOR_SIZE
    )
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

# ---------------------------------------------------------------------------

def main():
    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)
    client = OpenAI()

    # Get total count for progress tracking
    with Session() as session:
        total_count = session.execute(
            text("SELECT COUNT(*) FROM disclosures WHERE embedding IS NULL")
        ).scalar()
        print(f"Total rows to process: {total_count}")

    processed = 0
    skipped_no_file = 0
    skipped_no_text = 0
    embedded = 0
    errors = 0
    
    # Keep track of IDs we've already tried and failed
    failed_ids = set()

    while True:
        with Session() as session:
            # Build the query to exclude failed IDs
            base_query = """SELECT id, pdf_path
                           FROM disclosures
                           WHERE embedding IS NULL"""
            
            if failed_ids:
                # Convert set to list for SQL IN clause
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
                print("No more rows to process!")
                break

            for did, path in tqdm(rows, desc=f"Processing batch (embedded: {embedded}, skipped: {skipped_no_file + skipped_no_text})"):
                processed += 1
                
                # Check if file exists
                if not os.path.exists(path):
                    print(f"File not found: {path}")
                    skipped_no_file += 1
                    failed_ids.add(did)  # Remember this ID failed
                    continue

                try:
                    txt = extract_text(path)
                    if not txt.strip():
                        print(f"No text extracted from: {os.path.basename(path)}")
                        skipped_no_text += 1
                        failed_ids.add(did)  # Remember this ID failed
                        continue

                    chunks = list(chunk(txt, CHUNK_CHARS))
                    vecs = embed_texts(chunks, client)
                    doc_vec = np.mean(vecs, axis=0)

                    # optional sanity print on first doc
                    if embedded == 0:
                        print(f"Vector length = {len(doc_vec)}")  # should be 1536

                    session.execute(
                        text("UPDATE disclosures SET embedding = :v WHERE id = :i"),
                        {"v": doc_vec.tolist(), "i": did},
                    )
                    embedded += 1

                except Exception as e:
                    print(f"Error processing ID {did}: {e}")
                    errors += 1
                    failed_ids.add(did)  # Remember this ID failed
                    
            session.commit()
            print(f"Batch completed. Embedded: {embedded}, Skipped (no file): {skipped_no_file}, Skipped (no text): {skipped_no_text}, Errors: {errors}")

    print(f"\nFinal summary:")
    print(f"Total processed: {processed}")
    print(f"Successfully embedded: {embedded}")
    print(f"Skipped (file not found): {skipped_no_file}")
    print(f"Skipped (no text): {skipped_no_text}")
    print(f"Errors: {errors}")
    
    if failed_ids:
        print(f"Failed IDs (for reference): {sorted(list(failed_ids))[:10]}{'...' if len(failed_ids) > 10 else ''}")
        print(f"Total failed IDs: {len(failed_ids)}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Completed in {time.time() - t0:.1f}s") 