#!/usr/bin/env python3
"""
Populate the `embedding` column of the `disclosures` table.

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
import pdfplumber                       # machine-generated PDFs :contentReference[oaicite:1]{index=1}
from pypdf import PdfReader             # extra robustness :contentReference[oaicite:2]{index=2}
from pdf2image import convert_from_path # OCR fallback step 1 :contentReference[oaicite:3]{index=3}
import pytesseract                      # OCR fallback step 2 :contentReference[oaicite:4]{index=4}
from openai import OpenAI               # official SDK :contentReference[oaicite:5]{index=5}

# ---------------------------------------------------------------------------

load_dotenv()              # development convenience
EMBED_MODEL = "text-embedding-3-large"
VECTOR_SIZE = 1536         # must match your table definition
CHUNK_CHARS = 4000         # safe for v3 model 8-k token limit
BATCH_LIMIT = 500          # rows per DB round-trip

# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Try text extractor, then OCR if needed."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        pass
    # OCR fallback
    images = convert_from_path(pdf_path, dpi=300)
    return "\n".join(pytesseract.image_to_string(img) for img in images)

def chunk(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i : i + size]

def embed_texts(texts, client: OpenAI):
    """Return a list of np.float32 vectors (1536 dims)."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=VECTOR_SIZE  # **** Option 1 fix ****
    )
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

# ---------------------------------------------------------------------------

def main():
    engine = create_engine(os.environ["PG_DSN"])  # standard SQLAlchemy DSN :contentReference[oaicite:6]{index=6}
    Session = sessionmaker(bind=engine)
    client = OpenAI()

    with Session() as session:
        rows = session.execute(
            text("""SELECT id, pdf_path
                    FROM disclosures
                    WHERE embedding IS NULL
                    ORDER BY id
                    LIMIT :lim"""),
            {"lim": BATCH_LIMIT},
        ).all()

        for did, path in tqdm(rows, desc="Embedding rows"):
            txt = extract_text(path)
            if not txt.strip():
                continue

            chunks = list(chunk(txt, CHUNK_CHARS))
            vecs = embed_texts(chunks, client)
            doc_vec = np.mean(vecs, axis=0)

            # optional sanity print on first doc
            if did == rows[0][0]:
                print("Vector length =", len(doc_vec))  # should be 1536

            session.execute(
                text("UPDATE disclosures SET embedding = :v WHERE id = :i"),
                {"v": doc_vec.tolist(), "i": did},
            )
        session.commit()

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Completed in {time.time() - t0:.1f}s")

