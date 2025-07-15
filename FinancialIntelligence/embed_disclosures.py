#!/usr/bin/env python3
"""
Robust pgvector embedder – no random seg-faults
-----------------------------------------------
* One persistent worker per process; model is loaded once.
* Uses spawn, not fork/forkserver.
* PG table: disclosures(id INT PK, pdf_path TEXT, title TEXT, embedding VECTOR).
"""

import os, sys, gc, signal, concurrent.futures as cf
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import multiprocessing as mp


import warnings, logging
warnings.filterwarnings("ignore",
        message=r"CropBox missing from /Page, defaulting to MediaBox")

# Suppress pdfminer logging that repeats the same CropBox message.
for name in ("pdfminer", "pdfminer.layout", "pdfminer.pdfpage", "pdfplumber"):  # be thorough
    logging.getLogger(name).setLevel(logging.ERROR)

load_dotenv()    

# ---------- STATIC CONFIG ----------------------------------------------------
PG_DSN         = os.environ["PG_DSN"]          # postgresql+psycopg2://...
EMBED_MODEL    = "intfloat/multilingual-e5-large"
VECTOR_SIZE    = 1024
WORKERS        = 1        # >=1, tune to taste (GPU:1; CPU:cores-1)
WORKER_TIMEOUT = 300      # s per document
MAX_PAGES      = 20       # hard cap on pages read from a PDF
# -----------------------------------------------------------------------------


# --- keep heavy stuff in worker only -----------------------------------------
def _worker_init(gpu_ok=True):
    """Runs once per worker process."""
    import os, torch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"]       = "1"

    device = "cuda" if gpu_ok and torch.cuda.is_available() else "cpu"
    global _model
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBED_MODEL, device=device)
    torch.set_num_threads(1)  # even when on CPU
    print(f"[{os.getpid()}] model ready on {device}")


def _extract_text(pdf_path, max_pages, ocr=False):
    import pdfplumber, io
    from pathlib import Path

    txt_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            t = page.extract_text(x_tolerance=5, y_tolerance=5)
            if t:
                txt_parts.append(t)

    if txt_parts or not ocr:
        return " ".join(txt_parts)

    # ---------- fallback 1: pdfminer.six -----------------------------
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        t = pdfminer_extract(str(pdf_path), maxpages=max_pages)
        if t.strip():
            return t
    except Exception:
        pass

    # ---------- fallback 2: OCR via pytesseract ----------------------
    try:
        import pytesseract, PIL.Image
        for page in pdf.pages[:3]:            # OCR only first 3 pages
            img = page.to_image(resolution=200).original
            t  = pytesseract.image_to_string(PIL.Image.fromarray(img))
            if t.strip():
                txt_parts.append(t)
    except Exception:
        pass

    return " ".join(txt_parts)


def _embed_one(job):
    """Runs in worker process – no DB access here."""
    doc_id, pdf_path, title = job
    import pdfplumber

    try:
        # ---- extract lightweight text ---------------------------------------
        text = _extract_text(pdf_path, MAX_PAGES, ocr=True)

        if not text.strip():
            raise ValueError("empty text after extraction")

        # ---- embed -----------------------------------------------------------
        vec = _model.encode(
            text[:3000],   # clip; e5-large has ~4k token window
            normalize_embeddings=True
        ).astype(np.float32)

        if vec.shape[0] != VECTOR_SIZE:
            raise ValueError(f"dim {vec.shape[0]} != {VECTOR_SIZE}")

        return (doc_id, vec.tolist(), None)

    except Exception as e:   # normal Python exceptions
        return (doc_id, None, str(e))


# ---------- DB helpers --------------------------------------------------------
_engine  = create_engine(PG_DSN)
Session  = sessionmaker(bind=_engine)

def _fetch_batch(session, limit=1000):
    rows = session.execute(text("""
        SELECT id, pdf_path, title
        FROM   disclosures
        WHERE  embedding IS NULL
        LIMIT  :lim
    """), {"lim": limit}).fetchall()
    return [(r.id, r.pdf_path, r.title) for r in rows]


def _update_embeddings(session, results):
    """Persist embeddings.

    We batch‐update successfully generated vectors via executemany.

    * For every successful embedding we send an UPDATE with
      ``embedding = :v::vector`` where ``:v`` is a string representation
      accepted by pgvector, e.g. ``'[1,2,3]'``.
    * For failures we simply mark the row with ``NULL`` so it can be
      retried later (or you may choose to set a separate flag instead).
    """

    ok  = [r for r in results if r[2] is None]
    err = [r for r in results if r[2] is not None]

    # ---- successful rows --------------------------------------------------
    if ok:
        rows = []
        for doc_id, vec, _ in ok:
            # convert list[float] -> pgvector compatible literal
            vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
            rows.append({"id": doc_id, "v": vec_literal})

        # executemany style; SQLAlchemy will batch for us
        session.execute(
            text("UPDATE disclosures SET embedding = :v WHERE id = :id"),
            rows,
        )

    # ---- permanently failed rows -----------------------------------------
    if err:
        session.execute(
            text("UPDATE disclosures SET embedding = NULL WHERE id = :id"),
            [{"id": i} for i, _, _ in err],
        )

    return len(ok), len(err)


# ---------- main loop ---------------------------------------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"]       = "1"

    with Session() as sess, \
        cf.ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=_worker_init,
            initargs=(True,),           # tell worker gpu ok
            mp_context=mp.get_context("spawn"),
        ) as pool:

        total_ok = total_err = 0

        batch = _fetch_batch(sess, 1000)
        while batch:
            futures = {
                pool.submit(_embed_one, job): job[0] for job in batch
            }
            results = []
            for fut in cf.as_completed(futures, timeout=WORKER_TIMEOUT * len(batch)):
                try:
                    results.append(fut.result(timeout=0.1))
                except cf.TimeoutError:
                    results.append((futures[fut], None, "timeout"))
                except Exception as e:
                    results.append((futures[fut], None, f"{type(e).__name__}: {e}"))

            ok, err = _update_embeddings(sess, results)
            sess.commit()

            total_ok  += ok
            total_err += err
            print(f"Batch done – ok:{ok} err:{err} (cum ok:{total_ok} err:{total_err})")

            batch = _fetch_batch(sess, 1000)

    print(f"Finished. Embedded {total_ok}, permanently failed {total_err}.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # always

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
