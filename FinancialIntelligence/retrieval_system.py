# ============================================================
# retrieval_and_agent.py  (©2025, MIT License)
# ============================================================

import os, json, asyncio, logging, hashlib, re, math
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import asyncpg, numpy as np, torch, redis.asyncio as aioredis
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------
@dataclass(slots=True)
class RetrievalConfig:
    pg_dsn: str
    redis_url: str
    dense_model: str = "intfloat/multilingual-e5-large"
    colbert_model: str = "lightonai/Reason-ModernColBERT"
    cache_seconds: int = 3_600
    dense_threshold: float = 0.05
    colbert_threshold: float = 0.25
    ivfflat_lists: int = 100
    batch: int = 16
    max_seq: int = 512

# ------------------------------------------------------------------
# 2) DATA CLASSES
# ------------------------------------------------------------------
class DocumentChunk(BaseModel):
    content: str
    page: int
    chunk_id: str
    meta: Dict[str, Any]

class RetrievalResult(BaseModel):
    id: int
    code: str
    name: str
    title: str
    date: date
    score: float
    ctx: str
    pdf: str

# ------------------------------------------------------------------
# 3) RETRIEVAL SYSTEM
# ------------------------------------------------------------------
class FinancialRetrievalSystem:
    """Tiered retrieval: pg→dense, Redis→ColBERT."""

    def __init__(self, cfg: RetrievalConfig):
        self.cfg, self.log = cfg, logging.getLogger(__name__)
        self.pg: Optional[asyncpg.Pool] = None
        self.rd: Optional[aioredis.Redis] = None
        self.dense = SentenceTransformer(cfg.dense_model)
        self.colbert_tok = AutoTokenizer.from_pretrained(cfg.colbert_model)
        self.colbert = AutoModel.from_pretrained(cfg.colbert_model).eval()

    # ----------
    # bootstrap
    async def init(self):
        self.pg = await asyncpg.create_pool(self.cfg.pg_dsn, min_size=4, max_size=12)
        # self.rd = aioredis.from_url(self.cfg.redis_url)  # Temporarily disabled
        self.rd = None  # Disable Redis for now
        await self._ensure_index()

    async def close(self):
        await self.pg.close()
        if self.rd is not None:
            await self.rd.close()

    async def _ensure_index(self):
        async with self.pg.acquire() as c:
            await c.execute(
                f"""DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE indexname = 'disclosures_embedding_ivfflat_idx'
                    ) THEN
                        CREATE INDEX disclosures_embedding_ivfflat_idx
                        ON disclosures
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = {self.cfg.ivfflat_lists});
                    END IF;
                END$$;"""
            )

    # ----------
    # embeddings
    def _dense(self, txt: str) -> np.ndarray:
        emb = self.dense.encode([txt], normalize_embeddings=True, show_progress_bar=False)[0]
        # Pad to 1536 dimensions if needed to match database schema
        if len(emb) < 1536:
            # Pad with zeros and renormalize to maintain similarity properties
            padded = np.zeros(1536)
            padded[:len(emb)] = emb
            # Renormalize the padded vector
            norm = np.linalg.norm(padded)
            if norm > 0:
                padded = padded / norm
            return padded
        elif len(emb) > 1536:
            # Truncate to 1536 and renormalize
            truncated = emb[:1536]
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
            return truncated
        return emb

    def _colbert_vecs(self, txt: str) -> np.ndarray:
        tok = self.colbert_tok(txt, truncation=True, max_length=self.cfg.max_seq,
                               padding=True, return_tensors="pt")
        with torch.no_grad():
            out = self.colbert(**tok).last_hidden_state[0]
        emb = out / torch.norm(out, dim=1, keepdim=True)
        return emb.cpu().numpy()

    # ----------
    # query classifier (very light)
    def _query_kind(self, q: str) -> str:
        ql = q.lower()
        if any(w in ql for w in ["why", "impact", "原因", "影響"]):
            return "reason"
        if any(w in ql for w in ["compare", "vs", "比較"]):
            return "analytical"
        return "simple"

    # ----------
    # SQL helpers
    def _build_where(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        cond, params = ["embedding IS NOT NULL"], []
        p = 2  # 1 is taken by the embedding parameter
        if codes := filters.get("company_codes"):
            cond.append(f"company_code = ANY(${p})")
            params.append(codes); p += 1
        if cat := filters.get("category"):
            cond.append(f"category = ANY(${p})")
            params.append(cat); p += 1
        if sub := filters.get("subcategory"):
            cond.append(f"subcategory = ANY(${p})")
            params.append(sub); p += 1
        if dr := filters.get("date_range"):
            cond.append(f"disclosure_date BETWEEN ${p} AND ${p+1}")
            params.extend(dr); p += 2
        return " AND ".join(cond), params

    # ----------
    # dense retrieval
    async def _dense_retrieve(self, q: str, flt: Dict[str, Any], k: int) -> List[RetrievalResult]:
        emb = self._dense(q).tolist()
        wc, prms = self._build_where(flt)
        # Convert embedding to string format for PostgreSQL vector type
        emb_str = '[' + ','.join(map(str, emb)) + ']'
        sql = f"""
        SELECT id, company_code, company_name, title, disclosure_date, pdf_path,
               1 - (embedding <=> $1::vector) AS score
        FROM disclosures WHERE {wc}
          AND (1 - (embedding <=> $1::vector)) > {self.cfg.dense_threshold}
        ORDER BY embedding <=> $1::vector LIMIT {k};
        """
        async with self.pg.acquire() as c:
            rows = await c.fetch(sql, emb_str, *prms)
        return [RetrievalResult(id=r[0], code=r[1], name=r[2], title=r[3],
                                date=r[4], pdf=r[5], score=r[6],
                                ctx="") for r in rows]  # Empty context for now

    # ----------
    # reasoning retrieval (colbert rerank)
    async def _reason_retrieve(self, q: str, flt: Dict[str, Any], k: int) -> List[RetrievalResult]:
        # step-1: recall 150 docs w/ dense
        cand = await self._dense_retrieve(q, flt, 150)
        if not cand:
            return []
        qv = self._colbert_vecs(q)
        scored: List[Tuple[float, RetrievalResult]] = []

        for r in cand:
            key = f"colbert:{r.id}"
            vecs = None
            if self.rd is not None:
                vecs = await self.rd.get(key)
            
            if vecs is None:
                # fetch from pdf once, cache if Redis available
                text = r.ctx if r.ctx else Path(r.pdf).read_text(encoding="utf-8", errors="ignore")[:8_000]
                dv = self._colbert_vecs(text).astype(np.float32)
                if self.rd is not None:
                    await self.rd.set(key, dv.tobytes())
            else:
                dv = np.frombuffer(vecs, dtype=np.float32).reshape(-1, 768)

            sim = np.max(qv @ dv.T, axis=1).mean()
            if sim >= self.cfg.colbert_threshold:
                scored.append((sim, r))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for s, r in scored[:k]]

    # ----------
    async def search(self, q: str, *, filters=None, k=20, strategy=None) -> List[RetrievalResult]:
        filters = filters or {}
        # caching -------------------------------------------
        # Skip caching if Redis is disabled
        if self.rd is not None:
            cache_key = hashlib.md5(f"{q}|{json.dumps(filters,default=str)}|{k}".encode()).hexdigest()
            if blob := await self.rd.get(f"s:{cache_key}"):
                return [RetrievalResult.parse_raw(j) for j in json.loads(blob)]
        
        # select strategy -----------------------------------
        kind = strategy or self._query_kind(q)
        res = (await self._reason_retrieve(q, filters, k) if kind == "reason"
               else await self._dense_retrieve(q, filters, k))
        
        # cache only if Redis is available
        if self.rd is not None:
            await self.rd.setex(f"s:{cache_key}", self.cfg.cache_seconds,
                                json.dumps([r.json() for r in res]))
        return res

