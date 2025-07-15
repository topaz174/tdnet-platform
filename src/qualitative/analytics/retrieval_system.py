#!/usr/bin/env python3
"""
Enhanced Financial Retrieval System - Option A Implementation
Optimized for Japanese corporate disclosures with smart filtering and query enhancement.
"""

import os, json, asyncio, logging, hashlib, re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set
from collections import defaultdict

import asyncpg, numpy as np
import redis.asyncio as aioredis
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------
@dataclass(slots=True)
class EnhancedRetrievalConfig:
    pg_dsn: str
    redis_url: str
    dense_model: str = "intfloat/multilingual-e5-large"
    cache_seconds: int = 3_600
    dense_threshold: float = 0.01  # Lower threshold for better recall
    ivfflat_lists: int = 100
    max_results: int = 50
    rerank_candidates: int = 150

# ------------------------------------------------------------------
# 2) DATA CLASSES
# ------------------------------------------------------------------
class RetrievalResult(BaseModel):
    id: int
    code: str
    name: str
    title: str
    date: date
    score: float
    ctx: str
    pdf: str
    classification_l1: Optional[str] = None
    classification_l2: Optional[str] = None

class QueryClassification(BaseModel):
    query_type: str  # simple, temporal, comparative, analytical
    financial_terms: List[str]
    companies: List[str]
    time_indicators: List[str]
    metrics_mentioned: List[str]
    expanded_query: str

# ------------------------------------------------------------------
# 3) FINANCIAL KNOWLEDGE BASE
# ------------------------------------------------------------------
class FinancialKnowledgeBase:
    """Financial term expansion and classification logic"""
    
    # Japanese-English financial term mappings
    FINANCIAL_TERMS = {
        # Earnings & Performance
        "earnings": ["決算", "業績", "売上", "収益", "利益", "営業利益", "純利益", "経常利益"],
        "revenue": ["売上高", "売上", "収益", "営業収益", "売上収入"],
        "profit": ["利益", "純利益", "営業利益", "経常利益", "当期純利益"],
        "operating_income": ["営業利益", "営業収益"],
        "net_income": ["純利益", "当期純利益", "最終利益"],
        
        # Corporate Actions
        "dividend": ["配当", "配当金", "配当性向", "増配", "減配", "特別配当"],
        "buyback": ["自己株式取得", "株式買戻し", "自社株買い"],
        "split": ["株式分割", "株式併合"],
        "merger": ["合併", "買収", "M&A", "統合"],
        
        # Guidance & Forecasts
        "guidance": ["業績予想", "業績見通し", "計画", "予測", "見込み", "修正"],
        "forecast": ["予想", "見通し", "計画", "予測", "見込み"],
        "revision": ["修正", "変更", "見直し", "上方修正", "下方修正"],
        
        # Market & Competition
        "market_share": ["市場シェア", "シェア", "市場占有率"],
        "competition": ["競合", "競争", "ライバル"],
        "expansion": ["拡大", "展開", "進出", "成長"],
        
        # Financial Health
        "debt": ["負債", "借入", "有利子負債", "借入金"],
        "cash": ["現金", "キャッシュ", "現金同等物"],
        "assets": ["資産", "総資産", "流動資産", "固定資産"],
        "equity": ["純資産", "株主資本", "自己資本"],
        
        # Time indicators
        "quarterly": ["四半期", "Q1", "Q2", "Q3", "Q4", "第1四半期", "第2四半期", "第3四半期", "第4四半期"],
        "annual": ["年間", "通期", "年度", "期", "年次"],
        "monthly": ["月次", "月間", "毎月"],
    }
    
    # Classification patterns
    CLASSIFICATION_PATTERNS = {
        "earnings": ["決算", "業績", "四半期", "通期", "売上", "利益"],
        "dividends": ["配当", "配当金", "増配", "減配", "配当性向"],
        "corporate_actions": ["合併", "買収", "M&A", "分割", "自己株式"],
        "guidance": ["予想", "見通し", "修正", "計画"],
        "management": ["経営", "役員", "社長", "CEO", "取締役"],
        "financial_health": ["負債", "資産", "キャッシュ", "財務"],
    }
    
    # Temporal patterns
    TEMPORAL_PATTERNS = {
        "recent": ["最近", "直近", "最新", "今回"],
        "past": ["前年", "前期", "昨年", "去年", "過去"],
        "future": ["来年", "次期", "将来", "今後", "予定"],
        "comparison": ["比較", "対比", "前年同期", "同期"],
    }
    
    @classmethod
    def expand_query(cls, query: str) -> str:
        """Expand query with financial synonyms"""
        expanded_terms = set()
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.add(query)
        
        # Find and expand financial terms
        for eng_term, jp_terms in cls.FINANCIAL_TERMS.items():
            if eng_term in query_lower:
                expanded_terms.update(jp_terms)
            for jp_term in jp_terms:
                if jp_term in query:
                    expanded_terms.add(eng_term)
                    expanded_terms.update(jp_terms)
        
        return " ".join(expanded_terms)
    
    @classmethod
    def classify_query(cls, query: str) -> QueryClassification:
        """Classify and analyze query"""
        query_lower = query.lower()
        
        # Detect query type
        query_type = "simple"
        if any(term in query_lower for term in ["compare", "vs", "versus", "比較", "対比"]):
            query_type = "comparative"
        elif any(term in query_lower for term in ["trend", "change", "increase", "decrease", "変化", "推移", "増加", "減少", "recent", "last", "latest", "quarter", "最近", "直近"]):
            query_type = "temporal"
        elif any(term in query_lower for term in ["analysis", "performance", "ratio", "分析", "業績", "指標"]):
            query_type = "analytical"
        
        # Extract financial terms (search both English and Japanese)
        financial_terms = []
        for category, terms in cls.FINANCIAL_TERMS.items():
            # Check if the English category name is in query
            if category.replace("_", " ") in query_lower:
                financial_terms.extend(terms)
            # Check if any Japanese terms are in query
            for term in terms:
                if term in query or term.lower() in query_lower:
                    financial_terms.append(term)
        
        # Extract time indicators
        time_indicators = []
        for category, terms in cls.TEMPORAL_PATTERNS.items():
            for term in terms:
                if term in query:
                    time_indicators.append(term)
        
        # Simple company extraction (basic patterns)
        companies = []
        # Look for company codes (4-digit numbers)
        company_codes = re.findall(r'\b\d{4}\b', query)
        companies.extend(company_codes)
        
        # Extract metrics mentioned
        metrics_mentioned = []
        metric_patterns = ["利益率", "売上高", "ROE", "ROA", "営業利益率", "配当利回り"]
        for metric in metric_patterns:
            if metric in query:
                metrics_mentioned.append(metric)
        
        # Generate expanded query
        expanded_query = cls.expand_query(query)
        
        return QueryClassification(
            query_type=query_type,
            financial_terms=financial_terms,
            companies=companies,
            time_indicators=time_indicators,
            metrics_mentioned=metrics_mentioned,
            expanded_query=expanded_query
        )

# ------------------------------------------------------------------
# 4) ENHANCED RETRIEVAL SYSTEM
# ------------------------------------------------------------------
class EnhancedFinancialRetrievalSystem:
    """Smart financial retrieval with filtering and query enhancement"""

    def __init__(self, cfg: EnhancedRetrievalConfig):
        self.cfg = cfg
        self.log = logging.getLogger(__name__)
        self.pg: Optional[asyncpg.Pool] = None
        self.rd: Optional[aioredis.Redis] = None
        self.dense = SentenceTransformer(cfg.dense_model)
        self.kb = FinancialKnowledgeBase()

    # ----------
    # bootstrap
    async def init(self):
        # Use single connection instead of pool to avoid auth issues
        self.pg = await asyncpg.connect(self.cfg.pg_dsn)
        try:
            if self.cfg.redis_url:
                self.rd = aioredis.from_url(self.cfg.redis_url)
                await self.rd.ping()  # Test connection
            else:
                self.rd = None
        except Exception as e:
            self.log.warning(f"Redis unavailable: {e}")
            self.rd = None
        await self._ensure_indexes()

    async def close(self):
        if self.pg:
            await self.pg.close()
        if self.rd:
            await self.rd.close()

    async def _ensure_indexes(self):
        """Ensure all necessary indexes exist"""
        c = self.pg
        # Vector index for embeddings
        await c.execute(f"""
            DO $$
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
            END$$;
        """)
        
        # Additional indexes for filtering (create them one by one to avoid transaction issues)
        indexes_to_create = [
            "CREATE INDEX IF NOT EXISTS disclosures_company_code_idx ON disclosures (company_code);",
            "CREATE INDEX IF NOT EXISTS disclosures_disclosure_date_idx ON disclosures (disclosure_date DESC);",
            "CREATE INDEX IF NOT EXISTS disclosures_category_idx ON disclosures (category);",
            "CREATE INDEX IF NOT EXISTS disclosures_subcategory_idx ON disclosures (subcategory);",
            "CREATE INDEX IF NOT EXISTS disclosures_company_date_idx ON disclosures (company_code, disclosure_date DESC);"
        ]
        
        for index_query in indexes_to_create:
            try:
                await c.execute(index_query)
            except Exception as e:
                # Ignore errors if index already exists or column doesn't exist
                pass

    # ----------
    # embeddings
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate native 1024-dimensional embedding (optimized for new schema)"""
        emb = self.dense.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        
        # Native 1024 dimensions - no padding needed!
        # Database now uses embedding_1024 column with proper 1024-dim vectors
        return emb

    # ----------
    # smart filtering
    def _build_smart_filters(self, query_class: QueryClassification, 
                           user_filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build smart WHERE clause based on query analysis and user filters"""
        conditions = ["embedding IS NOT NULL"]
        params = []
        param_idx = 2  # Start at 2 since $1 is reserved for embedding
        
        # User-provided filters take priority
        if company_codes := user_filters.get("company_codes"):
            conditions.append(f"company_code = ANY(${param_idx})")
            params.append(company_codes)
            param_idx += 1
        elif query_class.companies:  # Auto-detected companies
            conditions.append(f"company_code = ANY(${param_idx})")
            params.append(query_class.companies)
            param_idx += 1
        
        # Date range filtering
        if date_range := user_filters.get("date_range"):
            conditions.append(f"disclosure_date BETWEEN ${param_idx} AND ${param_idx + 1}")
            params.extend(date_range)
            param_idx += 2
        elif query_class.time_indicators:
            # Auto-detect temporal filters
            if any(term in query_class.time_indicators for term in ["最近", "直近", "最新"]):
                recent_date = date.today() - timedelta(days=90)  # Last 3 months
                conditions.append(f"disclosure_date >= ${param_idx}")
                params.append(recent_date)
                param_idx += 1
        
        # Classification-based filtering (support both naming conventions)
        if user_filters.get("classifications") or user_filters.get("category"):
            # Use the filter provided by user
            category_filter = user_filters.get("classifications") or user_filters.get("category")
            conditions.append(f"category = ANY(${param_idx})")
            params.append(category_filter)
            param_idx += 1
        # NOTE: Disabled automatic category filtering as it prevents results
        # The database may have different category names than our classification patterns
        # else:
        #     # Auto-detect classification from query
        #     auto_classifications = []
        #     for classification, patterns in self.kb.CLASSIFICATION_PATTERNS.items():
        #         if any(pattern in query_class.expanded_query for pattern in patterns):
        #             auto_classifications.append(classification)
        #     
        #     if auto_classifications:
        #         conditions.append(f"category = ANY(${param_idx})")
        #         params.append(auto_classifications)
        #         param_idx += 1
        
        # Sub-classification filtering (support both naming conventions)
        if subcategory := user_filters.get("subcategory"):
            conditions.append(f"subcategory = ANY(${param_idx})")
            params.append(subcategory)
            param_idx += 1
        
        return " AND ".join(conditions), params

    # ----------
    # enhanced retrieval stages
    async def _stage1_filter_retrieve(self, query: str, query_class: QueryClassification,
                                    filters: Dict[str, Any], k: int) -> List[RetrievalResult]:
        """Stage 1: Fast filtering + dense retrieval"""
        
        # Use expanded query for better coverage
        search_text = query_class.expanded_query if query_class.expanded_query else query
        emb = self._generate_embedding(search_text).tolist()
        
        # Build smart filtering conditions
        where_clause, params = self._build_smart_filters(query_class, filters)
        
        # Convert embedding for PostgreSQL
        emb_str = '[' + ','.join(map(str, emb)) + ']'
        
        sql = f"""
        SELECT id, company_code, company_name, title, disclosure_date, pdf_path,
               category, subcategory,
               1 - (embedding <=> $1::vector) AS score
        FROM disclosures 
        WHERE {where_clause}
          AND (1 - (embedding <=> $1::vector)) > {self.cfg.dense_threshold}
        ORDER BY embedding <=> $1::vector 
        LIMIT {min(k, self.cfg.rerank_candidates)};
        """
        
        # Removed debug logging - system is now optimized
        
        rows = await self.pg.fetch(sql, emb_str, *params)
        
        self.log.info(f"SQL query returned {len(rows)} rows")
        if len(rows) == 0:
            # Debug: let's see if we have any embeddings at all
            debug_count = await self.pg.fetchval("SELECT COUNT(*) FROM disclosures WHERE embedding IS NOT NULL")
            self.log.info(f"Total documents with embeddings in DB: {debug_count}")
            
            # Debug: check what the similarity scores look like
            sample_scores = await self.pg.fetch("""
                SELECT 1 - (embedding <=> $1::vector) AS score
                FROM disclosures 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector 
                LIMIT 5
            """, emb_str)
            scores = [row['score'] for row in sample_scores]
            self.log.info(f"Sample similarity scores: {scores}")
            self.log.info(f"Current threshold: {self.cfg.dense_threshold}")
        
        return [
            RetrievalResult(
                id=r[0], code=r[1], name=r[2], title=r[3],
                date=r[4], pdf=r[5], classification_l1=r[6], 
                classification_l2=r[7], score=r[8], ctx=""
            ) for r in rows
        ]

    async def _stage2_rerank(self, candidates: List[RetrievalResult], 
                           query_class: QueryClassification) -> List[RetrievalResult]:
        """Stage 2: Context-aware reranking"""
        if not candidates:
            return candidates
        
        # Apply business logic reranking
        scored_results = []
        
        for result in candidates:
            boost_score = 0.0
            
            # Boost recent documents for temporal queries
            if query_class.query_type == "temporal":
                days_old = (date.today() - result.date).days
                if days_old < 30:
                    boost_score += 0.1
                elif days_old < 90:
                    boost_score += 0.05
            
            # Boost exact classification matches
            if query_class.financial_terms:
                for term in query_class.financial_terms:
                    if term in result.title:
                        boost_score += 0.05
            
            # Boost specific document types for specific queries
            if query_class.query_type == "comparative" and result.classification_l1 == "earnings":
                boost_score += 0.1
            
            # Apply diversity penalty for same company (avoid too many docs from one company)
            company_count = sum(1 for r in scored_results if r[1].code == result.code)
            if company_count > 2:
                boost_score -= 0.05 * (company_count - 2)
            
            final_score = result.score + boost_score
            scored_results.append((final_score, result))
        
        # Sort by boosted score and return
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored_results]

    async def _stage3_context_enrichment(self, results: List[RetrievalResult], 
                                       k: int) -> List[RetrievalResult]:
        """Stage 3: Add context snippets to top results"""
        if not results:
            return results
        
        # Only enrich top k results to save processing time
        top_results = results[:k]
        
        for result in top_results:
            try:
                # Try to read PDF context (basic implementation)
                if result.pdf and Path(result.pdf).exists():
                    text = Path(result.pdf).read_text(encoding="utf-8", errors="ignore")[:2000]
                    # Extract a meaningful snippet
                    lines = text.split('\n')
                    meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]
                    result.ctx = ' '.join(meaningful_lines[:3])  # First 3 meaningful lines
            except Exception as e:
                self.log.debug(f"Could not extract context for {result.id}: {e}")
                result.ctx = ""
        
        return top_results

    # ----------
    # main search interface
    async def search(self, query: str, *, filters=None, k=20, strategy=None) -> List[RetrievalResult]:
        """Enhanced multi-stage retrieval"""
        filters = filters or {}
        
        # Check cache first
        cache_key = None
        if self.rd:
            cache_key = hashlib.md5(f"{query}|{json.dumps(filters,default=str)}|{k}".encode()).hexdigest()
            try:
                if cached := await self.rd.get(f"search:{cache_key}"):
                    cached_data = json.loads(cached)
                    return [RetrievalResult.parse_obj(item) for item in cached_data]
            except Exception as e:
                self.log.debug(f"Cache read error: {e}")
        
        # Step 1: Query analysis and classification
        query_class = self.kb.classify_query(query)
        self.log.info(f"Query classified as: {query_class.query_type}, terms: {query_class.financial_terms}")
        
        # Step 2: Stage 1 - Smart filtered retrieval
        candidates = await self._stage1_filter_retrieve(query, query_class, filters, self.cfg.rerank_candidates)
        
        # Step 3: Stage 2 - Context-aware reranking
        reranked = await self._stage2_rerank(candidates, query_class)
        
        # Step 4: Stage 3 - Context enrichment for final results
        final_results = await self._stage3_context_enrichment(reranked, k)
        
        # Cache results if Redis available
        if self.rd and cache_key:
            try:
                cache_data = [r.dict() for r in final_results]
                await self.rd.setex(f"search:{cache_key}", self.cfg.cache_seconds, json.dumps(cache_data))
            except Exception as e:
                self.log.debug(f"Cache write error: {e}")
        
        return final_results

    # ----------
    # utility methods
    async def get_company_stats(self, company_code: str, days: int = 365) -> Dict[str, Any]:
        """Get statistics for a specific company"""
        since_date = date.today() - timedelta(days=days)
        
        async with self.pg.acquire() as c:
            stats = await c.fetchrow("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT classification_l1) as document_types,
                    MAX(disclosure_date) as latest_disclosure,
                    MIN(disclosure_date) as earliest_disclosure
                FROM disclosures 
                WHERE company_code = $1 AND disclosure_date >= $2
            """, company_code, since_date)
            
            # Get classification breakdown
            classifications = await c.fetch("""
                SELECT classification_l1, COUNT(*) as count
                FROM disclosures 
                WHERE company_code = $1 AND disclosure_date >= $2
                GROUP BY classification_l1
                ORDER BY count DESC
            """, company_code, since_date)
        
        return {
            "company_code": company_code,
            "period_days": days,
            "total_documents": stats["total_documents"],
            "document_types": stats["document_types"],
            "latest_disclosure": stats["latest_disclosure"],
            "earliest_disclosure": stats["earliest_disclosure"],
            "classification_breakdown": {row["classification_l1"]: row["count"] for row in classifications}
        }

# ------------------------------------------------------------------
# 5) COMPATIBILITY WRAPPER
# ------------------------------------------------------------------
# Keep the same interface as the original system
RetrievalConfig = EnhancedRetrievalConfig
FinancialRetrievalSystem = EnhancedFinancialRetrievalSystem