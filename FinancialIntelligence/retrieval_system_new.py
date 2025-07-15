"""
Production-ready financial document retrieval system
Integrates Reason-ModernColBERT with PostgreSQL pgvector
"""

import asyncio
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass
from pathlib import Path
import hashlib

import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import fitz  # PyMuPDF
from pydantic import BaseModel
import redis.asyncio as redis

# Configuration
@dataclass
class RetrievalConfig:
    postgres_url: str
    redis_url: str
    colbert_model_name: str = "lightonai/Reason-ModernColBERT"
    dense_model_name: str = "intfloat/multilingual-e5-large"
    max_seq_length: int = 512
    batch_size: int = 16
    similarity_threshold: float = 0.7
    max_results: int = 20

class DocumentChunk(BaseModel):
    content: str
    page_number: int
    chunk_id: str
    metadata: Dict[str, Any]

class RetrievalResult(BaseModel):
    document_id: int
    company_code: str
    company_name: str
    title: str
    disclosure_date: date
    similarity_score: float
    reasoning_context: str
    relevant_chunks: List[DocumentChunk]
    pdf_path: str

class FinancialRetrievalSystem:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.dense_model = None
        self.colbert_model = None
        self.colbert_tokenizer = None
        
        # Database connections
        self.pg_pool = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize all components"""
        await self._init_database()
        await self._init_redis()
        await self._init_models()
        
    async def _init_database(self):
        self.pg_pool = await asyncpg.create_pool(
            self.config.postgres_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
    async def _init_redis(self):
        self.redis_client = redis.from_url(
            self.config.redis_url,
            decode_responses=False  # Keep binary for embeddings
        )
        
    async def _init_models(self):
        """Initialize embedding models"""
        self.logger.info("Loading embedding models...")
        
        # Dense model for fast filtering
        self.dense_model = SentenceTransformer(self.config.dense_model_name)
        
        # ColBERT model for reasoning-based retrieval
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(self.config.colbert_model_name)
        self.colbert_model = AutoModel.from_pretrained(self.config.colbert_model_name)
        self.colbert_model.eval()
        
        self.logger.info("Models loaded successfully")
    
    def _extract_pdf_content(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract structured content from PDF"""
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    chunk = DocumentChunk(
                        content=text,
                        page_number=page_num + 1,
                        chunk_id=f"{Path(pdf_path).stem}_p{page_num + 1}",
                        metadata={
                            "word_count": len(text.split()),
                            "char_count": len(text)
                        }
                    )
                    chunks.append(chunk)
            doc.close()
        except Exception as e:
            self.logger.error(f"Error extracting PDF {pdf_path}: {e}")
            
        return chunks
    
    def _extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """Extract financial entities and metrics from text"""
        import re
        
        entities = {
            "amounts": [],
            "percentages": [],
            "dates": [],
            "financial_terms": []
        }
        
        # Japanese currency patterns
        yen_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:円|億円|兆円)'
        amounts = re.findall(yen_pattern, text)
        entities["amounts"] = amounts
        
        # Percentage patterns
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(pct_pattern, text)
        entities["percentages"] = percentages
        
        # Financial terms (Japanese)
        financial_terms = [
            "売上", "営業利益", "純利益", "配当", "株主総会", "決算", "業績予想",
            "設備投資", "自己株式", "株式分割", "増配", "減配", "業績修正"
        ]
        
        found_terms = [term for term in financial_terms if term in text]
        entities["financial_terms"] = found_terms
        
        return entities
    
    def _generate_dense_embedding(self, text: str) -> np.ndarray:
        """Generate dense embedding for fast filtering"""
        return self.dense_model.encode([text], normalize_embeddings=True)[0]
    
    def _generate_colbert_embeddings(self, text: str, is_query: bool = False) -> List[np.ndarray]:
        """Generate ColBERT token-level embeddings"""
        inputs = self.colbert_tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.colbert_model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
            
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings.tolist()
    
    async def process_document(self, document_id: int, pdf_path: str, force_reprocess: bool = False):
        """Process a single document and store embeddings"""
        try:
            # Check if already processed
            if not force_reprocess:
                async with self.pg_pool.acquire() as conn:
                    existing = await conn.fetchrow(
                        "SELECT content_hash FROM disclosures WHERE id = $1 AND dense_embedding IS NOT NULL",
                        document_id
                    )
                    if existing:
                        return
            
            # Extract content
            chunks = self._extract_pdf_content(pdf_path)
            if not chunks:
                self.logger.warning(f"No content extracted from {pdf_path}")
                return
            
            # Combine all content
            full_text = "\n".join([chunk.content for chunk in chunks])
            content_hash = hashlib.md5(full_text.encode()).hexdigest()
            
            # Generate embeddings
            dense_embedding = self._generate_dense_embedding(full_text)
            colbert_doc_embeddings = self._generate_colbert_embeddings(full_text, is_query=False)
            
            # Extract financial entities
            financial_entities = self._extract_financial_entities(full_text)
            
            # Extract reasoning context (first paragraph + any numerical data)
            reasoning_context = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
            
            # Store in database
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE disclosures SET 
                        dense_embedding = $1,
                        colbert_doc_embeddings = $2,
                        reasoning_context = $3,
                        financial_entities = $4,
                        content_hash = $5,
                        processing_metadata = $6
                    WHERE id = $7
                """, 
                    dense_embedding.tolist(),
                    colbert_doc_embeddings,
                    reasoning_context,
                    json.dumps(financial_entities),
                    content_hash,
                    json.dumps({
                        "processed_at": datetime.now().isoformat(),
                        "model_version": "reason-moderncolbert-v1",
                        "chunk_count": len(chunks)
                    }),
                    document_id
                )
                
            self.logger.info(f"Processed document {document_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_id}: {e}")
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query to determine retrieval strategy"""
        reasoning_indicators = [
            "revised", "changed", "increased", "decreased", "following", "after",
            "due to", "because", "trailing", "rolling", "year-over-year",
            "compared to", "versus", "trend", "correlation", "impact", "caused",
            "修正", "変更", "増加", "減少", "影響", "原因", "傾向"
        ]
        
        complex_financial = [
            "margin", "ratio", "growth", "performance", "analysis", "comparison",
            "利益率", "成長率", "業績", "分析", "比較"
        ]
        
        if any(indicator in query.lower() for indicator in reasoning_indicators):
            return "reasoning"
        elif any(term in query.lower() for term in complex_financial):
            return "analytical" 
        else:
            return "simple"
    
    async def _dense_retrieval(
        self, 
        query: str, 
        filters: Dict[str, Any] = None,
        limit: int = 20
    ) -> List[RetrievalResult]:
        """Fast dense vector retrieval with structured filters"""
        
        # Generate query embedding
        query_embedding = self._generate_dense_embedding(query)
        
        # Build filter conditions
        where_conditions = ["dense_embedding IS NOT NULL"]
        params = [query_embedding.tolist()]
        param_idx = 2
        
        if filters:
            if filters.get("company_codes"):
                where_conditions.append(f"company_code = ANY(${param_idx})")
                params.append(filters["company_codes"])
                param_idx += 1
                
            if filters.get("classifications"):
                where_conditions.append(f"classification_l1 = ANY(${param_idx})")
                params.append(filters["classifications"])
                param_idx += 1
                
            if filters.get("date_range"):
                where_conditions.append(f"disclosure_date BETWEEN ${param_idx} AND ${param_idx + 1}")
                params.extend(filters["date_range"])
                param_idx += 2
        
        where_clause = " AND ".join(where_conditions)
        
        query_sql = f"""
            SELECT 
                id, company_code, company_name, title, disclosure_date,
                pdf_path, reasoning_context,
                1 - (dense_embedding <=> $1) as similarity_score
            FROM disclosures 
            WHERE {where_clause}
                AND 1 - (dense_embedding <=> $1) > {self.config.similarity_threshold}
            ORDER BY dense_embedding <=> $1
            LIMIT {limit}
        """
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
            
        return [
            RetrievalResult(
                document_id=row["id"],
                company_code=row["company_code"],
                company_name=row["company_name"],
                title=row["title"],
                disclosure_date=row["disclosure_date"],
                similarity_score=row["similarity_score"],
                reasoning_context=row["reasoning_context"] or "",
                relevant_chunks=[],
                pdf_path=row["pdf_path"]
            )
            for row in rows
        ]
    
    def _colbert_score(self, query_embeddings: List[np.ndarray], doc_embeddings: List[np.ndarray]) -> float:
        """Compute ColBERT similarity score"""
        if not query_embeddings or not doc_embeddings:
            return 0.0
            
        query_embeddings = np.array(query_embeddings)
        doc_embeddings = np.array(doc_embeddings)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        # ColBERT scoring: max similarity for each query token
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Average over query tokens
        return float(np.mean(max_similarities))
    
    async def _reasoning_retrieval(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        limit: int = 20
    ) -> List[RetrievalResult]:
        """ColBERT-based reasoning retrieval"""
        
        # Generate query embeddings
        query_embeddings = self._generate_colbert_embeddings(query, is_query=True)
        
        # First filter with basic conditions
        where_conditions = ["colbert_doc_embeddings IS NOT NULL"]
        params = []
        param_idx = 1
        
        if filters:
            if filters.get("company_codes"):
                where_conditions.append(f"company_code = ANY(${param_idx})")
                params.append(filters["company_codes"])
                param_idx += 1
                
            if filters.get("classifications"):
                where_conditions.append(f"classification_l1 = ANY(${param_idx})")
                params.append(filters["classifications"])
                param_idx += 1
                
            if filters.get("date_range"):
                where_conditions.append(f"disclosure_date BETWEEN ${param_idx} AND ${param_idx + 1}")
                params.extend(filters["date_range"])
                param_idx += 2
        
        where_clause = " AND ".join(where_conditions)
        
        # Fetch candidates
        query_sql = f"""
            SELECT 
                id, company_code, company_name, title, disclosure_date,
                pdf_path, reasoning_context, colbert_doc_embeddings
            FROM disclosures 
            WHERE {where_clause}
            ORDER BY disclosure_date DESC
            LIMIT 200  -- Get more candidates for reranking
        """
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
        
        # Compute ColBERT scores
        results = []
        for row in rows:
            doc_embeddings = row["colbert_doc_embeddings"]
            if doc_embeddings:
                score = self._colbert_score(query_embeddings, doc_embeddings)
                
                if score > self.config.similarity_threshold:
                    results.append(RetrievalResult(
                        document_id=row["id"],
                        company_code=row["company_code"],
                        company_name=row["company_name"],
                        title=row["title"],
                        disclosure_date=row["disclosure_date"],
                        similarity_score=score,
                        reasoning_context=row["reasoning_context"] or "",
                        relevant_chunks=[],
                        pdf_path=row["pdf_path"]
                    ))
        
        # Sort by ColBERT score and return top results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]
    
    async def search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        force_strategy: str = None
    ) -> List[RetrievalResult]:
        """Main search interface with automatic strategy selection"""
        
        # Check cache first
        cache_key = f"search:{hashlib.md5(f'{query}:{filters}:{limit}'.encode()).hexdigest()}"
        
        try:
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                return [RetrievalResult.parse_raw(r) for r in json.loads(cached_result)]
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        # Determine strategy
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self._classify_query_type(query)
        
        # Execute retrieval
        if strategy == "reasoning":
            results = await self._reasoning_retrieval(query, filters, limit)
        else:
            results = await self._dense_retrieval(query, filters, limit)
        
        # Cache results
        try:
            cache_data = json.dumps([r.json() for r in results])
            await self.redis_client.setex(cache_key, 3600, cache_data)  # 1 hour cache
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
        
        return results
    
    async def batch_process_documents(self, document_ids: List[int], batch_size: int = 10):
        """Process multiple documents in batches"""
        
        # Get document paths
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, pdf_path FROM disclosures WHERE id = ANY($1)",
                document_ids
            )
        
        doc_map = {row["id"]: row["pdf_path"] for row in rows}
        
        # Process in batches
        for i in range(0, len(document_ids), batch_size):
            batch = document_ids[i:i + batch_size]
            tasks = [
                self.process_document(doc_id, doc_map[doc_id]) 
                for doc_id in batch if doc_id in doc_map
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info(f"Processed batch {i//batch_size + 1}")
    
    async def close(self):
        """Clean up resources"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()

# Usage example
async def main():
    config = RetrievalConfig(
        postgres_url="postgresql://user:pass@localhost/financial_db",
        redis_url="redis://localhost:6379/0"
    )
    
    system = FinancialRetrievalSystem(config)
    await system.initialize()
    
    # Process all unprocessed documents
    async with system.pg_pool.acquire() as conn:
        unprocessed = await conn.fetch(
            "SELECT id FROM disclosures WHERE dense_embedding IS NULL LIMIT 100"
        )
    
    if unprocessed:
        doc_ids = [row["id"] for row in unprocessed]
        await system.batch_process_documents(doc_ids)
    
    # Example searches
    queries = [
        "Companies that announced dividend increases in the last quarter",
        "Firms that revised earnings guidance upward",
        "Management changes following poor performance",
        "Share buyback announcements with strategic rationale"
    ]
    
    for query in queries:
        results = await system.search(query, limit=5)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"  - {result.company_name}: {result.title} (Score: {result.similarity_score:.3f})")
    
    await system.close()

if __name__ == "__main__":
    asyncio.run(main())
