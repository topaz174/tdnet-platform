#!/usr/bin/env python3
"""
Enhanced Financial Intelligence Agent - psycopg2 version
Uses synchronous database connections but keeps all enhanced features
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import psycopg2
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import logging

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from enhanced_retrieval_system import FinancialKnowledgeBase, QueryClassification, RetrievalResult

# ------------------------------------------------------------------
# ENHANCED RETRIEVAL WITH PSYCOPG2
# ------------------------------------------------------------------
class EnhancedFinancialRetrieval:
    """Enhanced retrieval using psycopg2 for compatibility"""
    
    def __init__(self, pg_dsn: str):
        self.pg_dsn = pg_dsn
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.kb = FinancialKnowledgeBase()
        self.log = logging.getLogger(__name__)
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate 1536-dim embedding compatible with database"""
        emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        
        # Pad to 1536 dimensions
        if len(emb) < 1536:
            padded = np.zeros(1536)
            padded[:len(emb)] = emb
            norm = np.linalg.norm(padded)
            if norm > 0:
                padded = padded / norm
            return padded
        return emb
    
    def _build_smart_filters(self, query_class: QueryClassification, user_filters: Dict[str, Any]) -> tuple:
        """Build smart WHERE clause and parameters"""
        conditions = ["embedding IS NOT NULL"]
        params = []
        
        # Company filtering
        if company_codes := user_filters.get("company_codes"):
            conditions.append("company_code = ANY(%s)")
            params.append(company_codes)
        elif query_class.companies:
            conditions.append("company_code = ANY(%s)")
            params.append(query_class.companies)
        
        # Date range filtering
        if date_range := user_filters.get("date_range"):
            conditions.append("disclosure_date BETWEEN %s AND %s")
            params.extend(date_range)
        elif query_class.time_indicators:
            if any(term in query_class.time_indicators for term in ["最近", "直近", "recent", "last"]):
                recent_date = date.today() - timedelta(days=90)
                conditions.append("disclosure_date >= %s")
                params.append(recent_date)
        
        # Category filtering
        if category := user_filters.get("category"):
            conditions.append("category = ANY(%s)")
            params.append(category)
        # Skip auto-classification for now to avoid SQL issues
        # else:
        #     # Auto-detect classification
        #     auto_classifications = []
        #     for classification, patterns in self.kb.CLASSIFICATION_PATTERNS.items():
        #         if any(pattern in query_class.expanded_query for pattern in patterns):
        #             auto_classifications.append(classification)
        #     
        #     if auto_classifications:
        #         conditions.append("category = ANY(%s)")
        #         params.append(auto_classifications)
        
        return " AND ".join(conditions), params
    
    def search(self, query: str, filters: Dict[str, Any] = None, k: int = 20) -> List[RetrievalResult]:
        """Enhanced search with all smart features"""
        filters = filters or {}
        
        # Step 1: Query classification
        query_class = self.kb.classify_query(query)
        self.log.info(f"Query classified as: {query_class.query_type}, terms: {query_class.financial_terms}")
        
        # Step 2: Generate embedding for expanded query
        search_text = query_class.expanded_query if query_class.expanded_query else query
        emb = self._generate_embedding(search_text)
        emb_str = '[' + ','.join(map(str, emb)) + ']'
        
        # Step 3: Build smart filters
        where_clause, where_params = self._build_smart_filters(query_class, filters)
        
        # Step 4: Execute search
        conn = psycopg2.connect(self.pg_dsn)
        cur = conn.cursor()
        
        # Count how many embedding parameters we need
        emb_count = where_clause.count('%s')
        
        sql = f"""
            SELECT id, company_code, company_name, title, disclosure_date, pdf_path,
                   category, subcategory,
                   1 - (embedding <=> %s::vector) AS score
            FROM disclosures 
            WHERE {where_clause}
              AND (1 - (embedding <=> %s::vector)) > 0.01
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
        """
        
        # Build parameters in correct order: where_params, then embeddings, then limit
        all_params = where_params + [emb_str, emb_str, emb_str, k]
        
        cur.execute(sql, all_params)
        rows = cur.fetchall()
        
        self.log.info(f"Found {len(rows)} documents")
        
        # Convert to RetrievalResult objects
        results = []
        for row in rows:
            result = RetrievalResult(
                id=row[0], code=row[1], name=row[2], title=row[3],
                date=row[4], pdf=row[5], classification_l1=row[6],
                classification_l2=row[7], score=row[8], ctx=""
            )
            results.append(result)
        
        conn.close()
        
        # Step 5: Apply business logic reranking
        return self._rerank_results(results, query_class)
    
    def _rerank_results(self, results: List[RetrievalResult], query_class: QueryClassification) -> List[RetrievalResult]:
        """Apply business logic reranking"""
        scored_results = []
        
        for result in results:
            boost_score = 0.0
            
            # Boost recent documents for temporal queries
            if query_class.query_type == "temporal":
                days_old = (date.today() - result.date).days
                if days_old < 30:
                    boost_score += 0.1
                elif days_old < 90:
                    boost_score += 0.05
            
            # Boost exact term matches
            if query_class.financial_terms:
                for term in query_class.financial_terms:
                    if term in result.title:
                        boost_score += 0.05
            
            # Apply diversity penalty for same company
            company_count = sum(1 for r in scored_results if r[1].code == result.code)
            if company_count > 2:
                boost_score -= 0.05 * (company_count - 2)
            
            final_score = result.score + boost_score
            scored_results.append((final_score, result))
        
        # Sort by boosted score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored_results]

# ------------------------------------------------------------------
# ENHANCED AGENT
# ------------------------------------------------------------------
class EnhancedFinancialAgent:
    """Enhanced financial agent with smart retrieval"""
    
    def __init__(self, pg_dsn: str, llm: ChatOpenAI):
        self.retrieval = EnhancedFinancialRetrieval(pg_dsn)
        self.llm = llm
        self.log = logging.getLogger(__name__)
    
    def run(self, query: str) -> str:
        """Process query with enhanced multi-stage analysis"""
        try:
            # Stage 1: Analyze query
            query_class = self.retrieval.kb.classify_query(query)
            
            # Stage 2: Search with enhancement
            results = self.retrieval.search(query, k=15)
            
            if not results:
                return "I couldn't find any relevant financial documents matching your query. Please try rephrasing or check if the company codes/terms are correct."
            
            # Stage 3: Analyze results
            analysis = self._analyze_results(results, query_class)
            
            # Stage 4: Compose comprehensive answer
            return self._compose_answer(query, results, analysis, query_class)
            
        except Exception as e:
            self.log.error(f"Error in agent workflow: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question."
    
    def _analyze_results(self, results: List[RetrievalResult], query_class: QueryClassification) -> Dict[str, Any]:
        """Analyze search results for insights"""
        analysis = {
            "total_documents": len(results),
            "companies": list(set(r.code for r in results)),
            "company_count": len(set(r.code for r in results)),
            "date_range": {
                "earliest": min(r.date for r in results),
                "latest": max(r.date for r in results)
            },
            "categories": list(set(r.classification_l1 for r in results if r.classification_l1))
        }
        
        # Company frequency
        company_counts = {}
        for result in results:
            company_counts[result.name] = company_counts.get(result.name, 0) + 1
        
        analysis["top_companies"] = dict(sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return analysis
    
    def _compose_answer(self, query: str, results: List[RetrievalResult], analysis: Dict[str, Any], query_class: QueryClassification) -> str:
        """Compose comprehensive answer"""
        
        # Build document summaries
        top_docs = results[:8]
        doc_summaries = []
        
        for i, doc in enumerate(top_docs, 1):
            summary = f"{i}. **{doc.name}** ({doc.code}) - {doc.title}\n"
            summary += f"   📅 Date: {doc.date} | Score: {doc.score:.3f}"
            if doc.classification_l1:
                summary += f" | Type: {doc.classification_l1}"
            doc_summaries.append(summary)
        
        documents_text = "\n\n".join(doc_summaries)
        
        # Build analysis summary
        analysis_text = f"""
**Query Analysis:**
- Type: {query_class.query_type.title()} Query
- Companies Found: {analysis['company_count']} ({', '.join(analysis['companies'][:5])})
- Date Range: {analysis['date_range']['earliest']} to {analysis['date_range']['latest']}
- Document Types: {', '.join(analysis['categories'])}

**Key Findings:**
- Total Relevant Documents: {analysis['total_documents']}
- Most Active Companies: {', '.join([f"{k} ({v} docs)" for k, v in list(analysis['top_companies'].items())[:3]])}
"""
        
        # Compose final prompt
        prompt = f"""You are an expert financial analyst specializing in Japanese corporate disclosures.

**User Question:** "{query}"

{analysis_text}

**Top Relevant Documents:**
{documents_text}

**Instructions:**
1. Provide a comprehensive analysis addressing the user's specific question
2. Cite specific companies, dates, and document types from the evidence
3. If this is a comparative query, highlight differences between companies
4. If this is a temporal query, explain trends and timing patterns
5. Be specific about what the documents reveal and any limitations
6. Use bullet points for clarity and include relevant financial metrics when mentioned
7. Write in English but include Japanese company names when appropriate

**Answer:**"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        return response.content

# ------------------------------------------------------------------
# DEMO USAGE
# ------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    
    # Initialize
    pg_dsn = os.getenv("PG_DSN")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    agent = EnhancedFinancialAgent(pg_dsn, llm)
    
    # Test queries
    test_queries = [
        "Which companies announced dividend increases in the last quarter?",
        "Show me recent earnings guidance revisions for Toyota and Honda",
        "What companies have had significant M&A activity recently?",
        "Compare the recent performance announcements of major tech companies",
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        answer = agent.run(query)
        print(answer)

if __name__ == "__main__":
    main()