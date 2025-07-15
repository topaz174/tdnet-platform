#!/usr/bin/env python3
"""
Direct test of enhanced retrieval using psycopg2 (sync version)
"""

import psycopg2
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from enhanced_retrieval_system import FinancialKnowledgeBase

load_dotenv()

class DirectEnhancedTest:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.kb = FinancialKnowledgeBase()
        self.conn = psycopg2.connect(os.getenv('PG_DSN'))
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate 1536-dim embedding matching database format"""
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
    
    def search(self, query: str, k: int = 5):
        """Direct search with enhanced features"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Step 1: Query classification
        query_class = self.kb.classify_query(query)
        print(f"   Query type: {query_class.query_type}")
        print(f"   Financial terms: {query_class.financial_terms}")
        print(f"   Expanded query: {query_class.expanded_query[:100]}...")
        
        # Step 2: Generate embedding
        search_text = query_class.expanded_query if query_class.expanded_query else query
        emb = self.generate_embedding(search_text)
        emb_str = '[' + ','.join(map(str, emb)) + ']'
        
        # Step 3: Build smart filters
        where_conditions = ["embedding IS NOT NULL"]
        params = [emb_str]
        
        # Add temporal filtering for temporal queries
        if query_class.query_type == "temporal" and any(term in query_class.time_indicators for term in ["最近", "直近", "recent"]):
            where_conditions.append("disclosure_date >= CURRENT_DATE - INTERVAL '90 days'")
        
        where_clause = " AND ".join(where_conditions)
        
        # Step 4: Execute search
        cur = self.conn.cursor()
        sql = f"""
            SELECT id, company_code, company_name, title, disclosure_date,
                   category, subcategory,
                   1 - (embedding <=> %s::vector) AS score
            FROM disclosures 
            WHERE {where_clause}
              AND (1 - (embedding <=> %s::vector)) > 0.01
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
        """
        
        cur.execute(sql, [emb_str, emb_str, emb_str, k])
        results = cur.fetchall()
        
        print(f"   Found {len(results)} results:")
        for i, row in enumerate(results, 1):
            print(f"   {i}. {row[2]} ({row[1]}) - Score: {row[7]:.4f}")
            print(f"      {row[3][:80]}...")
            print(f"      Date: {row[4]} | Category: {row[5]}")
        
        return results
    
    def close(self):
        self.conn.close()

def main():
    print("🧪 Direct Enhanced Retrieval Test")
    
    tester = DirectEnhancedTest()
    
    test_queries = [
        "dividend increases",
        "配当増配",  # dividend increase in Japanese
        "earnings guidance",
        "recent announcements",
        "Toyota Honda earnings"
    ]
    
    for query in test_queries:
        tester.search(query, k=3)
    
    tester.close()
    print("\n✅ Direct test completed!")

if __name__ == "__main__":
    main()