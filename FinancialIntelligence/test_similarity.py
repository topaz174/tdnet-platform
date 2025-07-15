#!/usr/bin/env python3
"""
Test similarity computation with actual database data
"""

import psycopg2
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def test_similarity():
    # Load the model
    print("Loading model...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # Generate test embedding
    test_query = "earnings"
    emb = model.encode([test_query], normalize_embeddings=True)[0]
    print(f"Generated embedding dimensions: {len(emb)}")
    
    # Pad to 1536 dimensions to match database
    if len(emb) < 1536:
        padded = np.zeros(1536)
        padded[:len(emb)] = emb
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded = padded / norm
        emb = padded
    
    print(f"Padded embedding dimensions: {len(emb)}")
    
    # Connect to database
    conn = psycopg2.connect(os.getenv('PG_DSN'))
    cur = conn.cursor()
    
    # Test similarity with a few documents
    emb_str = '[' + ','.join(map(str, emb)) + ']'
    
    print("\nTesting similarity calculation...")
    cur.execute("""
        SELECT id, company_name, title,
               1 - (embedding <=> %s::vector) AS score
        FROM disclosures 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector 
        LIMIT 5
    """, (emb_str, emb_str))
    
    results = cur.fetchall()
    print(f"Found {len(results)} results:")
    
    for row in results:
        print(f"  ID: {row[0]} | Score: {row[3]:.4f} | {row[1]} - {row[2][:50]}...")
    
    # Also test without similarity threshold
    cur.execute("""
        SELECT COUNT(*) as total_with_embeddings,
               MIN(1 - (embedding <=> %s::vector)) as min_score,
               MAX(1 - (embedding <=> %s::vector)) as max_score,
               AVG(1 - (embedding <=> %s::vector)) as avg_score
        FROM disclosures 
        WHERE embedding IS NOT NULL
    """, (emb_str, emb_str, emb_str))
    
    stats = cur.fetchone()
    print(f"\nSimilarity statistics:")
    print(f"  Total docs with embeddings: {stats[0]}")
    print(f"  Min similarity score: {stats[1]:.4f}")
    print(f"  Max similarity score: {stats[2]:.4f}")
    print(f"  Avg similarity score: {stats[3]:.4f}")
    
    conn.close()

if __name__ == "__main__":
    test_similarity()