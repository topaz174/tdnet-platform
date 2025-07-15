#!/usr/bin/env python3
"""
Quick Embedding Integrity Check
===============================

Fast check for obviously corrupted embeddings from segmentation faults.
Focuses on critical issues that would prevent proper functioning.
"""

import os
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

def main():
    print("🔍 QUICK EMBEDDING INTEGRITY CHECK")
    print("=" * 40)
    
    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        print("\n📊 Checking embedding statistics...")
        
        # Check basic counts
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_docs,
                COUNT(embedding) as with_embeddings,
                COUNT(*) - COUNT(embedding) as without_embeddings
            FROM disclosures
        """)).fetchone()
        
        total, with_emb, without_emb = result
        print(f"   Total documents: {total:,}")
        print(f"   With embeddings: {with_emb:,}")
        print(f"   Without embeddings: {without_emb:,}")
        print(f"   Coverage: {with_emb/total*100:.1f}%")
        
        # Check for obviously corrupted embeddings (common segfault patterns)
        print("\n🔍 Checking for corruption patterns...")
        
        # Check sample embeddings to verify they look normal
        sample_check = session.execute(text("""
            SELECT id, embedding
            FROM disclosures 
            WHERE embedding IS NOT NULL
            ORDER BY id
            LIMIT 10
        """)).fetchall()
        
        if sample_check:
            print(f"   ✅ Sample check (first 10 embeddings):")
            all_zero_count = 0
            dimension_issues = 0
            
            for doc_id, embedding in sample_check:
                try:
                    # Convert pgvector to numpy array
                    if isinstance(embedding, str):
                        # Parse string representation
                        import json
                        embedding_array = np.array(json.loads(embedding), dtype=np.float32)
                    else:
                        # Convert pgvector to list and then numpy
                        embedding_array = np.array(embedding, dtype=np.float32)
                    
                    # Check dimension
                    if len(embedding_array) != 1024:
                        dimension_issues += 1
                        print(f"      ID {doc_id}: ❌ Wrong dimension: {len(embedding_array)} != 1024")
                        continue
                    
                    # Check for all zeros
                    if np.all(np.abs(embedding_array) < 1e-10):
                        all_zero_count += 1
                        print(f"      ID {doc_id}: ⚠️  All-zero embedding")
                    else:
                        # Show first few values
                        sample_values = embedding_array[:3]
                        formatted_values = [f'{x:.3f}' for x in sample_values]
                        l2_norm = np.linalg.norm(embedding_array)
                        print(f"      ID {doc_id}: ✅ Normal values {formatted_values}... (norm: {l2_norm:.3f})")
                
                except Exception as e:
                    print(f"      ID {doc_id}: ❌ Parse error: {e}")
                    dimension_issues += 1
            
            issues_found = all_zero_count + dimension_issues
            
            if issues_found == 0:
                print(f"   🎉 All sample embeddings look healthy!")
            else:
                print(f"   ⚠️  Found issues in {issues_found}/10 samples:")
                if all_zero_count > 0:
                    print(f"      - {all_zero_count} all-zero embeddings")
                if dimension_issues > 0:
                    print(f"      - {dimension_issues} dimension/parsing issues")
        
        # Overall health assessment
        total_issues = 0  # We'll base this on sample results since we can't easily query vector dimensions
        
        print(f"\n🎯 HEALTH ASSESSMENT:")
        if sample_check:
            issues_found = all_zero_count + dimension_issues
            if issues_found == 0:
                print("   🎉 Sample embeddings look healthy!")
                print("   ✅ All tested embeddings have correct dimensions and non-zero values")
            else:
                print(f"   ⚠️  Found issues in {issues_found}/10 sample embeddings")
                estimated_issues = int((issues_found / 10) * with_emb)
                print(f"   📊 Estimated corruption: ~{estimated_issues:,} embeddings ({issues_found*10:.1f}%)")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if sample_check and (all_zero_count + dimension_issues) == 0:
            print("   ✅ Embeddings look healthy - can proceed with confidence")
            print("   🔄 Run 'python embed_disclosures_fixed.py' to process remaining documents")
        elif sample_check and (all_zero_count + dimension_issues) <= 2:
            print("   💡 Minor issues detected - mostly healthy")
            print("   🔧 Optionally run 'python validate_and_fix_embeddings.py' for full validation")
            print("   🔄 Can proceed with 'python embed_disclosures_fixed.py' for remaining documents")
        else:
            print("   🔧 Issues detected in sample - recommend full validation:")
            print("   1. Run 'python validate_and_fix_embeddings.py' to identify and fix all corrupted embeddings")
            print("   2. Then run 'python embed_disclosures_fixed.py' to re-process corrupted documents")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()