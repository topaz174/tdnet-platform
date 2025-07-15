#!/usr/bin/env python3
"""
Embedding Validation and Repair Script
=====================================

This script validates existing embeddings and identifies/fixes corrupted ones
that may have resulted from previous segmentation faults.

Validation checks:
1. Embedding dimension (should be 1024)
2. Embedding type (should be float array)
3. Non-zero embeddings (all-zero indicates processing failure)
4. Valid float ranges (no NaN, Inf values)
5. Proper normalization (L2 norm should be ~1.0)

Environment variables:
  PG_DSN           postgresql+psycopg2://user:pass@host:5432/dbname
"""

import os, sys, json
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Expected embedding configuration
EXPECTED_DIMENSION = 1024
BATCH_SIZE = 1000
TOLERANCE_L2_NORM = 0.1  # Allow 10% deviation from unit norm

def validate_embedding(embedding_data) -> dict:
    """Validate a single embedding and return validation results."""
    validation = {
        'is_valid': True,
        'issues': [],
        'dimension': None,
        'l2_norm': None,
        'has_nan': False,
        'has_inf': False,
        'is_zero': False
    }
    
    try:
        # Convert to numpy array
        if isinstance(embedding_data, str):
            # Handle string representation
            embedding = np.array(json.loads(embedding_data), dtype=np.float32)
        elif isinstance(embedding_data, list):
            embedding = np.array(embedding_data, dtype=np.float32)
        else:
            embedding = np.array(embedding_data, dtype=np.float32)
        
        # Check dimension
        validation['dimension'] = len(embedding)
        if len(embedding) != EXPECTED_DIMENSION:
            validation['is_valid'] = False
            validation['issues'].append(f'Wrong dimension: {len(embedding)} != {EXPECTED_DIMENSION}')
        
        # Check for NaN values
        if np.isnan(embedding).any():
            validation['is_valid'] = False
            validation['has_nan'] = True
            validation['issues'].append('Contains NaN values')
        
        # Check for infinite values
        if np.isinf(embedding).any():
            validation['is_valid'] = False
            validation['has_inf'] = True
            validation['issues'].append('Contains infinite values')
        
        # Check if all zeros (indicates failed processing)
        if np.all(embedding == 0):
            validation['is_valid'] = False
            validation['is_zero'] = True
            validation['issues'].append('All-zero embedding (processing failure)')
        
        # Check L2 norm (should be close to 1.0 for normalized embeddings)
        if not validation['is_zero']:
            l2_norm = np.linalg.norm(embedding)
            validation['l2_norm'] = float(l2_norm)
            
            # Allow some tolerance for normalization
            if abs(l2_norm - 1.0) > TOLERANCE_L2_NORM:
                validation['issues'].append(f'Unusual L2 norm: {l2_norm:.3f} (expected ~1.0)')
        
    except Exception as e:
        validation['is_valid'] = False
        validation['issues'].append(f'Parsing error: {str(e)}')
    
    return validation

def main():
    print("🔍 EMBEDDING VALIDATION AND REPAIR")
    print("=" * 50)
    
    # Database connection
    engine = create_engine(os.environ["PG_DSN"])
    Session = sessionmaker(bind=engine)
    
    # Statistics
    total_embeddings = 0
    valid_embeddings = 0
    corrupted_embeddings = 0
    corrupted_ids = []
    issue_summary = {}
    
    print("\n📊 Analyzing existing embeddings...")
    
    with Session() as session:
        # Get total count of embeddings
        total_result = session.execute(text("""
            SELECT COUNT(*) as total
            FROM disclosures 
            WHERE embedding IS NOT NULL
        """)).fetchone()
        
        total_embeddings = total_result[0]
        print(f"   Total embeddings to validate: {total_embeddings}")
        
        if total_embeddings == 0:
            print("   ✅ No embeddings found to validate")
            return
        
        # Process in batches
        offset = 0
        
        with tqdm(total=total_embeddings, desc="Validating embeddings") as pbar:
            while offset < total_embeddings:
                # Fetch batch
                rows = session.execute(text("""
                    SELECT id, embedding
                    FROM disclosures 
                    WHERE embedding IS NOT NULL
                    ORDER BY id
                    LIMIT :limit OFFSET :offset
                """), {"limit": BATCH_SIZE, "offset": offset}).fetchall()
                
                if not rows:
                    break
                
                # Validate each embedding in the batch
                for doc_id, embedding_data in rows:
                    validation = validate_embedding(embedding_data)
                    
                    if validation['is_valid']:
                        valid_embeddings += 1
                    else:
                        corrupted_embeddings += 1
                        corrupted_ids.append(doc_id)
                        
                        # Track issue types
                        for issue in validation['issues']:
                            issue_type = issue.split(':')[0]  # Get issue category
                            issue_summary[issue_type] = issue_summary.get(issue_type, 0) + 1
                    
                    pbar.update(1)
                
                offset += len(rows)
    
    print(f"\n📈 VALIDATION RESULTS:")
    print(f"   Total embeddings: {total_embeddings}")
    print(f"   ✅ Valid embeddings: {valid_embeddings}")
    print(f"   ❌ Corrupted embeddings: {corrupted_embeddings}")
    
    if corrupted_embeddings > 0:
        print(f"   📊 Corruption rate: {corrupted_embeddings/total_embeddings*100:.1f}%")
        
        print(f"\n🔍 ISSUE BREAKDOWN:")
        for issue_type, count in sorted(issue_summary.items()):
            print(f"   {issue_type}: {count} embeddings")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if corrupted_embeddings < 100:
            print(f"   Sample corrupted IDs: {corrupted_ids[:10]}")
            if len(corrupted_ids) > 10:
                print(f"   ... and {len(corrupted_ids) - 10} more")
        
        # Offer to fix corrupted embeddings
        print(f"\n🔧 REPAIR OPTIONS:")
        print(f"   1. Reset corrupted embeddings to NULL (will be re-processed)")
        print(f"   2. Export corrupted IDs to file for manual review")
        print(f"   3. Continue without changes")
        
        choice = input("\nChoose an option (1/2/3): ").strip()
        
        if choice == "1":
            print(f"\n🔧 Resetting {corrupted_embeddings} corrupted embeddings to NULL...")
            
            with Session() as session:
                # Reset corrupted embeddings in batches
                batch_size = 1000
                for i in range(0, len(corrupted_ids), batch_size):
                    batch_ids = corrupted_ids[i:i+batch_size]
                    placeholders = ','.join([f':id{j}' for j in range(len(batch_ids))])
                    params = {f'id{j}': batch_ids[j] for j in range(len(batch_ids))}
                    
                    session.execute(text(f"""
                        UPDATE disclosures 
                        SET embedding = NULL 
                        WHERE id IN ({placeholders})
                    """), params)
                
                session.commit()
            
            print(f"   ✅ Reset {corrupted_embeddings} embeddings to NULL")
            print(f"   💡 Run embed_disclosures_fixed.py to re-process these documents")
        
        elif choice == "2":
            # Export corrupted IDs
            output_file = "corrupted_embedding_ids.txt"
            with open(output_file, 'w') as f:
                f.write("# Corrupted Embedding IDs\n")
                f.write(f"# Total: {len(corrupted_ids)} documents\n")
                f.write("# Generated by validate_and_fix_embeddings.py\n\n")
                for doc_id in corrupted_ids:
                    f.write(f"{doc_id}\n")
            
            print(f"   ✅ Exported {len(corrupted_ids)} corrupted IDs to {output_file}")
        
        else:
            print("   ℹ️  No changes made")
    
    else:
        print("   🎉 All embeddings are valid!")
    
    # Final database state
    with Session() as session:
        final_result = session.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as with_embeddings,
                COUNT(*) - COUNT(embedding) as without_embeddings
            FROM disclosures
        """)).fetchone()
        
        total, with_emb, without_emb = final_result
        print(f"\n📈 FINAL DATABASE STATE:")
        print(f"   Total documents: {total}")
        print(f"   With valid embeddings: {with_emb}")
        print(f"   Without embeddings: {without_emb}")
        print(f"   Coverage: {with_emb/total*100:.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)