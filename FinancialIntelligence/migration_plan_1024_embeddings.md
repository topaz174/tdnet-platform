# Migration Plan: 1536 → 1024 Dimensional Embeddings

## Overview
Convert from inefficient 1536-dimensional embeddings (with zero-padding) to native 1024-dimensional embeddings from the `intfloat/multilingual-e5-large` model.

## Benefits
- **Storage Reduction**: ~35MB saved (512 dims × 4 bytes × 17,259 docs)
- **Performance Improvement**: 33% faster vector operations (1024 vs 1536)
- **Accuracy**: Eliminates zero-padding artifacts that may affect similarity
- **Simplicity**: No more dimension mismatch handling in retrieval code

## Migration Options

### Option A: In-Place Migration (Recommended)
**Safest approach with minimal downtime**

1. **Backup current embeddings** (optional safety measure):
   ```sql
   CREATE TABLE disclosures_embedding_backup AS 
   SELECT id, embedding FROM disclosures WHERE embedding IS NOT NULL;
   ```

2. **Run re-embedding script**:
   ```bash
   python re_embed_1024_dimensions.py --batch-size 50
   ```

3. **Update retrieval system** (remove padding logic)

4. **Cleanup backup** (after verification):
   ```sql
   DROP TABLE disclosures_embedding_backup;
   ```

### Option B: New Column Migration
**Safer but requires more storage temporarily**

1. **Add new column**:
   ```sql
   ALTER TABLE disclosures ADD COLUMN embedding_1024 vector(1024);
   ```

2. **Create indexes on new column**:
   ```sql
   CREATE INDEX disclosures_embedding_1024_hnsw 
   ON disclosures USING hnsw (embedding_1024 vector_cosine_ops);
   
   CREATE INDEX disclosures_embedding_1024_ivfflat 
   ON disclosures USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists=100);
   ```

3. **Run re-embedding to new column**:
   ```bash
   # Modify script to update embedding_1024 column instead
   python re_embed_1024_dimensions.py --batch-size 50
   ```

4. **Update retrieval system** to use `embedding_1024`

5. **Drop old column and rename**:
   ```sql
   ALTER TABLE disclosures DROP COLUMN embedding;
   ALTER TABLE disclosures RENAME COLUMN embedding_1024 TO embedding;
   ```

## Recommended Steps (Option A)

### Step 1: Verify Current State
```bash
python re_embed_1024_dimensions.py --verify-only
```

### Step 2: Optional Backup
```sql
-- Connect to PostgreSQL
psql postgresql://postgres:clapg1234@127.0.0.1:5432/tdnet

-- Create backup table
CREATE TABLE disclosures_embedding_backup AS 
SELECT id, embedding FROM disclosures WHERE embedding IS NOT NULL;

-- Verify backup
SELECT COUNT(*) FROM disclosures_embedding_backup;
```

### Step 3: Run Re-embedding
```bash
# Start re-embedding process
python re_embed_1024_dimensions.py --batch-size 50

# Monitor progress in re_embedding.log
tail -f re_embedding.log
```

### Step 4: Update Retrieval System
After re-embedding completes, update the enhanced retrieval system:

1. Remove zero-padding logic from `enhanced_retrieval_system.py`
2. Update target dimension from 1536 → 1024
3. Test the system

### Step 5: Verify and Cleanup
```bash
# Verify new dimensions
python re_embed_1024_dimensions.py --verify-only

# Test retrieval system
python complete_enhanced_agent.py
```

If everything works correctly:
```sql
-- Drop backup table
DROP TABLE disclosures_embedding_backup;
```

## Estimated Timeline

- **Re-embedding**: ~2-3 hours for 17,259 documents (batch size 50)
- **Index recreation**: Automatic (PostgreSQL handles this)
- **Code updates**: ~30 minutes
- **Testing**: ~30 minutes
- **Total**: ~3-4 hours

## Resuming if Interrupted

The script supports resuming from any document ID:
```bash
# If process stopped at document ID 5000
python re_embed_1024_dimensions.py --start-id 5000
```

## Risk Mitigation

1. **Backup**: Optional backup table protects against data loss
2. **Logging**: Comprehensive logging in `re_embedding.log`
3. **Batch processing**: Small batches minimize memory usage
4. **Resume capability**: Can restart from any point
5. **Verification**: Built-in dimension verification

## Performance Expectations

- **Processing rate**: ~150-200 documents/minute
- **Memory usage**: ~2-3GB during processing
- **Disk I/O**: Moderate (reading PDFs + database updates)

## Post-Migration Validation

1. Check all embeddings are 1024 dimensions
2. Verify retrieval quality with test queries
3. Confirm performance improvements
4. Monitor system stability

## Rollback Plan (if needed)

If using Option A with backup:
```sql
-- Restore from backup
UPDATE disclosures 
SET embedding = b.embedding 
FROM disclosures_embedding_backup b 
WHERE disclosures.id = b.id;
```

## Code Changes Required

After migration, update `enhanced_retrieval_system.py`:

```python
# Remove this padding logic:
def _generate_embedding(self, text: str) -> np.ndarray:
    emb = self.dense.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    
    # OLD: Padding logic (remove this)
    # target_dim = 1536
    # if len(emb) < target_dim:
    #     padded = np.zeros(target_dim)
    #     padded[:len(emb)] = emb
    #     norm = np.linalg.norm(padded)
    #     if norm > 0:
    #         padded = padded / norm
    #     return padded
    
    # NEW: Direct return (native 1024 dimensions)
    return emb
```