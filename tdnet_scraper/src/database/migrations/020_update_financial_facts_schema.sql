-- ============================================================
-- Migration 020: Revamp financial_facts schema
--   • rename value_num  → value
--   • drop: canonical_name, taxonomy_element, context_raw, is_raw, decimals
--   • enforce new column order: id, section_id, concept_id, context_id, unit_id, value, created_at
-- ============================================================

BEGIN;

-- 1. Drop the existing unique constraint first
ALTER TABLE financial_facts DROP CONSTRAINT IF EXISTS financial_facts_section_concept_context_unique;

-- 2. Rename value_num to value
ALTER TABLE financial_facts RENAME COLUMN value_num TO value;

-- 3. Drop unwanted columns
ALTER TABLE financial_facts DROP COLUMN IF EXISTS canonical_name;
ALTER TABLE financial_facts DROP COLUMN IF EXISTS taxonomy_element;
ALTER TABLE financial_facts DROP COLUMN IF EXISTS context_raw;
ALTER TABLE financial_facts DROP COLUMN IF EXISTS is_raw;
ALTER TABLE financial_facts DROP COLUMN IF EXISTS decimals;

-- 4. Recreate the unique constraint
ALTER TABLE financial_facts 
    ADD CONSTRAINT financial_facts_section_concept_context_unique 
    UNIQUE (section_id, concept_id, context_id);

-- 5. Recreate indexes for performance (drop first to avoid conflicts)
DROP INDEX IF EXISTS idx_financial_facts_concept_id;
DROP INDEX IF EXISTS idx_financial_facts_context_id;
CREATE INDEX idx_financial_facts_concept_id ON financial_facts (concept_id);
CREATE INDEX idx_financial_facts_context_id ON financial_facts (context_id);

COMMIT; 