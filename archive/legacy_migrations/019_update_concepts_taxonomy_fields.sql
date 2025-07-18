-- Migration 019: Update concepts table taxonomy fields

-- 1. Rename 'standard' column to 'taxonomy_prefix' and adjust type
ALTER TABLE concepts 
    RENAME COLUMN standard TO taxonomy_prefix;

ALTER TABLE concepts 
    ALTER COLUMN taxonomy_prefix TYPE VARCHAR(32);

-- 2. Add taxonomy_version column
ALTER TABLE concepts 
    ADD COLUMN IF NOT EXISTS taxonomy_version DATE;

-- 3. Update any indexes or constraints that reference the old column name
DROP INDEX IF EXISTS idx_concepts_standard_local_name;
CREATE INDEX IF NOT EXISTS idx_concepts_taxonomy_prefix_local_name 
    ON concepts(taxonomy_prefix, local_name);

-- 4. Update the unique constraint if it exists
DO $$
BEGIN
    -- Drop old constraint if exists
    BEGIN
        ALTER TABLE concepts DROP CONSTRAINT IF EXISTS concepts_standard_local_name_key;
    EXCEPTION
        WHEN OTHERS THEN NULL;
    END;
    
    -- Add new constraint
    BEGIN
        ALTER TABLE concepts ADD CONSTRAINT concepts_taxonomy_prefix_local_name_key 
            UNIQUE (taxonomy_prefix, local_name);
    EXCEPTION
        WHEN duplicate_table THEN NULL;
    END;
END $$; 