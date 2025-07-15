-- Migration 018: Introduce parent_filing_id integer foreign key and migrate existing parent_doc_ref values

-- 1. Add new column as self-referential FK
ALTER TABLE xbrl_filings
    ADD COLUMN IF NOT EXISTS parent_filing_id INTEGER REFERENCES xbrl_filings(id);

-- 2. Attempt to migrate existing parent_doc_ref values that are numeric into the new column
DO $$
BEGIN
    -- Only run if parent_doc_ref exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'xbrl_filings' AND column_name = 'parent_doc_ref'
    ) THEN
        UPDATE xbrl_filings
           SET parent_filing_id = parent_doc_ref::INTEGER
         WHERE parent_filing_id IS NULL
           AND parent_doc_ref ~ '^[0-9]+$';

        -- 3. (Optional) Drop the old column once data migrated
        ALTER TABLE xbrl_filings
            DROP COLUMN parent_doc_ref;
    END IF;
END $$;

-- 4. Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_xbrl_filings_parent_filing_id ON xbrl_filings(parent_filing_id); 