-- Migration 012: Add has_xbrl boolean column to disclosures table
-- This migration adds a boolean column to indicate whether a disclosure has a valid XBRL file
-- The column is populated based on the current xbrl_path values (non-null and non-empty)

-- Add the has_xbrl column as nullable first
ALTER TABLE disclosures 
ADD COLUMN has_xbrl BOOLEAN;

-- Populate the column based on existing xbrl_path values
-- Set to TRUE if xbrl_path is not null and not empty string
-- Set to FALSE otherwise
UPDATE disclosures 
SET has_xbrl = (
    xbrl_path IS NOT NULL 
    AND xbrl_path != '' 
    AND xbrl_path != 'NULL'  -- Handle cases where 'NULL' was stored as string
);

-- Set the column to NOT NULL with default FALSE after population
ALTER TABLE disclosures 
ALTER COLUMN has_xbrl SET NOT NULL,
ALTER COLUMN has_xbrl SET DEFAULT FALSE;

-- Add an index on has_xbrl for performance when filtering
CREATE INDEX IF NOT EXISTS idx_disclosures_has_xbrl 
ON disclosures (has_xbrl);

-- Add a partial index for disclosures with XBRL files (most useful queries)
CREATE INDEX IF NOT EXISTS idx_disclosures_has_xbrl_true 
ON disclosures (disclosure_date DESC) 
WHERE has_xbrl = TRUE;

-- Display statistics after migration
DO $$
DECLARE
    total_count INTEGER;
    xbrl_count INTEGER;
    xbrl_percentage NUMERIC;
BEGIN
    SELECT COUNT(*) INTO total_count FROM disclosures;
    SELECT COUNT(*) INTO xbrl_count FROM disclosures WHERE has_xbrl = TRUE;
    
    IF total_count > 0 THEN
        xbrl_percentage := ROUND((xbrl_count::NUMERIC / total_count::NUMERIC) * 100, 2);
    ELSE
        xbrl_percentage := 0;
    END IF;
    
    RAISE NOTICE 'Migration 012 completed:';
    RAISE NOTICE '  Total disclosures: %', total_count;
    RAISE NOTICE '  Disclosures with XBRL: %', xbrl_count;
    RAISE NOTICE '  Percentage with XBRL: %%%', xbrl_percentage;
END $$; 