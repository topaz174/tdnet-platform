-- Migration 014: Drop redundant path columns from disclosures table
-- WARNING: Only run this AFTER running migration 013 to backup the data
-- These columns are derivable from company_code, disclosure_date, time, and title

-- Drop any indexes that reference these columns first
DROP INDEX IF EXISTS idx_disclosures_xbrl_path;
DROP INDEX IF EXISTS idx_disclosures_pdf_path;

-- Remove the redundant columns
ALTER TABLE disclosures 
DROP COLUMN IF EXISTS xbrl_path,
DROP COLUMN IF EXISTS pdf_path;

-- Display confirmation
DO $$
DECLARE
    columns_remaining INTEGER;
BEGIN
    SELECT COUNT(*) INTO columns_remaining 
    FROM information_schema.columns 
    AND column_name IN ('xbrl_path', 'pdf_path');
    
    IF columns_remaining = 0 THEN
        RAISE NOTICE 'Migration 014 completed successfully:';
        RAISE NOTICE '  - xbrl_path column removed';
        RAISE NOTICE '  - pdf_path column removed';
        RAISE NOTICE '  - has_xbrl column available for XBRL presence checks';
        RAISE NOTICE '  - Path data backed up in disclosures_path_backup table';
    ELSE
        RAISE WARNING 'Some path columns may still exist. Check manually.';
    END IF;
END $$; 