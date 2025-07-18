-- Migration: Rename reports table to xbrl_filings
-- This migration renames the reports table to better reflect its purpose as a table
-- containing all disclosures with XBRL files (not just earnings reports)

-- 1. Rename the main table
ALTER TABLE reports RENAME TO xbrl_filings;

-- 2. Update the foreign key reference in report_sections
ALTER TABLE report_sections RENAME COLUMN report_id TO filing_id;

-- 3. Rename report_sections to filing_sections for consistency
ALTER TABLE report_sections RENAME TO filing_sections;

-- 4. Update indexes that reference the old table name
DROP INDEX IF EXISTS idx_reports_company_period;
DROP INDEX IF EXISTS idx_reports_disclosure;
DROP INDEX IF EXISTS idx_sections_report;

-- Recreate indexes with new names
CREATE INDEX idx_xbrl_filings_company_period ON xbrl_filings(company_id, period_end);
CREATE INDEX idx_xbrl_filings_disclosure ON xbrl_filings(disclosure_id);
CREATE INDEX idx_filing_sections_filing ON filing_sections(filing_id);

-- 5. Update the unique constraint that had report_role (if it exists)
-- Note: This constraint may have been removed in a previous migration
-- We'll try to update it but handle the case where it doesn't exist
DO $$
BEGIN
    -- Try to drop the old constraint if it exists
    BEGIN
        ALTER TABLE xbrl_filings DROP CONSTRAINT IF EXISTS reports_company_id_period_end_report_role_key;
    EXCEPTION
        WHEN OTHERS THEN
            -- Constraint doesn't exist, that's fine
            NULL;
    END;
    
    -- Add the new constraint (without report_role since it was removed)
    BEGIN
        ALTER TABLE xbrl_filings ADD CONSTRAINT xbrl_filings_company_id_period_end_key 
        UNIQUE (company_id, period_end);
    EXCEPTION
        WHEN duplicate_table THEN
            -- Constraint already exists, that's fine
            NULL;
    END;
END $$; 