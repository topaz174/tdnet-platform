-- ============================================================
-- Migration 021: Add rel_path to filing_sections
-- Store the relative path to the HTML file for each section
-- ============================================================

BEGIN;

-- 1. Add rel_path column
ALTER TABLE filing_sections 
ADD COLUMN rel_path TEXT;

-- 2. Set a temporary default for existing rows (will be updated by re-running load_xbrl_filings)
UPDATE filing_sections 
SET rel_path = CASE 
    WHEN statement_role = 'Summary' THEN 'Summary/unknown.htm'
    ELSE 'Attachment/unknown.htm'
END
WHERE rel_path IS NULL;

-- 3. Make rel_path NOT NULL
ALTER TABLE filing_sections 
ALTER COLUMN rel_path SET NOT NULL;

-- 4. Create unique indexes to guarantee constraints
-- One role per filing
CREATE UNIQUE INDEX IF NOT EXISTS filing_sections_role_uniq
    ON filing_sections (filing_id, statement_role);

-- One path per filing  
CREATE UNIQUE INDEX IF NOT EXISTS filing_sections_path_uniq
    ON filing_sections (filing_id, rel_path);

COMMIT; 