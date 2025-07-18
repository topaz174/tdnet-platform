-- ============================================================
-- Migration 028: Re-add fiscal_quarter column to xbrl_filings
--
-- Rationale: period_type values (e.g., 'FY', 'Year') are inconsistently
-- labelled across filings.  A numeric fiscal_quarter (1-4, 99=Year)
-- provides a stable, sortable value for analytics.
-- ============================================================

-- 1. Add column back if missing
ALTER TABLE xbrl_filings
    ADD COLUMN IF NOT EXISTS fiscal_quarter SMALLINT;

-- 2. Populate missing values from period_type where possible
-- Map common period_type strings to quarter numbers
UPDATE xbrl_filings
SET fiscal_quarter = CASE UPPER(COALESCE(period_type, ''))
    WHEN 'Q1'            THEN 1
    WHEN 'FIRSTQUARTER'  THEN 1
    WHEN 'Q2'            THEN 2
    WHEN 'SECONDQUARTER' THEN 2
    WHEN 'HALFYEAR'      THEN 2
    WHEN 'Q3'            THEN 3
    WHEN 'THIRDQUARTER'  THEN 3
    WHEN 'Q4'            THEN 99
    WHEN 'YEAR'          THEN 99
    WHEN 'FY'            THEN 99
    ELSE fiscal_quarter  -- leave as-is if unknown
END
WHERE fiscal_quarter IS NULL;

-- 3. Index for common queries
CREATE INDEX IF NOT EXISTS idx_xbrl_filings_period_quarter
    ON xbrl_filings (period_end, fiscal_quarter); 