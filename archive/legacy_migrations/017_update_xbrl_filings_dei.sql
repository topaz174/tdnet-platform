-- Migration 017: Add DEI metadata columns to xbrl_filings and remove obsolete columns

/*
    New columns added (all nullable for back-compat):
        - accounting_standard   VARCHAR(10)
        - consolidated_flag    BOOLEAN
        - industry_code        VARCHAR(10)
        - period_type          VARCHAR(8)
        - submission_no        INTEGER
        - amendment_flag       BOOLEAN
        - report_amendment_flag BOOLEAN
        - xbrl_amendment_flag  BOOLEAN
        - parent_doc_ref       TEXT

    Removed columns:
        - default_unit_id
        - facts_raw
*/

-- 1. Add the new columns
ALTER TABLE xbrl_filings
    ADD COLUMN IF NOT EXISTS accounting_standard        VARCHAR(10),
    ADD COLUMN IF NOT EXISTS consolidated_flag         BOOLEAN,
    ADD COLUMN IF NOT EXISTS industry_code             VARCHAR(10),
    ADD COLUMN IF NOT EXISTS period_type               VARCHAR(8),
    ADD COLUMN IF NOT EXISTS submission_no             INTEGER,
    ADD COLUMN IF NOT EXISTS amendment_flag            BOOLEAN,
    ADD COLUMN IF NOT EXISTS report_amendment_flag     BOOLEAN,
    ADD COLUMN IF NOT EXISTS xbrl_amendment_flag       BOOLEAN,
    ADD COLUMN IF NOT EXISTS parent_doc_ref            TEXT;

-- 2. Drop obsolete columns
ALTER TABLE xbrl_filings
    DROP COLUMN IF EXISTS default_unit_id,
    DROP COLUMN IF EXISTS facts_raw;

-- 3. Optional indexes for frequent queries
CREATE INDEX IF NOT EXISTS idx_xbrl_filings_accounting_standard ON xbrl_filings(accounting_standard);
CREATE INDEX IF NOT EXISTS idx_xbrl_filings_industry_code       ON xbrl_filings(industry_code);
CREATE INDEX IF NOT EXISTS idx_xbrl_filings_period_type        ON xbrl_filings(period_type); 