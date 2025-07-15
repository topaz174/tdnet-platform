-- ============================================================
-- Migration 027: Remove redundant fiscal_quarter column from xbrl_filings
-- 
-- The fiscal_quarter column is redundant since period_type provides the same
-- information in a more standardized format. This migration removes the
-- fiscal_quarter column to simplify the schema.
-- ============================================================

-- Drop the redundant fiscal_quarter column
ALTER TABLE xbrl_filings 
DROP COLUMN IF EXISTS fiscal_quarter CASCADE;

-- Add a comment to document the change
COMMENT ON COLUMN xbrl_filings.period_type IS 'Period type from XBRL filing (e.g., Q1, Q2, Q3, FY) - replaces fiscal_quarter'; 