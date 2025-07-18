-- ============================================================
-- Migration 025: Rename consolidated_flag to has_consolidated in xbrl_filings
-- ============================================================

BEGIN;

ALTER TABLE xbrl_filings
RENAME COLUMN consolidated_flag TO has_consolidated;

COMMIT; 