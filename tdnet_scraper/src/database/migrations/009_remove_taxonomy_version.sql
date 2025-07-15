-- ============================================================
-- Migration 009: Remove Taxonomy Version Column
-- Remove taxonomy_version column from xbrl_filings table since it's no longer needed
-- ============================================================

-- Drop the taxonomy_version column from xbrl_filings table

DROP MATERIALIZED VIEW IF EXISTS mv_flat_facts;

ALTER TABLE xbrl_filings 
DROP COLUMN IF EXISTS taxonomy_version; 