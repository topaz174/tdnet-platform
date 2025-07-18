-- ============================================================
-- Migration 023: Drop category and subcategory columns from disclosures
-- These columns are no longer needed as we have normalized tables
-- ============================================================

BEGIN;

-- Drop the category and subcategory columns from disclosures table
ALTER TABLE disclosures DROP COLUMN IF EXISTS category CASCADE;
ALTER TABLE disclosures DROP COLUMN IF EXISTS subcategory CASCADE;

COMMIT; 