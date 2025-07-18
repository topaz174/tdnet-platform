-- ============================================================
-- Migration 022: Remove filing_sections role unique constraint
-- Allow multiple sections with same role per filing (for consolidated/non-consolidated variants)
-- ============================================================

BEGIN;

-- Drop the unique constraint that prevents multiple sections with same role
DROP INDEX IF EXISTS filing_sections_role_uniq;

COMMIT; 