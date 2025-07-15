-- ============================================================
-- Migration 024: Revamp filing_sections schema
-- Add period_prefix, consolidated, layout_code columns
-- Update statement_role to use Japanese and English names
-- ============================================================

BEGIN;

-- Add new columns to filing_sections
ALTER TABLE filing_sections 
ADD COLUMN period_prefix CHAR(1),
ADD COLUMN consolidated BOOLEAN,
ADD COLUMN layout_code SMALLINT,
ADD COLUMN statement_role_ja TEXT,
ADD COLUMN statement_role_en TEXT;

-- Drop the original statement_role column since we now have ja and en versions
ALTER TABLE filing_sections DROP COLUMN statement_role CASCADE;

-- Update the unique constraint to include layout_code
-- First drop the old constraint if it exists
DROP INDEX IF EXISTS filing_sections_role_uniq;

-- Create new unique constraint including layout_code
CREATE UNIQUE INDEX filing_sections_unique_idx 
ON filing_sections (filing_id, statement_role_ja, period_prefix, consolidated, layout_code);

COMMIT; 