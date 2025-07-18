-- ============================================================
-- Migration 010: Add Q4 Fiscal Span Support
-- Add fiscal_span = 4 to distinguish Q4/year-end from full annual totals
-- This helps differentiate between YearEndMember (Q4) and AnnualMember (full year)
-- ============================================================

-- Drop existing constraint
ALTER TABLE context_dims 
DROP CONSTRAINT IF EXISTS context_dims_fiscal_span_check;

-- Add new constraint that includes Q4 (4)
ALTER TABLE context_dims 
ADD CONSTRAINT context_dims_fiscal_span_check 
CHECK (fiscal_span IN (1, 2, 3, 4, 99) OR fiscal_span IS NULL); 