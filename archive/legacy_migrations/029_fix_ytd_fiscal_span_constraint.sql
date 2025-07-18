-- ============================================================
-- Migration 029: Fix YTD Fiscal Span Constraint
-- Update constraint to allow fiscal_span = 0 for YTD contexts
-- ============================================================

-- Drop existing constraint
ALTER TABLE context_dims 
DROP CONSTRAINT IF EXISTS context_dims_fiscal_span_check;

-- Add new constraint that includes YTD (0)
ALTER TABLE context_dims 
ADD CONSTRAINT context_dims_fiscal_span_check 
CHECK (fiscal_span IN (0, 1, 2, 3, 4, 99) OR fiscal_span IS NULL); 