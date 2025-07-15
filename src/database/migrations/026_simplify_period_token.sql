-- ============================================================
-- Migration 026: Simplify period_token and fix column name
-- 
-- 1. Fix the typo: period_atoken -> period_token
-- 2. Replace period_token with simplified period_base (Current, Next, Prior, etc.)
-- 3. Handle YTD (Year-to-Date) cases with fiscal_span = 0
-- ============================================================

-- First, rename the column to fix the typo
ALTER TABLE context_dims RENAME COLUMN period_atoken TO period_token;

-- Replace period_token with period_base (simplified version)
ALTER TABLE context_dims RENAME COLUMN period_token TO period_base;

-- Update the index to use the new column name
DROP INDEX IF EXISTS idx_context_dims_period_token;
CREATE INDEX idx_context_dims_period_base ON context_dims(period_base);

-- Add a comment explaining the new structure
COMMENT ON COLUMN context_dims.period_base IS 'Simplified period base (e.g., Current, Next, Prior) - just the first word from original period token';
COMMENT ON COLUMN context_dims.fiscal_span IS 'Fiscal span: 0=YTD, 1=Q1, 2=Q2, 3=Q3, 99=Year'; 