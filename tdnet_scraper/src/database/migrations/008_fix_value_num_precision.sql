-- ============================================================
-- Migration 008: Fix Value Precision for Decimal Support
-- Change value_num from NUMERIC(38,0) to NUMERIC(38,10) to support decimal values
-- This is critical for ratio/percentage facts that have decimal precision
-- ============================================================

-- Drop materialized view that depends on value_num column
DROP MATERIALIZED VIEW IF EXISTS mv_flat_facts;

-- Update value_num column to support decimal places
ALTER TABLE financial_facts 
ALTER COLUMN value_num TYPE NUMERIC(38,2);

-- Add PCT unit for percentage values (scale=-2 means divide by 100)
INSERT INTO units (id, currency, scale, unit_code, note) 
VALUES (7, 'PUR', -2, 'Percent', 'Percentage values (scale -2, divide by 100)')
ON CONFLICT (id) DO NOTHING;