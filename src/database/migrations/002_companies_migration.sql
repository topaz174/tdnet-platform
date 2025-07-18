-- Migration to make company_master the canonical companies table
-- This migration:
-- 1. Adds a surrogate key (id) to company_master
-- 2. Drops the old companies table
-- 3. Renames company_master to companies
-- 4. Renames the sequence

BEGIN;

-- Step 1: Add surrogate key to company_master
ALTER TABLE company_master ADD COLUMN id SERIAL PRIMARY KEY;

-- Step 2: Drop any materialized views that depend on the old companies table
DROP MATERIALIZED VIEW IF EXISTS mv_flat_facts CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_disclosures_classified CASCADE;

-- Step 3: Drop the old companies table
DROP TABLE IF EXISTS companies CASCADE;

-- Step 4: Rename company_master to companies
ALTER TABLE company_master RENAME TO companies;

-- Step 5: Rename the sequence
ALTER SEQUENCE company_master_id_seq RENAME TO companies_id_seq; 

COMMIT; 