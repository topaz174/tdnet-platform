-- ============================================================
-- Migration 005: Normalize Concepts and Contexts
-- Create lookup tables for concepts and contexts to replace 
-- raw string storage in financial_facts table
-- ============================================================

-- Canonical financial concepts (metrics)
CREATE TABLE IF NOT EXISTS concepts (
    id       SERIAL PRIMARY KEY,
    name_en  TEXT UNIQUE NOT NULL,  -- e.g. 'Revenue'
    name_ja  TEXT UNIQUE NOT NULL   -- e.g. '売上高'
);

-- Raw XBRL tags mapped to canonical concepts
CREATE TABLE IF NOT EXISTS concept_tags (
    raw_tag          TEXT PRIMARY KEY,           -- e.g. 'NetSalesIFRS'
    concept_id       INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
);

-- Canonical XBRL contexts
CREATE TABLE IF NOT EXISTS  context_dims (
    /* the literal string from XBRL, e.g. "CurrentAccumulatedQ3Duration_ConsolidatedMember_ResultMember" */
    context_id        TEXT PRIMARY KEY,

    /* core decomposition */
    period_atoken      TEXT      NOT NULL,               -- CurrentAccumultedQ3, PriorYear etc.
    period_type       TEXT      NOT NULL  CHECK (period_type IN ('Instant','Duration')),
    fiscal_span SMALLINT CHECK (fiscal_span IN (1, 2, 3, 99) OR fiscal_span IS NULL),
    consolidated      BOOLEAN,
    forecast_variant  TEXT      CHECK (forecast_variant IN ('Result', 'Forecast', 'Upper', 'Lower')),

    other_members     JSONB
);

-- Create indexes for lookup performance
CREATE INDEX idx_concept_tags_concept_id ON concept_tags(concept_id);
CREATE INDEX idx_context_dims_period_token ON context_dims(period_token);
CREATE INDEX idx_context_dims_fiscal_span ON context_dims(fiscal_span); 