-- Drop existing view first
DROP MATERIALIZED VIEW IF EXISTS mv_flat_facts;

/*
  This materialized view shows ALL facts from financial_facts table
  without any precedence filtering or data cleaning.
  
  It includes comprehensive metadata from the updated schema:
  - Raw concept tags and labels
  - Filing metadata (accounting standard, consolidated flag, etc.)
  - Context information
  - Audit trail information
*/

CREATE MATERIALIZED VIEW mv_flat_facts AS
SELECT
    /* identification */
  f.id AS fact_id,
  c.company_code AS ticker,
  c.name_en AS company_name,

    /* period */
  xf.period_start,
  xf.period_end,
  xf.fiscal_year,
  xf.fiscal_quarter,

    /* statement / metric */
  fs.statement_role_ja,
  fs.statement_role_en,
  fs.period_prefix,
  fs.consolidated,
  fs.layout_code,
  COALESCE(con.std_label_ja, con.local_name) AS canonical_name_jp,
  COALESCE(con.std_label_en, con.local_name) AS canonical_name_en,
  con.item_type AS concept_type,
  con.taxonomy_prefix,
  con.local_name AS concept_local_name,
  f.value,

    /* units */
  COALESCE(u.unit_code, 'JPY_Mil') AS unit_code,
  COALESCE(u.scale, 6) AS scale_power,
  COALESCE(u.currency, 'JPY') AS currency,

    /* context information */
  COALESCE(ctx.period_base, 'Current') AS period_base,
  COALESCE(ctx.period_type, 'Duration') AS period_type,
    ctx.fiscal_span,
  ctx.consolidated AS context_consolidated,
    ctx.forecast_variant,

  /* filing metadata */
  xf.disclosure_id,
  xf.accounting_standard,
  xf.has_consolidated AS filing_has_consolidated,
  xf.industry_code,
  
  /* context and audit trail */
  f.context_id AS context_raw,
  fs.rel_path AS source_file_path,
  
  /* filing metadata */
  xf.amendment_flag,
  xf.parent_filing_id,
  CASE 
    WHEN fs.statement_role_en = 'Summary'
    THEN 'Summary'
    ELSE 'Attachment'
  END AS source_section_type,
  xf.id AS filing_id,
  xf.company_id,
  d.disclosure_date + d.time AS filing_timestamp

FROM financial_facts f
JOIN filing_sections fs ON fs.id = f.section_id
JOIN xbrl_filings xf ON xf.id = fs.filing_id
JOIN disclosures d ON d.id = xf.disclosure_id
JOIN companies c ON c.id = xf.company_id
LEFT JOIN units u ON u.id = f.unit_id
LEFT JOIN concepts con ON con.id = f.concept_id
LEFT JOIN context_dims ctx ON ctx.context_id = f.context_id

WITH DATA;

-- Index the most common filter predicates
CREATE INDEX IF NOT EXISTS mv_flat_facts_ticker_canonical_idx
  ON mv_flat_facts (ticker, canonical_name_jp, period_end);

CREATE INDEX IF NOT EXISTS mv_flat_facts_period_idx
  ON mv_flat_facts (period_end, fiscal_quarter);

CREATE INDEX IF NOT EXISTS mv_flat_facts_amendment_idx
  ON mv_flat_facts (amendment_flag, parent_filing_id);

CREATE INDEX IF NOT EXISTS mv_flat_facts_section_type_idx
  ON mv_flat_facts (source_section_type, statement_role_en);

CREATE INDEX IF NOT EXISTS mv_flat_facts_concept_idx
  ON mv_flat_facts (taxonomy_prefix, concept_local_name);

CREATE INDEX IF NOT EXISTS mv_flat_facts_filing_timestamp_idx
  ON mv_flat_facts (filing_timestamp);

CREATE INDEX IF NOT EXISTS mv_flat_facts_period_base_idx
  ON mv_flat_facts (period_base, fiscal_span);

/* ----------------------------------------------------------------
   Refresh command to run after each ETL load:
   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_flat_facts;
   ---------------------------------------------------------------- */
