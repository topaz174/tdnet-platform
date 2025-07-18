-- Drop existing view first
DROP MATERIALIZED VIEW IF EXISTS mv_facts_cleaned;

/*
  This materialized view implements fact-level data cleaning logic:
  
  1. FACT-LEVEL PRECEDENCE: For each unique (company, period_end, concept, context) combination,
     select the best fact based on precedence rules.
     
  2. CORRECT PRECEDENCE ORDER:
     - Amendment facts over original facts
     - Attachment facts over summary facts  
     - Newer filings as tie-breaker
     
  3. COMPREHENSIVE COVERAGE: Show summary facts that don't exist in attachments,
     and original facts that don't exist in amendments.
*/

CREATE MATERIALIZED VIEW mv_facts_cleaned AS
WITH
-- Step 1: Enrich every fact with metadata needed for ranking
fact_plus AS (
    SELECT
    f.id AS fact_id,
    f.section_id,
    f.concept_id,
    f.context_id,
    f.unit_id,
    f.value,
    
    -- Filing info
    xf.id AS filing_id,
        xf.company_id,
        xf.period_start,
        xf.period_end,
        xf.fiscal_year,
            xf.period_type,
    xf.amendment_flag,
    xf.parent_filing_id,
    xf.disclosure_id,
    xf.accounting_standard,
    xf.has_consolidated,
    xf.industry_code,
    d.disclosure_date + d.time AS filing_timestamp,
    
    -- Section info
    fs.statement_role_ja,
    fs.statement_role_en,
    fs.period_prefix,
    fs.consolidated,
    fs.layout_code,
    fs.rel_path,
    
    -- Determine section type for precedence
  CASE 
    WHEN fs.statement_role_en = 'Summary'
    THEN 'Summary'
    ELSE 'Attachment'
    END AS section_type
    
  FROM financial_facts f
  JOIN filing_sections fs ON fs.id = f.section_id
  JOIN xbrl_filings xf ON xf.id = fs.filing_id
  JOIN disclosures d ON d.id = xf.disclosure_id
),

-- Step 2: Apply fact-level precedence ranking
-- For each unique (company, period_end, concept, context) combination, 
-- rank facts by precedence rules
ranked_facts AS (
    SELECT
    fp.*,
        ROW_NUMBER() OVER (
      PARTITION BY fp.company_id, fp.period_end, fp.concept_id, fp.context_id
            ORDER BY
        -- 1. Amendment facts over original facts (amendments have priority)
        fp.amendment_flag DESC,
        -- 2. Attachment facts over summary facts (attachments have priority)
        CASE fp.section_type
          WHEN 'Attachment' THEN 3
          WHEN 'Summary' THEN 2
          WHEN 'Other' THEN 1
                     ELSE 0
                END DESC,
        -- 3. Newer filings as tie-breaker
        fp.filing_timestamp DESC,
        -- 4. Higher fact ID as final tie-breaker
        fp.fact_id DESC
    ) AS precedence_rank
  FROM fact_plus fp
  -- Skip narrative and other non-financial sections
  WHERE fp.section_type IN ('Attachment', 'Summary')
)

-- Step 3: Select the highest-precedence fact from each group and join descriptive tables
SELECT
  /* identification */
  rf.fact_id,
  c.ticker AS ticker,
  c.company_name_english AS company_name_en,

  /* period */
  rf.period_start,
  rf.period_end,
  rf.fiscal_year,
  rf.fiscal_quarter,

  /* statement / metric */
  rf.statement_role_ja,
  rf.statement_role_en,
  rf.period_prefix,
  rf.consolidated,
  rf.layout_code,
  COALESCE(con.std_label_ja, con.local_name) AS canonical_name_jp,
    COALESCE(con.std_label_en, con.local_name) AS canonical_name_en,
  con.item_type AS concept_type,
  con.taxonomy_prefix,
  con.local_name AS concept_local_name,
  rf.value,

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
  rf.disclosure_id,
  rf.accounting_standard,
  rf.has_consolidated AS filing_has_consolidated,
  rf.industry_code,
  
  /* context and audit trail */
  rf.context_id AS context_raw,
  rf.rel_path AS source_file_path,
  
  /* cleaning metadata */
  rf.amendment_flag,
  rf.parent_filing_id,
  rf.section_type AS source_section_type,
  rf.filing_id,
  rf.company_id,
  rf.filing_timestamp,
  rf.precedence_rank

FROM ranked_facts rf
JOIN companies c ON c.id = rf.company_id
LEFT JOIN units u ON u.id = rf.unit_id
LEFT JOIN concepts con ON con.id = rf.concept_id
LEFT JOIN context_dims ctx ON ctx.context_id = rf.context_id
WHERE rf.precedence_rank = 1  -- Only keep the highest-precedence fact from each group

WITH DATA;

-- Index the most common filter predicates
CREATE INDEX IF NOT EXISTS mv_facts_cleaned_ticker_canonical_idx
  ON mv_facts_cleaned (ticker, canonical_name_jp, period_end);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_period_idx
  ON mv_facts_cleaned (period_end, fiscal_quarter);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_amendment_idx
  ON mv_facts_cleaned (amendment_flag, parent_filing_id);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_section_type_idx
  ON mv_facts_cleaned (source_section_type, statement_role_en);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_concept_idx
  ON mv_facts_cleaned (taxonomy_prefix, concept_local_name);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_filing_timestamp_idx
  ON mv_facts_cleaned (filing_timestamp);

CREATE INDEX IF NOT EXISTS mv_facts_cleaned_period_base_idx
  ON mv_facts_cleaned (period_base, fiscal_span);

/* ----------------------------------------------------------------
   Refresh command to run after each ETL load:
   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_facts_cleaned;
   ---------------------------------------------------------------- */ 