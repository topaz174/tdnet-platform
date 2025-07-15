-- Drop existing view first
DROP MATERIALIZED VIEW IF EXISTS mv_disclosures_classified;

CREATE MATERIALIZED VIEW mv_disclosures_classified AS
SELECT
    /* core disclosure information */
    d.id,
    d.disclosure_date,
    d.time,
    d.company_code,
    d.company_name,
    d.title,
    d.xbrl_url,
    d.has_xbrl,
    d.exchange,
    d.update_history,
    d.page_number,
    d.scraped_at,

    /* classification from disclosure_labels table - as JSON arrays with bilingual support */
    JSONB_AGG(DISTINCT jsonb_build_object(
        'en', dc.name,
        'jp', dc.name_jp
    )) FILTER (WHERE dc.name IS NOT NULL) AS categories,
    
    JSONB_AGG(DISTINCT jsonb_build_object(
        'en', ds.name,
        'jp', ds.name_jp
    )) FILTER (WHERE ds.name IS NOT NULL) AS subcategories,
    
    MAX(dl.labeled_at) AS classified_at

FROM disclosures d
LEFT JOIN disclosure_labels dl ON dl.disclosure_id = d.id
LEFT JOIN disclosure_categories dc ON dc.id = dl.category_id
LEFT JOIN disclosure_subcategories ds ON ds.id = dl.subcat_id
GROUP BY d.id, d.disclosure_date, d.time, d.company_code, d.company_name, d.title, 
         d.xbrl_url, d.has_xbrl, d.exchange, d.update_history, 
         d.page_number, d.scraped_at

WITH DATA;

-- Index the most common filter predicates
CREATE INDEX IF NOT EXISTS mv_disclosures_classified_date_idx
  ON mv_disclosures_classified (disclosure_date DESC);

CREATE INDEX IF NOT EXISTS mv_disclosures_classified_company_idx
  ON mv_disclosures_classified (company_code, disclosure_date DESC);

CREATE INDEX IF NOT EXISTS mv_disclosures_classified_classified_at_idx
  ON mv_disclosures_classified (classified_at DESC);

/* ----------------------------------------------------------------
   Refresh command to run after classification updates:
   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_disclosures_classified;
   ---------------------------------------------------------------- */ 