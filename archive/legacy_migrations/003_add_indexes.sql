-- 003_add_indexes.sql

-- Database Performance Indexes
-- For faster lookup of reports by company and period
CREATE INDEX idx_reports_company_period ON reports(company_id, period_end);

-- For faster lookup of reports by TDnet disclosure ID
CREATE INDEX idx_reports_disclosure ON reports(disclosure_id);

-- For faster lookup of sections belonging to a report
CREATE INDEX idx_sections_report ON report_sections(report_id);

-- For faster lookup of facts by section and canonical_name
CREATE INDEX idx_facts_lookup ON financial_facts(canonical_name, section_id);



-- Optional, for quick filtering by canonical name and value
-- CREATE INDEX idx_facts_canon_value ON financial_facts(canonical_name, value_num);

