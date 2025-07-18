-- psql -U your_username -d tdnet -f 001_schema.sql

-- 1. Dimension Tables

CREATE TABLE IF NOT EXISTS exchanges (
    id         SERIAL PRIMARY KEY,
    mic        VARCHAR(10) UNIQUE NOT NULL,   -- e.g. 'XTKS'
    name_en    TEXT NOT NULL,
    name_ja    TEXT
);

CREATE TABLE IF NOT EXISTS sectors (
    id         SERIAL PRIMARY KEY,
    code       VARCHAR(10) UNIQUE NOT NULL,   -- e.g. JPX-33 code
    name_en    TEXT NOT NULL,
    name_ja    TEXT
);

CREATE TABLE IF NOT EXISTS companies (
    id            SERIAL PRIMARY KEY,
    company_code  VARCHAR(5) UNIQUE NOT NULL, -- e.g. '6758'
    name_en       TEXT NOT NULL,
    name_ja       TEXT,
    exchange_id   INTEGER REFERENCES exchanges(id),
    sector_id     INTEGER REFERENCES sectors(id),
    created_at    TIMESTAMP DEFAULT now()
);

-- 1c. Context & Unit Lookup

CREATE TABLE IF NOT EXISTS units (
    id          SERIAL PRIMARY KEY,
    currency    CHAR(3) NOT NULL DEFAULT 'JPY',
    scale       SMALLINT NOT NULL,            -- 6=millions, 0=ones, -3=thousands
    unit_code   VARCHAR(20) UNIQUE,           -- e.g. 'JPY_Mil', 'PCT'
    note        TEXT
);



-- 2. Reports (One row per filing)

CREATE TABLE IF NOT EXISTS reports (
    id                SERIAL PRIMARY KEY,
    company_id        INTEGER NOT NULL REFERENCES companies(id),
    disclosure_id     INTEGER NOT NULL REFERENCES disclosures(id),
    period_start      DATE NOT NULL,
    period_end        DATE NOT NULL,
    fiscal_year       SMALLINT GENERATED ALWAYS AS (EXTRACT(YEAR FROM period_end)) STORED,
    fiscal_quarter    SMALLINT,
    default_unit_id   INTEGER REFERENCES units(id) DEFAULT 1,  -- assumes 'JPY_Mil'
    facts_raw         JSONB,
    taxonomy_version  TEXT,
    created_at        TIMESTAMP DEFAULT now(),
    UNIQUE (company_id, period_end, report_role)
);

-- 2b. Report Sections

CREATE TABLE IF NOT EXISTS report_sections (
    id             SERIAL PRIMARY KEY,
    report_id      INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    statement_role TEXT NOT NULL,  -- e.g. 'ConsolidatedBalanceSheetTDnet'
    UNIQUE (report_id, statement_role)
);


-- 3. Financial Facts (One value per row)

CREATE TABLE IF NOT EXISTS financial_facts (
    id               BIGSERIAL PRIMARY KEY,
    section_id       INTEGER NOT NULL REFERENCES report_sections(id),
    canonical_name   TEXT NOT NULL,     -- e.g. 'Revenue', 'NetIncome'
    taxonomy_element TEXT,              -- raw XBRL tag
    context_raw      TEXT NOT NULL,     -- raw context string
    unit_id          INTEGER REFERENCES units(id),
    value_num        NUMERIC(38,0) NOT NULL,
    is_raw           BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT now(),
    UNIQUE (section_id, canonical_name, context_raw)
);




-- 4. Materialized View: Operating Margin

-- CREATE MATERIALIZED VIEW IF NOT EXISTS kpi_op_margin AS
-- SELECT
--     r.id AS report_id,
--     MAX(CASE WHEN f.canonical_name = 'OperatingIncome' THEN value_num END) AS op_inc,
--     MAX(CASE WHEN f.canonical_name = 'Revenue'         THEN value_num END) AS revenue,
--     ROUND(
--         MAX(CASE WHEN f.canonical_name = 'OperatingIncome' THEN value_num END) /
--         NULLIF(MAX(CASE WHEN f.canonical_name = 'Revenue' THEN value_num END), 0) * 100, 1
--     ) AS op_margin_pct
-- FROM reports r
-- JOIN report_sections s ON s.report_id = r.id
-- JOIN financial_facts f ON f.section_id = s.id
-- GROUP BY r.id
-- WITH DATA;

-- To refresh this view after every ETL run:
-- REFRESH MATERIALIZED VIEW CONCURRENTLY kpi_op_margin;

-- 5. ETL Tracking

CREATE TABLE IF NOT EXISTS etl_runs (
    id             SERIAL PRIMARY KEY,
    run_started_at TIMESTAMP DEFAULT now(),
    mapping_hash   CHAR(64) NOT NULL,  -- SHA-256 of columns.json
    files_parsed   INTEGER,
    rows_inserted  INTEGER
);

-- 6. Notes for AI Agent Pipeline

--   - Tag-to-canonical mappings live in columns.json (hashed into etl_runs)
--   - AI agent queries SQL views or fact tables — it never processes XBRL directly
--   - JSONB blobs stored for traceability, not analytics
--   - Downstream metrics (e.g., margins, growth rates) are recomputed, not stored
