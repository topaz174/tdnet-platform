-- SQL Migration to Unify Schemas (Additive Only)
-- This script merges the target schema into the current one without deleting any existing objects.

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;


BEGIN;

-- =============================================================================
-- Step 1: Create New Tables, Sequences, and Types
-- All objects from the target schema that do not exist in the current schema are created here.
-- =============================================================================

-- Dimension tables for company information
CREATE TABLE public.sectors (
    id integer NOT NULL,
    code character varying(10) NOT NULL,
    name_en text NOT NULL,
    name_ja text
);
CREATE SEQUENCE public.sectors_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.sectors ALTER COLUMN id SET DEFAULT nextval('public.sectors_id_seq'::regclass);
ALTER SEQUENCE public.sectors_id_seq OWNED BY public.sectors.id;

CREATE TABLE public.exchanges (
    id integer NOT NULL,
    mic character varying(10) NOT NULL,
    name_en text NOT NULL,
    name_ja text
);
CREATE SEQUENCE public.exchanges_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.exchanges ALTER COLUMN id SET DEFAULT nextval('public.exchanges_id_seq'::regclass);
ALTER SEQUENCE public.exchanges_id_seq OWNED BY public.exchanges.id;

-- The new `companies` table will coexist with the old `company_master` table.
CREATE TABLE public.companies (
    id integer NOT NULL,
    company_code character varying(5) NOT NULL,
    name_en text NOT NULL,
    name_ja text,
    exchange_id integer,
    sector_id integer,
    created_at timestamp without time zone DEFAULT now()
);
CREATE SEQUENCE public.companies_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.companies ALTER COLUMN id SET DEFAULT nextval('public.companies_id_seq'::regclass);
ALTER SEQUENCE public.companies_id_seq OWNED BY public.companies.id;


-- Tables for disclosure classification
CREATE TABLE public.disclosure_categories (
    id integer NOT NULL,
    name text NOT NULL,
    name_jp text
);
CREATE SEQUENCE public.disclosure_categories_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.disclosure_categories ALTER COLUMN id SET DEFAULT nextval('public.disclosure_categories_id_seq'::regclass);
ALTER SEQUENCE public.disclosure_categories_id_seq OWNED BY public.disclosure_categories.id;

CREATE TABLE public.disclosure_subcategories (
    id integer NOT NULL,
    category_id integer NOT NULL,
    name text NOT NULL,
    name_jp text
);
CREATE SEQUENCE public.disclosure_subcategories_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.disclosure_subcategories ALTER COLUMN id SET DEFAULT nextval('public.disclosure_subcategories_id_seq'::regclass);
ALTER SEQUENCE public.disclosure_subcategories_id_seq OWNED BY public.disclosure_subcategories.id;

CREATE TABLE public.disclosure_labels (
    id integer NOT NULL,
    disclosure_id integer NOT NULL,
    category_id integer,
    subcat_id integer,
    labeled_at timestamp without time zone DEFAULT now()
);
CREATE SEQUENCE public.disclosure_labels_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.disclosure_labels ALTER COLUMN id SET DEFAULT nextval('public.disclosure_labels_id_seq'::regclass);
ALTER SEQUENCE public.disclosure_labels_id_seq OWNED BY public.disclosure_labels.id;

-- XBRL-related tables. `xbrl_filings` will coexist with the old `reports` table.
CREATE TABLE public.xbrl_filings (
    id integer NOT NULL,
    company_id integer NOT NULL,
    disclosure_id integer NOT NULL,
    period_start date NOT NULL,
    period_end date NOT NULL,
    accounting_standard character varying(10),
    has_consolidated boolean,
    industry_code character varying(10),
    period_type character varying(8),
    submission_no integer,
    amendment_flag boolean,
    report_amendment_flag boolean,
    xbrl_amendment_flag boolean,
    parent_filing_id integer,
    fiscal_quarter smallint,
    created_at timestamp without time zone DEFAULT now(),
    fiscal_year smallint GENERATED ALWAYS AS (EXTRACT(year FROM period_end)) STORED
);
COMMENT ON COLUMN public.xbrl_filings.period_type IS 'Period type from XBRL filing (e.g., Q1, Q2, Q3, FY) - replaces fiscal_quarter';
CREATE SEQUENCE public.xbrl_filings_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.xbrl_filings ALTER COLUMN id SET DEFAULT nextval('public.xbrl_filings_id_seq'::regclass);
ALTER SEQUENCE public.xbrl_filings_id_seq OWNED BY public.xbrl_filings.id;

CREATE TABLE public.filing_sections (
    id integer NOT NULL,
    filing_id integer NOT NULL,
    rel_path text NOT NULL,
    period_prefix character(1),
    consolidated boolean,
    layout_code smallint,
    statement_role_ja text,
    statement_role_en text
);
CREATE SEQUENCE public.filing_sections_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.filing_sections ALTER COLUMN id SET DEFAULT nextval('public.filing_sections_id_seq'::regclass);
ALTER SEQUENCE public.filing_sections_id_seq OWNED BY public.filing_sections.id;

CREATE TABLE public.concepts (
    id integer NOT NULL,
    taxonomy_prefix character varying(32) NOT NULL,
    local_name text NOT NULL,
    std_label_en text,
    std_label_ja text,
    item_type character varying(12),
    taxonomy_version date
);
CREATE SEQUENCE public.concepts_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.concepts ALTER COLUMN id SET DEFAULT nextval('public.concepts_id_seq'::regclass);
ALTER SEQUENCE public.concepts_id_seq OWNED BY public.concepts.id;

CREATE TABLE public.concept_overrides (
    concept_id integer NOT NULL,
    override_label_en text,
    override_label_ja text
);

CREATE TABLE public.context_dims (
    context_id text NOT NULL,
    period_base text NOT NULL,
    period_type text NOT NULL,
    fiscal_span smallint,
    consolidated boolean,
    forecast_variant text,
    CONSTRAINT context_dims_fiscal_span_check CHECK (((fiscal_span = ANY (ARRAY[0, 1, 2, 3, 4, 99])) OR (fiscal_span IS NULL))),
    CONSTRAINT context_dims_forecast_variant_check CHECK ((forecast_variant = ANY (ARRAY['Result'::text, 'Forecast'::text, 'Upper'::text, 'Lower'::text]))),
    CONSTRAINT context_dims_period_type_check CHECK ((period_type = ANY (ARRAY['Instant'::text, 'Duration'::text])))
);
COMMENT ON COLUMN public.context_dims.period_base IS 'Simplified period base (e.g., Current, Next, Prior) - just the first word from original period token';
COMMENT ON COLUMN public.context_dims.fiscal_span IS 'Fiscal span: 0=YTD, 1=Q1, 2=Q2, 3=Q3, 99=Year';

CREATE TABLE public.units (
    id integer NOT NULL,
    currency character(3) DEFAULT 'JPY'::bpchar NOT NULL,
    scale smallint NOT NULL,
    unit_code character varying(20),
    note text
);
CREATE SEQUENCE public.units_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.units ALTER COLUMN id SET DEFAULT nextval('public.units_id_seq'::regclass);
ALTER SEQUENCE public.units_id_seq OWNED BY public.units.id;

CREATE TABLE public.financial_facts (
    id bigint NOT NULL,
    section_id integer NOT NULL,
    unit_id integer,
    value numeric(38,2) NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    concept_id integer NOT NULL,
    context_id text NOT NULL
);
CREATE SEQUENCE public.financial_facts_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.financial_facts ALTER COLUMN id SET DEFAULT nextval('public.financial_facts_id_seq'::regclass);
ALTER SEQUENCE public.financial_facts_id_seq OWNED BY public.financial_facts.id;

CREATE TABLE public.etl_runs (
    id integer NOT NULL,
    run_started_at timestamp without time zone DEFAULT now(),
    mapping_hash character(64) NOT NULL,
    files_parsed integer,
    rows_inserted integer
);
CREATE SEQUENCE public.etl_runs_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER TABLE public.etl_runs ALTER COLUMN id SET DEFAULT nextval('public.etl_runs_id_seq'::regclass);
ALTER SEQUENCE public.etl_runs_id_seq OWNED BY public.etl_runs.id;

-- =============================================================================
-- Step 2: Alter Existing Tables
-- This section adds new columns to tables that already exist.
-- =============================================================================

-- Add the 'has_xbrl' column to the 'disclosures' table. All other columns are kept.
ALTER TABLE public.disclosures ADD COLUMN IF NOT EXISTS has_xbrl boolean NOT NULL DEFAULT false;

-- =============================================================================
-- Step 3: Populate New Tables and Columns
-- =============================================================================

-- Populate dimension tables from existing data
INSERT INTO public.sectors (code, name_en, name_ja)
SELECT 'TSE' || LPAD(ROW_NUMBER() OVER (ORDER BY sector_english)::text, 4, '0'), sector_english, sector_japanese
FROM (SELECT DISTINCT sector_english, sector_japanese FROM public.company_master WHERE sector_english IS NOT NULL) as s;

INSERT INTO public.exchanges (mic, name_en, name_ja)
SELECT
    CASE
        WHEN exchange = 'TSE' THEN 'XJPX' WHEN exchange = 'NSE' THEN 'XNGO'
        WHEN exchange = 'SSE' THEN 'XFKA' WHEN exchange = 'FSE' THEN 'XSAP'
        ELSE 'XXXX'
    END,
    exchange,
    CASE
        WHEN exchange = 'TSE' THEN '東京証券取引所' WHEN exchange = 'NSE' THEN '名古屋証券取引所'
        WHEN exchange = 'SSE' THEN '札幌証券取引所' WHEN exchange = 'FSE' THEN '福岡証券取引所'
        ELSE exchange
    END
FROM (SELECT DISTINCT exchange FROM public.disclosures WHERE exchange IS NOT NULL) as e;

-- Populate the new 'companies' table from the existing 'company_master'
INSERT INTO public.companies (company_code, name_en, name_ja, sector_id, created_at)
SELECT cm.securities_code::varchar(5), cm.company_name_english, cm.company_name_japanese, s.id, cm.created_at
FROM public.company_master cm
LEFT JOIN public.sectors s ON cm.sector_english = s.name_en AND cm.sector_japanese = s.name_ja;

-- Populate new classification tables from the existing `category` and `subcategory` columns.
INSERT INTO public.disclosure_categories (name)
SELECT DISTINCT category FROM public.disclosures WHERE category IS NOT NULL AND category <> '';

INSERT INTO public.disclosure_subcategories (category_id, name)
SELECT DISTINCT dc.id, d.subcategory
FROM public.disclosures d
JOIN public.disclosure_categories dc ON d.category = dc.name
WHERE d.subcategory IS NOT NULL AND d.subcategory <> '';

INSERT INTO public.disclosure_labels (disclosure_id, category_id, subcat_id, labeled_at)
SELECT d.id, dc.id, dsc.id, d.scraped_at
FROM public.disclosures d
LEFT JOIN public.disclosure_categories dc ON d.category = dc.name
LEFT JOIN public.disclosure_subcategories dsc ON d.subcategory = dsc.name AND dc.id = dsc.category_id
WHERE d.category IS NOT NULL OR d.subcategory IS NOT NULL;

-- Populate the new 'has_xbrl' column
UPDATE public.disclosures SET has_xbrl = (xbrl_url IS NOT NULL AND xbrl_url <> '');

-- Populate the new 'xbrl_filings' table from the existing 'reports' table.
INSERT INTO public.xbrl_filings (disclosure_id, company_id, period_start, period_end)
SELECT r.disclosure_id, c.id, d.disclosure_date - interval '3 months', d.disclosure_date
FROM public.reports r
JOIN public.disclosures d ON r.disclosure_id = d.id
JOIN public.companies c ON d.company_code = c.company_code;

-- =============================================================================
-- Step 4: Add New Constraints and Indexes
-- This section adds all constraints and indexes for the newly created tables and columns.
-- =============================================================================

-- Primary Keys for new tables
ALTER TABLE ONLY public.sectors ADD CONSTRAINT sectors_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.exchanges ADD CONSTRAINT exchanges_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.companies ADD CONSTRAINT companies_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.disclosure_categories ADD CONSTRAINT disclosure_categories_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.disclosure_subcategories ADD CONSTRAINT disclosure_subcategories_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.disclosure_labels ADD CONSTRAINT disclosure_labels_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.xbrl_filings ADD CONSTRAINT xbrl_filings_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.filing_sections ADD CONSTRAINT filing_sections_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.concepts ADD CONSTRAINT concepts_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.concept_overrides ADD CONSTRAINT label_overrides_pkey PRIMARY KEY (concept_id);
ALTER TABLE ONLY public.context_dims ADD CONSTRAINT context_dims_pkey PRIMARY KEY (context_id);
ALTER TABLE ONLY public.units ADD CONSTRAINT units_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.etl_runs ADD CONSTRAINT etl_runs_pkey PRIMARY KEY (id);

-- Unique Constraints for new tables
ALTER TABLE ONLY public.sectors ADD CONSTRAINT sectors_code_key UNIQUE (code);
ALTER TABLE ONLY public.exchanges ADD CONSTRAINT exchanges_mic_key UNIQUE (mic);
ALTER TABLE ONLY public.companies ADD CONSTRAINT companies_company_code_key UNIQUE (company_code);
ALTER TABLE ONLY public.disclosure_categories ADD CONSTRAINT disclosure_categories_name_key UNIQUE (name);
ALTER TABLE ONLY public.disclosure_subcategories ADD CONSTRAINT disclosure_subcategories_name_key UNIQUE (name);
ALTER TABLE ONLY public.disclosure_labels ADD CONSTRAINT disclosure_labels_disclosure_id_category_id_subcat_id_key UNIQUE (disclosure_id, category_id, subcat_id);
ALTER TABLE ONLY public.concepts ADD CONSTRAINT concepts_taxonomy_prefix_local_name_key UNIQUE (taxonomy_prefix, local_name);
ALTER TABLE ONLY public.units ADD CONSTRAINT units_unit_code_key UNIQUE (unit_code);
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_section_concept_context_unique UNIQUE (section_id, concept_id, context_id);

-- Foreign Keys for new tables
ALTER TABLE ONLY public.companies ADD CONSTRAINT companies_exchange_id_fkey FOREIGN KEY (exchange_id) REFERENCES public.exchanges(id);
ALTER TABLE ONLY public.companies ADD CONSTRAINT companies_sector_id_fkey FOREIGN KEY (sector_id) REFERENCES public.sectors(id);
ALTER TABLE ONLY public.disclosure_subcategories ADD CONSTRAINT disclosure_subcategories_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.disclosure_categories(id);
ALTER TABLE ONLY public.disclosure_labels ADD CONSTRAINT disclosure_labels_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.disclosure_categories(id);
ALTER TABLE ONLY public.disclosure_labels ADD CONSTRAINT disclosure_labels_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.disclosure_labels ADD CONSTRAINT disclosure_labels_subcat_id_fkey FOREIGN KEY (subcat_id) REFERENCES public.disclosure_subcategories(id);
ALTER TABLE ONLY public.xbrl_filings ADD CONSTRAINT xbrl_filings_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id);
ALTER TABLE ONLY public.xbrl_filings ADD CONSTRAINT xbrl_filings_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id);
ALTER TABLE ONLY public.xbrl_filings ADD CONSTRAINT xbrl_filings_parent_filing_id_fkey FOREIGN KEY (parent_filing_id) REFERENCES public.xbrl_filings(id);
ALTER TABLE ONLY public.filing_sections ADD CONSTRAINT filing_sections_filing_id_fkey FOREIGN KEY (filing_id) REFERENCES public.xbrl_filings(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.concept_overrides ADD CONSTRAINT label_overrides_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.concepts(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_context_id_fkey FOREIGN KEY (context_id) REFERENCES public.context_dims(context_id);
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.filing_sections(id);
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_unit_id_fkey FOREIGN KEY (unit_id) REFERENCES public.units(id);
ALTER TABLE ONLY public.financial_facts ADD CONSTRAINT financial_facts_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.concepts(id);

-- New Indexes
CREATE INDEX idx_disclosures_has_xbrl ON public.disclosures USING btree (has_xbrl);
CREATE INDEX idx_disclosures_has_xbrl_true ON public.disclosures USING btree (disclosure_date DESC) WHERE (has_xbrl = true);
CREATE INDEX idx_disclosure_labels_category_id ON public.disclosure_labels USING btree (category_id);
CREATE INDEX idx_disclosure_labels_disclosure_id ON public.disclosure_labels USING btree (disclosure_id);
CREATE INDEX idx_disclosure_labels_subcat_id ON public.disclosure_labels USING btree (subcat_id);
CREATE INDEX idx_disclosure_subcategories_category_id ON public.disclosure_subcategories USING btree (category_id);
CREATE INDEX idx_xbrl_filings_accounting_standard ON public.xbrl_filings USING btree (accounting_standard);
CREATE INDEX idx_xbrl_filings_company_period ON public.xbrl_filings USING btree (company_id, period_end);
CREATE INDEX idx_xbrl_filings_disclosure ON public.xbrl_filings USING btree (disclosure_id);
CREATE INDEX idx_xbrl_filings_industry_code ON public.xbrl_filings USING btree (industry_code);
CREATE INDEX idx_xbrl_filings_parent_filing_id ON public.xbrl_filings USING btree (parent_filing_id);
CREATE INDEX idx_xbrl_filings_period_quarter ON public.xbrl_filings USING btree (period_end, fiscal_quarter);
CREATE INDEX idx_xbrl_filings_period_type ON public.xbrl_filings USING btree (period_type);
CREATE UNIQUE INDEX filing_sections_path_uniq ON public.filing_sections USING btree (filing_id, rel_path);
CREATE UNIQUE INDEX filing_sections_unique_idx ON public.filing_sections USING btree (filing_id, statement_role_ja, period_prefix, consolidated, layout_code);
CREATE INDEX idx_filing_sections_filing ON public.filing_sections USING btree (filing_id);
CREATE INDEX idx_concepts_item_type ON public.concepts USING btree (item_type);
CREATE INDEX idx_concepts_local_name ON public.concepts USING btree (local_name);
CREATE INDEX idx_concepts_standard ON public.concepts USING btree (taxonomy_prefix);
CREATE INDEX idx_concepts_taxonomy_prefix_local_name ON public.concepts USING btree (taxonomy_prefix, local_name);
CREATE INDEX idx_context_dims_fiscal_span ON public.context_dims USING btree (fiscal_span);
CREATE INDEX idx_context_dims_period_base ON public.context_dims USING btree (period_base);
CREATE INDEX idx_financial_facts_concept_id ON public.financial_facts USING btree (concept_id);
CREATE INDEX idx_financial_facts_context_id ON public.financial_facts USING btree (context_id);

COMMIT;