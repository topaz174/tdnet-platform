--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.4

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

--
-- Name: companies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.companies (
    id integer NOT NULL,
    company_code character varying(5) NOT NULL,
    name_en text NOT NULL,
    name_ja text,
    exchange_id integer,
    sector_id integer,
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.companies OWNER TO postgres;

--
-- Name: companies_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.companies_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.companies_id_seq OWNER TO postgres;

--
-- Name: companies_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.companies_id_seq OWNED BY public.companies.id;


--
-- Name: concept_overrides; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.concept_overrides (
    concept_id integer NOT NULL,
    override_label_en text,
    override_label_ja text
);


ALTER TABLE public.concept_overrides OWNER TO postgres;

--
-- Name: concepts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.concepts (
    id integer NOT NULL,
    taxonomy_prefix character varying(32) NOT NULL,
    local_name text NOT NULL,
    std_label_en text,
    std_label_ja text,
    item_type character varying(12),
    taxonomy_version date
);


ALTER TABLE public.concepts OWNER TO postgres;

--
-- Name: concepts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.concepts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.concepts_id_seq OWNER TO postgres;

--
-- Name: concepts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.concepts_id_seq OWNED BY public.concepts.id;


--
-- Name: context_dims; Type: TABLE; Schema: public; Owner: postgres
--

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


ALTER TABLE public.context_dims OWNER TO postgres;

--
-- Name: COLUMN context_dims.period_base; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.context_dims.period_base IS 'Simplified period base (e.g., Current, Next, Prior) - just the first word from original period token';


--
-- Name: COLUMN context_dims.fiscal_span; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.context_dims.fiscal_span IS 'Fiscal span: 0=YTD, 1=Q1, 2=Q2, 3=Q3, 99=Year';


--
-- Name: disclosure_categories; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.disclosure_categories (
    id integer NOT NULL,
    name text NOT NULL,
    name_jp text
);


ALTER TABLE public.disclosure_categories OWNER TO postgres;

--
-- Name: disclosure_categories_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.disclosure_categories_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.disclosure_categories_id_seq OWNER TO postgres;

--
-- Name: disclosure_categories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.disclosure_categories_id_seq OWNED BY public.disclosure_categories.id;


--
-- Name: disclosure_category_patterns; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.disclosure_category_patterns (
    id integer NOT NULL,
    category_id integer NOT NULL,
    regex_pattern text NOT NULL
);


ALTER TABLE public.disclosure_category_patterns OWNER TO postgres;

--
-- Name: disclosure_category_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.disclosure_category_patterns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.disclosure_category_patterns_id_seq OWNER TO postgres;

--
-- Name: disclosure_category_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.disclosure_category_patterns_id_seq OWNED BY public.disclosure_category_patterns.id;


--
-- Name: disclosure_labels; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.disclosure_labels (
    id integer NOT NULL,
    disclosure_id integer NOT NULL,
    category_id integer,
    subcat_id integer,
    labeled_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.disclosure_labels OWNER TO postgres;

--
-- Name: disclosure_labels_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.disclosure_labels_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.disclosure_labels_id_seq OWNER TO postgres;

--
-- Name: disclosure_labels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.disclosure_labels_id_seq OWNED BY public.disclosure_labels.id;


--
-- Name: disclosure_subcategories; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.disclosure_subcategories (
    id integer NOT NULL,
    category_id integer NOT NULL,
    name text NOT NULL,
    name_jp text
);


ALTER TABLE public.disclosure_subcategories OWNER TO postgres;

--
-- Name: disclosure_subcategories_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.disclosure_subcategories_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.disclosure_subcategories_id_seq OWNER TO postgres;

--
-- Name: disclosure_subcategories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.disclosure_subcategories_id_seq OWNED BY public.disclosure_subcategories.id;


--
-- Name: disclosures; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.disclosures (
    id integer NOT NULL,
    disclosure_date date,
    "time" time without time zone,
    company_code character varying,
    company_name character varying,
    title text,
    xbrl_url text,
    exchange character varying,
    update_history text,
    page_number integer,
    scraped_at timestamp without time zone,
    has_xbrl boolean DEFAULT false NOT NULL
);


ALTER TABLE public.disclosures OWNER TO postgres;

--
-- Name: disclosures_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.disclosures_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.disclosures_id_seq OWNER TO postgres;

--
-- Name: disclosures_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.disclosures_id_seq OWNED BY public.disclosures.id;


--
-- Name: etl_runs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.etl_runs (
    id integer NOT NULL,
    run_started_at timestamp without time zone DEFAULT now(),
    mapping_hash character(64) NOT NULL,
    files_parsed integer,
    rows_inserted integer
);


ALTER TABLE public.etl_runs OWNER TO postgres;

--
-- Name: etl_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.etl_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.etl_runs_id_seq OWNER TO postgres;

--
-- Name: etl_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.etl_runs_id_seq OWNED BY public.etl_runs.id;


--
-- Name: exchanges; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.exchanges (
    id integer NOT NULL,
    mic character varying(10) NOT NULL,
    name_en text NOT NULL,
    name_ja text
);


ALTER TABLE public.exchanges OWNER TO postgres;

--
-- Name: exchanges_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.exchanges_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.exchanges_id_seq OWNER TO postgres;

--
-- Name: exchanges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.exchanges_id_seq OWNED BY public.exchanges.id;


--
-- Name: filing_sections; Type: TABLE; Schema: public; Owner: postgres
--

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


ALTER TABLE public.filing_sections OWNER TO postgres;

--
-- Name: financial_facts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.financial_facts (
    id bigint NOT NULL,
    section_id integer NOT NULL,
    unit_id integer,
    value numeric(38,2) NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    concept_id integer NOT NULL,
    context_id text NOT NULL
);


ALTER TABLE public.financial_facts OWNER TO postgres;

--
-- Name: financial_facts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.financial_facts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.financial_facts_id_seq OWNER TO postgres;

--
-- Name: financial_facts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.financial_facts_id_seq OWNED BY public.financial_facts.id;


--
-- Name: mv_disclosures_classified; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres
--

CREATE MATERIALIZED VIEW public.mv_disclosures_classified AS
 SELECT d.id,
    d.disclosure_date,
    d."time",
    d.company_code,
    d.company_name,
    d.title,
    d.xbrl_url,
    d.has_xbrl,
    d.exchange,
    d.update_history,
    d.page_number,
    d.scraped_at,
    jsonb_agg(DISTINCT jsonb_build_object('en', dc.name, 'jp', dc.name_jp)) FILTER (WHERE (dc.name IS NOT NULL)) AS categories,
    jsonb_agg(DISTINCT jsonb_build_object('en', ds.name, 'jp', ds.name_jp)) FILTER (WHERE (ds.name IS NOT NULL)) AS subcategories,
    max(dl.labeled_at) AS classified_at
   FROM (((public.disclosures d
     LEFT JOIN public.disclosure_labels dl ON ((dl.disclosure_id = d.id)))
     LEFT JOIN public.disclosure_categories dc ON ((dc.id = dl.category_id)))
     LEFT JOIN public.disclosure_subcategories ds ON ((ds.id = dl.subcat_id)))
  GROUP BY d.id, d.disclosure_date, d."time", d.company_code, d.company_name, d.title, d.xbrl_url, d.has_xbrl, d.exchange, d.update_history, d.page_number, d.scraped_at
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.mv_disclosures_classified OWNER TO postgres;

--
-- Name: units; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.units (
    id integer NOT NULL,
    currency character(3) DEFAULT 'JPY'::bpchar NOT NULL,
    scale smallint NOT NULL,
    unit_code character varying(20),
    note text
);


ALTER TABLE public.units OWNER TO postgres;

--
-- Name: xbrl_filings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.xbrl_filings (
    id integer NOT NULL,
    company_id integer NOT NULL,
    disclosure_id integer NOT NULL,
    period_start date NOT NULL,
    period_end date NOT NULL,
    fiscal_year smallint GENERATED ALWAYS AS (EXTRACT(year FROM period_end)) STORED,
    created_at timestamp without time zone DEFAULT now(),
    accounting_standard character varying(10),
    has_consolidated boolean,
    industry_code character varying(10),
    period_type character varying(8),
    submission_no integer,
    amendment_flag boolean,
    report_amendment_flag boolean,
    xbrl_amendment_flag boolean,
    parent_filing_id integer,
    fiscal_quarter smallint
);


ALTER TABLE public.xbrl_filings OWNER TO postgres;

--
-- Name: COLUMN xbrl_filings.period_type; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.xbrl_filings.period_type IS 'Period type from XBRL filing (e.g., Q1, Q2, Q3, FY) - replaces fiscal_quarter';


--
-- Name: mv_flat_facts; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres
--

CREATE MATERIALIZED VIEW public.mv_flat_facts AS
 SELECT f.id AS fact_id,
    c.company_code AS ticker,
    c.name_en AS company_name,
    xf.period_start,
    xf.period_end,
    xf.fiscal_year,
    xf.fiscal_quarter,
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
    COALESCE(u.unit_code, 'JPY_Mil'::character varying) AS unit_code,
    COALESCE((u.scale)::integer, 6) AS scale_power,
    COALESCE(u.currency, 'JPY'::bpchar) AS currency,
    COALESCE(ctx.period_base, 'Current'::text) AS period_base,
    COALESCE(ctx.period_type, 'Duration'::text) AS period_type,
    ctx.fiscal_span,
    ctx.consolidated AS context_consolidated,
    ctx.forecast_variant,
    xf.disclosure_id,
    xf.accounting_standard,
    xf.has_consolidated AS filing_has_consolidated,
    xf.industry_code,
    f.context_id AS context_raw,
    fs.rel_path AS source_file_path,
    xf.amendment_flag,
    xf.parent_filing_id,
        CASE
            WHEN (fs.statement_role_en = ANY (ARRAY['ProfitLoss'::text, 'ComprehensiveIncome'::text, 'BalanceSheet'::text, 'ChangesInEquity'::text, 'CashFlows'::text, 'SegmentInformation'::text, 'Narrative'::text, 'Dividend forecast revision'::text, 'Earnings forecast revision'::text])) THEN 'Attachment'::text
            WHEN (fs.statement_role_en = 'Summary'::text) THEN 'Summary'::text
            ELSE 'Other'::text
        END AS source_section_type,
    xf.id AS filing_id,
    xf.company_id,
    (d.disclosure_date + d."time") AS filing_timestamp
   FROM (((((((public.financial_facts f
     JOIN public.filing_sections fs ON ((fs.id = f.section_id)))
     JOIN public.xbrl_filings xf ON ((xf.id = fs.filing_id)))
     JOIN public.disclosures d ON ((d.id = xf.disclosure_id)))
     JOIN public.companies c ON ((c.id = xf.company_id)))
     LEFT JOIN public.units u ON ((u.id = f.unit_id)))
     LEFT JOIN public.concepts con ON ((con.id = f.concept_id)))
     LEFT JOIN public.context_dims ctx ON ((ctx.context_id = f.context_id)))
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.mv_flat_facts OWNER TO postgres;

--
-- Name: report_sections_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.report_sections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.report_sections_id_seq OWNER TO postgres;

--
-- Name: report_sections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.report_sections_id_seq OWNED BY public.filing_sections.id;


--
-- Name: reports_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.reports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.reports_id_seq OWNER TO postgres;

--
-- Name: reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.reports_id_seq OWNED BY public.xbrl_filings.id;


--
-- Name: sectors; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sectors (
    id integer NOT NULL,
    code character varying(10) NOT NULL,
    name_en text NOT NULL,
    name_ja text
);


ALTER TABLE public.sectors OWNER TO postgres;

--
-- Name: sectors_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sectors_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sectors_id_seq OWNER TO postgres;

--
-- Name: sectors_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sectors_id_seq OWNED BY public.sectors.id;


--
-- Name: units_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.units_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.units_id_seq OWNER TO postgres;

--
-- Name: units_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.units_id_seq OWNED BY public.units.id;


--
-- Name: companies id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies ALTER COLUMN id SET DEFAULT nextval('public.companies_id_seq'::regclass);


--
-- Name: concepts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts ALTER COLUMN id SET DEFAULT nextval('public.concepts_id_seq'::regclass);


--
-- Name: disclosure_categories id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_categories ALTER COLUMN id SET DEFAULT nextval('public.disclosure_categories_id_seq'::regclass);


--
-- Name: disclosure_category_patterns id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_category_patterns ALTER COLUMN id SET DEFAULT nextval('public.disclosure_category_patterns_id_seq'::regclass);


--
-- Name: disclosure_labels id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels ALTER COLUMN id SET DEFAULT nextval('public.disclosure_labels_id_seq'::regclass);


--
-- Name: disclosure_subcategories id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_subcategories ALTER COLUMN id SET DEFAULT nextval('public.disclosure_subcategories_id_seq'::regclass);


--
-- Name: disclosures id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosures ALTER COLUMN id SET DEFAULT nextval('public.disclosures_id_seq'::regclass);


--
-- Name: etl_runs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etl_runs ALTER COLUMN id SET DEFAULT nextval('public.etl_runs_id_seq'::regclass);


--
-- Name: exchanges id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.exchanges ALTER COLUMN id SET DEFAULT nextval('public.exchanges_id_seq'::regclass);


--
-- Name: filing_sections id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.filing_sections ALTER COLUMN id SET DEFAULT nextval('public.report_sections_id_seq'::regclass);


--
-- Name: financial_facts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts ALTER COLUMN id SET DEFAULT nextval('public.financial_facts_id_seq'::regclass);


--
-- Name: sectors id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sectors ALTER COLUMN id SET DEFAULT nextval('public.sectors_id_seq'::regclass);


--
-- Name: units id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.units ALTER COLUMN id SET DEFAULT nextval('public.units_id_seq'::regclass);


--
-- Name: xbrl_filings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.xbrl_filings ALTER COLUMN id SET DEFAULT nextval('public.reports_id_seq'::regclass);


--
-- Name: companies companies_company_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_company_code_key UNIQUE (company_code);


--
-- Name: companies companies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (id);


--
-- Name: concepts concepts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_pkey PRIMARY KEY (id);


--
-- Name: concepts concepts_taxonomy_prefix_local_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_taxonomy_prefix_local_name_key UNIQUE (taxonomy_prefix, local_name);


--
-- Name: context_dims context_dims_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.context_dims
    ADD CONSTRAINT context_dims_pkey PRIMARY KEY (context_id);


--
-- Name: disclosure_categories disclosure_categories_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_categories
    ADD CONSTRAINT disclosure_categories_name_key UNIQUE (name);


--
-- Name: disclosure_categories disclosure_categories_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_categories
    ADD CONSTRAINT disclosure_categories_pkey PRIMARY KEY (id);


--
-- Name: disclosure_category_patterns disclosure_category_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_category_patterns
    ADD CONSTRAINT disclosure_category_patterns_pkey PRIMARY KEY (id);


--
-- Name: disclosure_labels disclosure_labels_disclosure_id_category_id_subcat_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels
    ADD CONSTRAINT disclosure_labels_disclosure_id_category_id_subcat_id_key UNIQUE (disclosure_id, category_id, subcat_id);


--
-- Name: disclosure_labels disclosure_labels_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels
    ADD CONSTRAINT disclosure_labels_pkey PRIMARY KEY (id);


--
-- Name: disclosure_subcategories disclosure_subcategories_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_subcategories
    ADD CONSTRAINT disclosure_subcategories_name_key UNIQUE (name);


--
-- Name: disclosure_subcategories disclosure_subcategories_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_subcategories
    ADD CONSTRAINT disclosure_subcategories_pkey PRIMARY KEY (id);


--
-- Name: disclosures disclosures_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosures
    ADD CONSTRAINT disclosures_pkey PRIMARY KEY (id);


--
-- Name: etl_runs etl_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.etl_runs
    ADD CONSTRAINT etl_runs_pkey PRIMARY KEY (id);


--
-- Name: exchanges exchanges_mic_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.exchanges
    ADD CONSTRAINT exchanges_mic_key UNIQUE (mic);


--
-- Name: exchanges exchanges_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.exchanges
    ADD CONSTRAINT exchanges_pkey PRIMARY KEY (id);


--
-- Name: financial_facts financial_facts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts
    ADD CONSTRAINT financial_facts_pkey PRIMARY KEY (id);


--
-- Name: financial_facts financial_facts_section_concept_context_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts
    ADD CONSTRAINT financial_facts_section_concept_context_unique UNIQUE (section_id, concept_id, context_id);


--
-- Name: concept_overrides label_overrides_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concept_overrides
    ADD CONSTRAINT label_overrides_pkey PRIMARY KEY (concept_id);


--
-- Name: filing_sections report_sections_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.filing_sections
    ADD CONSTRAINT report_sections_pkey PRIMARY KEY (id);


--
-- Name: xbrl_filings reports_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.xbrl_filings
    ADD CONSTRAINT reports_pkey PRIMARY KEY (id);


--
-- Name: sectors sectors_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sectors
    ADD CONSTRAINT sectors_code_key UNIQUE (code);


--
-- Name: sectors sectors_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sectors
    ADD CONSTRAINT sectors_pkey PRIMARY KEY (id);


--
-- Name: units units_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.units
    ADD CONSTRAINT units_pkey PRIMARY KEY (id);


--
-- Name: units units_unit_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.units
    ADD CONSTRAINT units_unit_code_key UNIQUE (unit_code);


--
-- Name: filing_sections_path_uniq; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX filing_sections_path_uniq ON public.filing_sections USING btree (filing_id, rel_path);


--
-- Name: filing_sections_unique_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX filing_sections_unique_idx ON public.filing_sections USING btree (filing_id, statement_role_ja, period_prefix, consolidated, layout_code);


--
-- Name: idx_concepts_item_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_concepts_item_type ON public.concepts USING btree (item_type);


--
-- Name: idx_concepts_local_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_concepts_local_name ON public.concepts USING btree (local_name);


--
-- Name: idx_concepts_standard; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_concepts_standard ON public.concepts USING btree (taxonomy_prefix);


--
-- Name: idx_concepts_taxonomy_prefix_local_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_concepts_taxonomy_prefix_local_name ON public.concepts USING btree (taxonomy_prefix, local_name);


--
-- Name: idx_context_dims_fiscal_span; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_context_dims_fiscal_span ON public.context_dims USING btree (fiscal_span);


--
-- Name: idx_context_dims_period_base; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_context_dims_period_base ON public.context_dims USING btree (period_base);


--
-- Name: idx_disclosure_labels_category_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosure_labels_category_id ON public.disclosure_labels USING btree (category_id);


--
-- Name: idx_disclosure_labels_disclosure_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosure_labels_disclosure_id ON public.disclosure_labels USING btree (disclosure_id);


--
-- Name: idx_disclosure_labels_subcat_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosure_labels_subcat_id ON public.disclosure_labels USING btree (subcat_id);


--
-- Name: idx_disclosure_subcategories_category_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosure_subcategories_category_id ON public.disclosure_subcategories USING btree (category_id);


--
-- Name: idx_disclosures_has_xbrl; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosures_has_xbrl ON public.disclosures USING btree (has_xbrl);


--
-- Name: idx_disclosures_has_xbrl_true; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_disclosures_has_xbrl_true ON public.disclosures USING btree (disclosure_date DESC) WHERE (has_xbrl = true);


--
-- Name: idx_filing_sections_filing; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_filing_sections_filing ON public.filing_sections USING btree (filing_id);


--
-- Name: idx_financial_facts_concept_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_financial_facts_concept_id ON public.financial_facts USING btree (concept_id);


--
-- Name: idx_financial_facts_context_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_financial_facts_context_id ON public.financial_facts USING btree (context_id);


--
-- Name: idx_xbrl_filings_accounting_standard; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_accounting_standard ON public.xbrl_filings USING btree (accounting_standard);


--
-- Name: idx_xbrl_filings_company_period; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_company_period ON public.xbrl_filings USING btree (company_id, period_end);


--
-- Name: idx_xbrl_filings_disclosure; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_disclosure ON public.xbrl_filings USING btree (disclosure_id);


--
-- Name: idx_xbrl_filings_industry_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_industry_code ON public.xbrl_filings USING btree (industry_code);


--
-- Name: idx_xbrl_filings_parent_filing_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_parent_filing_id ON public.xbrl_filings USING btree (parent_filing_id);


--
-- Name: idx_xbrl_filings_period_quarter; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_period_quarter ON public.xbrl_filings USING btree (period_end, fiscal_quarter);


--
-- Name: idx_xbrl_filings_period_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_xbrl_filings_period_type ON public.xbrl_filings USING btree (period_type);


--
-- Name: mv_disclosures_classified_classified_at_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_disclosures_classified_classified_at_idx ON public.mv_disclosures_classified USING btree (classified_at DESC);


--
-- Name: mv_disclosures_classified_company_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_disclosures_classified_company_idx ON public.mv_disclosures_classified USING btree (company_code, disclosure_date DESC);


--
-- Name: mv_disclosures_classified_date_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_disclosures_classified_date_idx ON public.mv_disclosures_classified USING btree (disclosure_date DESC);


--
-- Name: mv_flat_facts_amendment_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_amendment_idx ON public.mv_flat_facts USING btree (amendment_flag, parent_filing_id);


--
-- Name: mv_flat_facts_concept_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_concept_idx ON public.mv_flat_facts USING btree (taxonomy_prefix, concept_local_name);


--
-- Name: mv_flat_facts_filing_timestamp_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_filing_timestamp_idx ON public.mv_flat_facts USING btree (filing_timestamp);


--
-- Name: mv_flat_facts_period_base_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_period_base_idx ON public.mv_flat_facts USING btree (period_base, fiscal_span);


--
-- Name: mv_flat_facts_period_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_period_idx ON public.mv_flat_facts USING btree (period_end, fiscal_quarter);


--
-- Name: mv_flat_facts_section_type_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_section_type_idx ON public.mv_flat_facts USING btree (source_section_type, statement_role_en);


--
-- Name: mv_flat_facts_ticker_canonical_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX mv_flat_facts_ticker_canonical_idx ON public.mv_flat_facts USING btree (ticker, canonical_name_jp, period_end);


--
-- Name: companies companies_exchange_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_exchange_id_fkey FOREIGN KEY (exchange_id) REFERENCES public.exchanges(id);


--
-- Name: companies companies_sector_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_sector_id_fkey FOREIGN KEY (sector_id) REFERENCES public.sectors(id);


--
-- Name: disclosure_category_patterns disclosure_category_patterns_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_category_patterns
    ADD CONSTRAINT disclosure_category_patterns_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.disclosure_categories(id);


--
-- Name: disclosure_labels disclosure_labels_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels
    ADD CONSTRAINT disclosure_labels_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.disclosure_categories(id);


--
-- Name: disclosure_labels disclosure_labels_disclosure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels
    ADD CONSTRAINT disclosure_labels_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id) ON DELETE CASCADE;


--
-- Name: disclosure_labels disclosure_labels_subcat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_labels
    ADD CONSTRAINT disclosure_labels_subcat_id_fkey FOREIGN KEY (subcat_id) REFERENCES public.disclosure_subcategories(id);


--
-- Name: disclosure_subcategories disclosure_subcategories_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.disclosure_subcategories
    ADD CONSTRAINT disclosure_subcategories_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.disclosure_categories(id);


--
-- Name: financial_facts financial_facts_context_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts
    ADD CONSTRAINT financial_facts_context_id_fkey FOREIGN KEY (context_id) REFERENCES public.context_dims(context_id);


--
-- Name: financial_facts financial_facts_section_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts
    ADD CONSTRAINT financial_facts_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.filing_sections(id);


--
-- Name: financial_facts financial_facts_unit_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.financial_facts
    ADD CONSTRAINT financial_facts_unit_id_fkey FOREIGN KEY (unit_id) REFERENCES public.units(id);


--
-- Name: concept_overrides label_overrides_concept_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concept_overrides
    ADD CONSTRAINT label_overrides_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.concepts(id) ON DELETE CASCADE;


--
-- Name: filing_sections report_sections_report_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.filing_sections
    ADD CONSTRAINT report_sections_report_id_fkey FOREIGN KEY (filing_id) REFERENCES public.xbrl_filings(id) ON DELETE CASCADE;


--
-- Name: xbrl_filings reports_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.xbrl_filings
    ADD CONSTRAINT reports_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id);


--
-- Name: xbrl_filings reports_disclosure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.xbrl_filings
    ADD CONSTRAINT reports_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id);


--
-- Name: xbrl_filings xbrl_filings_parent_filing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.xbrl_filings
    ADD CONSTRAINT xbrl_filings_parent_filing_id_fkey FOREIGN KEY (parent_filing_id) REFERENCES public.xbrl_filings(id);


--
-- PostgreSQL database dump complete
--

