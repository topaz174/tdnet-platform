--
-- PostgreSQL database dump
--

-- Dumped from database version 14.18 (Ubuntu 14.18-1.pgdg22.04+1)
-- Dumped by pg_dump version 14.18 (Ubuntu 14.18-0ubuntu0.22.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: alex
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO alex;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: company_master; Type: TABLE; Schema: public; Owner: alex
--

CREATE TABLE public.company_master (
    securities_code character varying(10) NOT NULL,
    ticker character varying(10),
    company_name_japanese text NOT NULL,
    company_name_english text,
    company_name_kana text,
    sector_japanese character varying(100),
    sector_english character varying(100),
    company_address text,
    corporate_number character varying(20),
    listing_classification character varying(50),
    consolidation_yes_no character varying(5),
    listing_date date,
    market_status character varying(50),
    aliases text[],
    keywords text[],
    fiscal_year_end character varying(10),
    edinet_code character varying(20),
    bloomberg_code character varying(20),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.company_master OWNER TO alex;

--
-- Name: disclosures_id_seq; Type: SEQUENCE; Schema: public; Owner: alex
--

CREATE SEQUENCE public.disclosures_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.disclosures_id_seq OWNER TO alex;

--
-- Name: disclosures; Type: TABLE; Schema: public; Owner: alex
--

CREATE TABLE public.disclosures (
    id integer DEFAULT nextval('public.disclosures_id_seq'::regclass) NOT NULL,
    disclosure_date date,
    "time" time without time zone,
    company_code character varying,
    company_name character varying,
    title text,
    xbrl_url text,
    pdf_path text,
    exchange character varying,
    update_history text,
    page_number integer,
    scraped_at timestamp without time zone,
    category text,
    subcategory text,
    embedding public.vector(1024),
    xbrl_path text,
    extraction_status character varying(20) DEFAULT 'pending'::character varying,
    extraction_method character varying(20),
    extraction_date timestamp without time zone,
    extraction_error text,
    chunks_extracted integer DEFAULT 0,
    extraction_duration double precision DEFAULT 0.0,
    extraction_file_path text,
    extraction_metadata jsonb
);


ALTER TABLE public.disclosures OWNER TO alex;

--
-- Name: document_chunks; Type: TABLE; Schema: public; Owner: alex
--

CREATE TABLE public.document_chunks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    disclosure_id integer NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    content_type character varying(50),
    section_code character varying(50),
    heading_text text,
    char_length integer,
    tokens integer,
    vectorize boolean DEFAULT true,
    is_numeric boolean DEFAULT false,
    disclosure_hash character varying(64),
    source_file text,
    page_number integer,
    metadata jsonb,
    embedding public.vector(1024),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.document_chunks OWNER TO alex;

--
-- Name: reports; Type: TABLE; Schema: public; Owner: alex
--

CREATE TABLE public.reports (
    id integer NOT NULL,
    disclosure_id integer
);


ALTER TABLE public.reports OWNER TO alex;

--
-- Name: company_master company_master_pkey; Type: CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.company_master
    ADD CONSTRAINT company_master_pkey PRIMARY KEY (securities_code);


--
-- Name: disclosures disclosures_pkey; Type: CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.disclosures
    ADD CONSTRAINT disclosures_pkey PRIMARY KEY (id);


--
-- Name: document_chunks document_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_pkey PRIMARY KEY (id);


--
-- Name: document_chunks document_chunks_unique_chunk_per_disclosure; Type: CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_unique_chunk_per_disclosure UNIQUE (disclosure_id, chunk_index);


--
-- Name: reports reports_pkey; Type: CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.reports
    ADD CONSTRAINT reports_pkey PRIMARY KEY (id);


--
-- Name: disclosures_category_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_category_idx ON public.disclosures USING btree (category);


--
-- Name: disclosures_company_code_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_company_code_idx ON public.disclosures USING btree (company_code);


--
-- Name: disclosures_company_date_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_company_date_idx ON public.disclosures USING btree (company_code, disclosure_date DESC);


--
-- Name: disclosures_date_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_date_idx ON public.disclosures USING btree (disclosure_date);


--
-- Name: disclosures_disclosure_date_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_disclosure_date_idx ON public.disclosures USING btree (disclosure_date DESC);


--
-- Name: disclosures_embedding_hnsw; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_embedding_hnsw ON public.disclosures USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: disclosures_embedding_ivfflat_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_embedding_ivfflat_idx ON public.disclosures USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: disclosures_embedding_null_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_embedding_null_idx ON public.disclosures USING btree (id) WHERE (embedding IS NULL);


--
-- Name: disclosures_extraction_date_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_extraction_date_idx ON public.disclosures USING btree (extraction_date DESC);


--
-- Name: disclosures_extraction_status_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_extraction_status_idx ON public.disclosures USING btree (extraction_status);


--
-- Name: disclosures_status_date_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_status_date_idx ON public.disclosures USING btree (extraction_status, disclosure_date DESC);


--
-- Name: disclosures_subcategory_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX disclosures_subcategory_idx ON public.disclosures USING btree (subcategory);


--
-- Name: document_chunks_chunk_index_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_chunk_index_idx ON public.document_chunks USING btree (disclosure_id, chunk_index);


--
-- Name: document_chunks_content_search_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_content_search_idx ON public.document_chunks USING gin (to_tsvector('english'::regconfig, content));


--
-- Name: document_chunks_content_type_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_content_type_idx ON public.document_chunks USING btree (content_type);


--
-- Name: document_chunks_disclosure_content_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_disclosure_content_idx ON public.document_chunks USING btree (disclosure_id, content_type, vectorize);


--
-- Name: document_chunks_disclosure_id_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_disclosure_id_idx ON public.document_chunks USING btree (disclosure_id);


--
-- Name: document_chunks_hash_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_hash_idx ON public.document_chunks USING btree (disclosure_hash);


--
-- Name: document_chunks_page_number_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_page_number_idx ON public.document_chunks USING btree (page_number) WHERE (page_number IS NOT NULL);


--
-- Name: document_chunks_section_code_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_section_code_idx ON public.document_chunks USING btree (section_code);


--
-- Name: document_chunks_vectorize_idx; Type: INDEX; Schema: public; Owner: alex
--

CREATE INDEX document_chunks_vectorize_idx ON public.document_chunks USING btree (vectorize) WHERE (vectorize = true);


--
-- Name: document_chunks update_document_chunks_updated_at; Type: TRIGGER; Schema: public; Owner: alex
--

CREATE TRIGGER update_document_chunks_updated_at BEFORE UPDATE ON public.document_chunks FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: document_chunks document_chunks_disclosure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.document_chunks
    ADD CONSTRAINT document_chunks_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id);


--
-- Name: reports reports_disclosure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alex
--

ALTER TABLE ONLY public.reports
    ADD CONSTRAINT reports_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES public.disclosures(id);


--
-- PostgreSQL database dump complete
--

