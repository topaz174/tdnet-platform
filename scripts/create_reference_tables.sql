-- TDnet Database Reference Tables Creation Script
-- This script creates the reference tables that will serve as the baseline for migrations
-- Tables: company_master, disclosures, document_chunks

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create function for updating timestamp columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- COMPANY_MASTER TABLE
-- =====================================================================

CREATE TABLE company_master (
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

-- Primary key for company_master
ALTER TABLE company_master ADD CONSTRAINT company_master_pkey PRIMARY KEY (securities_code);

-- =====================================================================
-- DISCLOSURES TABLE
-- =====================================================================

-- Create sequence for disclosures id
CREATE SEQUENCE disclosures_id_seq;

CREATE TABLE disclosures (
    id integer NOT NULL DEFAULT nextval('disclosures_id_seq'::regclass),
    disclosure_date date,
    time time without time zone,
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
    embedding vector(1024),
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

-- Primary key for disclosures
ALTER TABLE disclosures ADD CONSTRAINT disclosures_pkey PRIMARY KEY (id);

-- Indexes for disclosures table
CREATE INDEX disclosures_category_idx ON disclosures USING btree (category);
CREATE INDEX disclosures_company_code_idx ON disclosures USING btree (company_code);
CREATE INDEX disclosures_company_date_idx ON disclosures USING btree (company_code, disclosure_date DESC);
CREATE INDEX disclosures_date_idx ON disclosures USING btree (disclosure_date);
CREATE INDEX disclosures_disclosure_date_idx ON disclosures USING btree (disclosure_date DESC);
CREATE INDEX disclosures_embedding_hnsw ON disclosures USING hnsw (embedding vector_cosine_ops);
CREATE INDEX disclosures_embedding_ivfflat_idx ON disclosures USING ivfflat (embedding vector_cosine_ops) WITH (lists='100');
CREATE INDEX disclosures_embedding_null_idx ON disclosures USING btree (id) WHERE embedding IS NULL;
CREATE INDEX disclosures_extraction_date_idx ON disclosures USING btree (extraction_date DESC);
CREATE INDEX disclosures_extraction_status_idx ON disclosures USING btree (extraction_status);
CREATE INDEX disclosures_status_date_idx ON disclosures USING btree (extraction_status, disclosure_date DESC);
CREATE INDEX disclosures_subcategory_idx ON disclosures USING btree (subcategory);

-- =====================================================================
-- DOCUMENT_CHUNKS TABLE
-- =====================================================================

CREATE TABLE document_chunks (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
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
    embedding vector(1024),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

-- Primary key and unique constraints for document_chunks
ALTER TABLE document_chunks ADD CONSTRAINT document_chunks_pkey PRIMARY KEY (id);
ALTER TABLE document_chunks ADD CONSTRAINT document_chunks_unique_chunk_per_disclosure UNIQUE (disclosure_id, chunk_index);

-- Indexes for document_chunks table
CREATE INDEX document_chunks_chunk_index_idx ON document_chunks USING btree (disclosure_id, chunk_index);
CREATE INDEX document_chunks_content_search_idx ON document_chunks USING gin (to_tsvector('english'::regconfig, content));
CREATE INDEX document_chunks_content_type_idx ON document_chunks USING btree (content_type);
CREATE INDEX document_chunks_disclosure_content_idx ON document_chunks USING btree (disclosure_id, content_type, vectorize);
CREATE INDEX document_chunks_disclosure_id_idx ON document_chunks USING btree (disclosure_id);
CREATE INDEX document_chunks_hash_idx ON document_chunks USING btree (disclosure_hash);
CREATE INDEX document_chunks_page_number_idx ON document_chunks USING btree (page_number) WHERE page_number IS NOT NULL;
CREATE INDEX document_chunks_section_code_idx ON document_chunks USING btree (section_code);
CREATE INDEX document_chunks_vectorize_idx ON document_chunks USING btree (vectorize) WHERE vectorize = true;

-- Foreign key constraints
ALTER TABLE document_chunks ADD CONSTRAINT document_chunks_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES disclosures(id);

-- Trigger for updating updated_at column
CREATE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================
-- REPORTS TABLE (Referenced in foreign key)
-- =====================================================================
-- Note: The reports table is referenced by the disclosures table foreign key constraint
-- Adding a placeholder structure if it doesn't exist

CREATE TABLE IF NOT EXISTS reports (
    id integer PRIMARY KEY,
    disclosure_id integer,
    CONSTRAINT reports_disclosure_id_fkey FOREIGN KEY (disclosure_id) REFERENCES disclosures(id)
);

-- =====================================================================
-- COMPLETION MESSAGE
-- =====================================================================

-- Display creation summary
DO $$
BEGIN
    RAISE NOTICE 'Reference tables created successfully:';
    RAISE NOTICE '- company_master: Master company information table';
    RAISE NOTICE '- disclosures: Main disclosures table with vector embeddings';
    RAISE NOTICE '- document_chunks: Text chunks from documents with embeddings';
    RAISE NOTICE '- reports: Referenced table for foreign key constraint';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables are ready to serve as reference point for migrations.';
END $$; 