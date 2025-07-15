# TDnet Database Scripts

This directory contains database management scripts for the TDnet Financial Intelligence Platform.

## Reference Tables Creation

### Overview

The reference tables serve as the baseline schema for the TDnet platform, representing the target structure that integrates both quantitative (XBRL) and qualitative (PDF/text) data processing capabilities.

### Tables Created

1. **`company_master`** - Master company information table
   - Company metadata, securities codes, sector information
   - Primary key: `securities_code`

2. **`disclosures`** - Main disclosures table with vector embeddings
   - Core disclosure metadata, extraction status tracking
   - Vector embeddings for semantic search
   - Primary key: `id` (auto-incrementing)

3. **`document_chunks`** - Text chunks from documents with embeddings
   - Chunked text content from PDFs and XBRLs
   - Vector embeddings for retrieval
   - Foreign key reference to `disclosures`

4. **`reports`** - Referenced table for foreign key constraints
   - Placeholder table to maintain referential integrity

### Usage

#### SQL Script (Direct)

```bash
# Execute directly in PostgreSQL
psql -d tdnet -f scripts/create_reference_tables.sql
```

#### Python Script (Recommended)

```bash
# Using environment variable for database URL
export DATABASE_URL="postgresql://user:password@localhost/tdnet"
python scripts/create_reference_tables.py

# Or specify database URL directly
python scripts/create_reference_tables.py --database-url "postgresql://user:password@localhost/tdnet"

# Force creation (skip existing tables check)
python scripts/create_reference_tables.py --force
```

### Prerequisites

#### Required PostgreSQL Extensions

- `vector` - For pgvector support (embeddings)
- `uuid-ossp` - For UUID generation

#### Python Dependencies

```bash
pip install psycopg2-binary
```

### Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `TDNET_DB_URL` - Alternative database URL variable

### Features

#### Comprehensive Indexing

The script creates optimized indexes for:
- Vector similarity search (HNSW and IVFFlat)
- Text search (GIN indexes)
- Date-based queries
- Status and category filtering

#### Data Integrity

- Foreign key constraints between tables
- Unique constraints for data consistency
- Triggers for automatic timestamp updates

#### Vector Search Ready

- pgvector extension with 1024-dimension embeddings
- Multiple index types for different query patterns
- Optimized for similarity search operations

### Verification

After running the script, verify the tables were created:

```sql
-- List all tables
\dt

-- Check table structure
\d company_master
\d disclosures
\d document_chunks

-- Verify indexes
\di
```

### Next Steps

1. **Data Migration**: Plan migration from existing schemas
2. **Data Population**: Populate `company_master` with reference data
3. **Testing**: Validate with sample data
4. **Production**: Deploy to production environment

### Logging

The Python script creates logs in the `logs/` directory:
- `logs/create_reference_tables.log` - Execution log

### Error Handling

The scripts include comprehensive error handling:
- Database connection validation
- Existing table detection
- Transaction rollback on failure
- Detailed error logging

## Notes

- These tables represent the **target schema** for database consolidation
- The structure accommodates both XBRL quantitative data and PDF qualitative data
- Vector embeddings enable advanced semantic search capabilities
- Designed for high-performance financial data processing 