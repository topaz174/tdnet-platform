# Data Pipeline Analysis & Implementation Strategy

## Current State Assessment

### ✅ **Strong Foundation Already in Place**

Your current setup is actually **excellent** and provides a solid foundation:

1. **Comprehensive Data Coverage**: 10 years of Japanese corporate disclosures
2. **Efficient Retrieval Setup**: PostgreSQL with pgvector and proper indexing
3. **Pre-computed Embeddings**: 1024-dimensional vectors already available
4. **Good Database Design**: Proper indexes for company, date, and category filtering
5. **File Organization**: Structured local storage with paths in database

### 📊 **Current Schema Analysis**

```sql
-- Your existing 'disclosures' table is well-designed:
disclosures (
    id,                    -- Primary key
    disclosure_date,       -- Perfect for temporal filtering
    company_code,          -- Essential for company identification
    company_name,          -- Human-readable company names
    title,                 -- Document titles for context
    pdf_path,             -- Local file access
    xbrl_path,            -- Structured data source
    category/subcategory, -- Document classification
    embedding vector(1024) -- Pre-computed semantic search
)
```

**Strengths:**
- ✅ Vector search capability (HNSW + IVFFlat indexes)
- ✅ Efficient company and date filtering
- ✅ Document metadata for context
- ✅ Links to both PDF and XBRL sources

## Recommended Architecture: PostgreSQL-Centric Approach

### 🎯 **Vector Database Decision: Stick with PostgreSQL**

**Recommendation: Do NOT add external vector databases (Qdrant/Weaviate/Pinecone)**

**Rationale:**
1. **Scale is Manageable**: 10 years of data fits well within PostgreSQL + pgvector capabilities
2. **Performance is Adequate**: HNSW indexes provide excellent vector search performance
3. **Simplicity**: Single database = simpler architecture, deployment, and maintenance
4. **ACID Guarantees**: Financial data benefits from transactional consistency
5. **Cost Efficiency**: No additional infrastructure or licensing costs
6. **Integration**: Your agent can query everything in one place

### 📋 **Required Database Schema Extensions**

Add these tables to support the agent's structured data processing:

```sql
-- 1. Company Master Table (for robust company identification)
CREATE TABLE company_master (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name TEXT NOT NULL,
    company_name_en TEXT,
    company_name_kana TEXT,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market VARCHAR(20) DEFAULT 'TSE',
    listing_date DATE,
    market_cap_jpy BIGINT,
    status VARCHAR(20) DEFAULT 'active',
    aliases TEXT[], -- Alternative company names
    keywords TEXT[], -- Search keywords
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Financial Metrics (extracted from XBRLs)
CREATE TABLE financial_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_id INTEGER REFERENCES disclosures(id),
    ticker VARCHAR(10),
    period_end_date DATE,
    period_type VARCHAR(20), -- 'annual', 'quarterly', 'interim'
    fiscal_year INTEGER,
    fiscal_period INTEGER, -- 1,2,3,4 for quarters
    metric_name VARCHAR(100), -- 'revenue', 'net_income', 'total_assets'
    metric_value_jpy BIGINT,
    metric_value_original BIGINT,
    original_currency VARCHAR(3) DEFAULT 'JPY',
    unit_scale VARCHAR(20), -- 'thousands', 'millions'
    calculation_method TEXT, -- How the metric was derived
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(disclosure_id, metric_name, period_end_date)
);

-- 3. Financial Statements (structured XBRL data)
CREATE TABLE financial_statements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_id INTEGER REFERENCES disclosures(id),
    statement_type VARCHAR(50), -- 'income', 'balance', 'cashflow'
    period_end_date DATE,
    period_type VARCHAR(20),
    fiscal_year INTEGER,
    fiscal_period INTEGER,
    currency VARCHAR(3) DEFAULT 'JPY',
    data JSONB, -- Full XBRL data structure
    line_items JSONB, -- Standardized line items
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Document Content (chunked text for better RAG)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_id INTEGER REFERENCES disclosures(id),
    chunk_index INTEGER,
    content TEXT,
    content_type VARCHAR(50), -- 'management_discussion', 'financial_summary', 'notes'
    page_number INTEGER,
    embedding vector(1024), -- Chunk-level embeddings for better precision
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Business Events (extracted insights)
CREATE TABLE business_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_id INTEGER REFERENCES disclosures(id),
    event_type VARCHAR(100), -- 'acquisition', 'management_change', 'restructuring'
    event_date DATE,
    description TEXT,
    impact_assessment TEXT,
    confidence_score FLOAT,
    extraction_method VARCHAR(50), -- 'llm', 'rule_based'
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX financial_metrics_ticker_date_idx ON financial_metrics(ticker, period_end_date DESC);
CREATE INDEX financial_metrics_metric_date_idx ON financial_metrics(metric_name, period_end_date DESC);
CREATE INDEX document_chunks_disclosure_idx ON document_chunks(disclosure_id);
CREATE INDEX document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX business_events_type_date_idx ON business_events(event_type, event_date DESC);
```

## Extraction Pipeline Architecture

### 🔄 **Three-Stage Processing Pipeline**

```python
# Stage 1: XBRL Processing (Structured Data Extraction)
class XBRLProcessor:
    """Extract structured financial data from XBRL files"""
    
    def process_xbrl_file(self, xbrl_path: str, disclosure_id: int):
        # 1. Parse XBRL using libraries like Arelle or py-xbrl
        # 2. Extract financial statements (Income, Balance, Cash Flow)
        # 3. Standardize to common metric names
        # 4. Store in financial_statements and financial_metrics tables
        
# Stage 2: PDF Text Processing (Enhanced Content Extraction)  
class PDFProcessor:
    """Process PDFs for better text chunking and content categorization"""
    
    def process_pdf_file(self, pdf_path: str, disclosure_id: int):
        # 1. Extract text with structure preservation
        # 2. Identify sections (MD&A, financial summary, notes)
        # 3. Chunk text semantically (not just by size)
        # 4. Generate embeddings for each chunk
        # 5. Store in document_chunks table
        
# Stage 3: Business Intelligence Extraction
class BusinessEventExtractor:
    """Extract business events and insights using LLM"""
    
    def extract_events(self, text_content: str, disclosure_id: int):
        # 1. Use LLM to identify significant business events
        # 2. Classify event types and assess impact
        # 3. Extract dates and quantitative impacts
        # 4. Store in business_events table
```

### 🛠 **Implementation Strategy**

#### **Phase 1: Database Schema Setup (Week 1)**
```sql
-- Add new tables to existing database
-- Populate company_master from existing disclosure data
INSERT INTO company_master (ticker, company_name, sector)
SELECT DISTINCT 
    company_code,
    company_name,
    CASE category WHEN 'earnings' THEN 'various' ELSE 'unknown' END
FROM disclosures;
```

#### **Phase 2: XBRL Processing Pipeline (Weeks 2-3)**
```python
# Key libraries for XBRL processing
pip install Arelle  # XBRL processing
pip install lxml   # XML parsing
pip install pandas # Data manipulation

# Processing workflow:
1. Scan xbrl_path column in disclosures table
2. For each XBRL file:
   - Parse financial statements
   - Extract standardized metrics (revenue, profit, assets, etc.)
   - Store in financial_metrics and financial_statements
   - Handle Japanese GAAP specifics
```

#### **Phase 3: Enhanced PDF Processing (Weeks 3-4)**
```python
# Enhanced text extraction and chunking
pip install pdfplumber  # Better PDF text extraction
pip install spacy       # Japanese text processing
pip install transformers # For embeddings

# Processing workflow:
1. Re-process existing PDFs with better chunking
2. Generate chunk-level embeddings (more precise than document-level)
3. Categorize content types (MD&A, financial data, notes)
4. Store in document_chunks table
```

#### **Phase 4: Agent Integration (Week 4-5)**
```python
# Modify agent to use new schema
class StructuredDataProcessor:
    def _get_company_data(self, company: Company, metrics: List[str], time_periods: List[str]):
        # Query financial_metrics table instead of generic structure
        query = """
        SELECT fm.ticker, fm.period_end_date, fm.metric_name, 
               fm.metric_value_jpy, cm.company_name
        FROM financial_metrics fm
        JOIN company_master cm ON fm.ticker = cm.ticker
        WHERE fm.ticker = %s
        """
        
class NarrativeAnalysisProcessor:
    def _retrieve_documents(self, query: str, company_filter: str):
        # Use document_chunks for more precise retrieval
        # Combine with original disclosures table for metadata
```

## Integration with Current Agent

### 🔗 **Minimal Changes Required**

The agent we built is designed to be flexible and will integrate seamlessly:

```python
# Agent initialization with your database
sql_db = SQLDatabase.from_uri("postgresql://user:pass@localhost/your_db")

# Your existing vector retriever can use either:
# Option 1: Keep using disclosures.embedding (document-level)
# Option 2: Upgrade to document_chunks.embedding (chunk-level, better precision)

class PostgreSQLVectorRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Option 1: Use existing disclosures table
        sql_query = """
        SELECT d.title, d.pdf_path, d.company_name, d.disclosure_date,
               1 - (d.embedding <=> %s::vector) as similarity
        FROM disclosures d
        WHERE 1 - (d.embedding <=> %s::vector) > 0.7
        ORDER BY similarity DESC
        LIMIT 10
        """
        
        # Option 2: Use new chunked approach (recommended)
        sql_query = """
        SELECT dc.content, d.title, d.company_name, d.disclosure_date,
               1 - (dc.embedding <=> %s::vector) as similarity
        FROM document_chunks dc
        JOIN disclosures d ON dc.disclosure_id = d.id
        WHERE 1 - (dc.embedding <=> %s::vector) > 0.7
        ORDER BY similarity DESC
        LIMIT 20
        """
```

## Performance and Scalability Considerations

### 📈 **Database Performance Optimization**

```sql
-- Additional performance indexes
CREATE INDEX CONCURRENTLY financial_metrics_composite_idx 
ON financial_metrics(ticker, metric_name, period_end_date DESC);

CREATE INDEX CONCURRENTLY document_chunks_type_embedding_idx 
ON document_chunks(content_type) INCLUDE (embedding);

-- Partitioning for large datasets
CREATE TABLE financial_metrics_y2024 PARTITION OF financial_metrics
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### 🚀 **Scaling Strategy**

1. **Current Scale**: PostgreSQL handles 10 years of data efficiently
2. **Growth Path**: Use table partitioning by year for large datasets
3. **Read Replicas**: Add read replicas for analytics workloads
4. **Connection Pooling**: Use PgBouncer for high-concurrency scenarios

## Cost-Benefit Analysis

### 💰 **PostgreSQL-Only Approach vs External Vector DB**

| Aspect | PostgreSQL + pgvector | External Vector DB |
|--------|----------------------|-------------------|
| **Infrastructure Cost** | ✅ Single database | ❌ Additional service |
| **Operational Complexity** | ✅ Simple | ❌ Multi-system management |
| **Data Consistency** | ✅ ACID guarantees | ❌ Eventual consistency |
| **Query Performance** | ✅ Good for your scale | ✅ Slightly better at massive scale |
| **Development Speed** | ✅ Single connection | ❌ Multiple integrations |
| **Backup/Recovery** | ✅ Single system | ❌ Multiple systems |

**Recommendation**: PostgreSQL-only approach is optimal for your use case.

## Implementation Timeline

### 📅 **4-Week Implementation Plan**

**Week 1: Foundation**
- Create new database tables
- Populate company_master table
- Set up XBRL processing environment

**Week 2: XBRL Processing**
- Implement XBRL parser
- Process historical XBRL files (batch job)
- Populate financial_metrics and financial_statements

**Week 3: Enhanced PDF Processing**
- Implement improved text chunking
- Generate chunk-level embeddings
- Populate document_chunks table

**Week 4: Agent Integration**
- Modify agent for new schema
- Implement enhanced retrieval
- Testing and optimization

### 🎯 **Success Metrics**

- **Data Coverage**: 95%+ of XBRLs successfully processed
- **Query Performance**: <500ms for financial metric queries
- **Retrieval Quality**: Improved relevance with chunk-level embeddings
- **Agent Compatibility**: Seamless integration with existing agent

## Conclusion

Your current setup provides an excellent foundation. The recommended approach is:

1. **Keep PostgreSQL + pgvector** (no external vector databases needed)
2. **Add structured tables** for XBRL data and enhanced text processing
3. **Implement 3-stage processing pipeline** for comprehensive data extraction
4. **Integrate with existing agent** through minimal schema adaptations

This approach maximizes the value of your existing infrastructure while providing the structured data access patterns needed by the agent we built.