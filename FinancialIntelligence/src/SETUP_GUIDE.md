# Setup Guide: From Raw Data to Working Agent

This guide walks you through setting up the complete data pipeline and connecting it to the Advanced Hybrid Financial Intelligence Agent.

## Prerequisites

### System Requirements
- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- 8GB+ RAM for processing large document collections
- 50GB+ disk space for document storage and database

### Current Data Status
✅ **You already have:**
- PostgreSQL database with `disclosures` table
- 10 years of PDF and XBRL files in local directories
- Pre-computed 1024-dimensional embeddings
- Proper indexes for efficient querying

## Phase 1: Environment Setup (30 minutes)

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
# MeCab for Japanese text processing
sudo apt-get update
sudo apt-get install mecab mecab-ipadic mecab-ipadic-utf8 libmecab-dev

# Development tools
sudo apt-get install build-essential python3-dev
```

**macOS:**
```bash
# Using Homebrew
brew install mecab mecab-ipadic

# Development tools (if needed)
xcode-select --install
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv financial_intelligence_env
source financial_intelligence_env/bin/activate  # Linux/macOS
# or
financial_intelligence_env\Scripts\activate     # Windows

# Install core dependencies
pip install -r requirements_extraction.txt

# Install Arelle for XBRL processing
pip install arelle-release

# Verify MeCab installation
python -c "import MeCab; print('MeCab OK')"
```

### 3. Verify PostgreSQL Setup

```sql
-- Connect to your database and verify extensions
\c your_database_name

-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Verify your disclosures table
\d disclosures

-- Check data counts
SELECT 
    COUNT(*) as total_disclosures,
    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings,
    COUNT(CASE WHEN xbrl_path IS NOT NULL THEN 1 END) as with_xbrl,
    COUNT(CASE WHEN pdf_path IS NOT NULL THEN 1 END) as with_pdf
FROM disclosures;
```

## Phase 2: Database Schema Enhancement (15 minutes)

### 1. Run Schema Setup

```bash
# Setup enhanced database schema
python extraction_pipeline.py \
    --connection-string "postgresql://username:password@localhost/your_database" \
    --setup-schema
```

This creates the additional tables needed by the agent:
- `company_master` - Enhanced company information
- `financial_metrics` - Structured financial data from XBRLs
- `financial_statements` - Full XBRL statement data
- `document_chunks` - Enhanced text chunks with better categorization
- `business_events` - Extracted business events and insights

### 2. Verify Schema Creation

```sql
-- Check new tables were created
\dt

-- Verify company_master population
SELECT COUNT(*) FROM company_master;
SELECT ticker, company_name, sector FROM company_master LIMIT 10;
```

## Phase 3: Data Extraction Pipeline (2-4 hours)

### Option A: Full Pipeline (Recommended for first run)

```bash
# Process all data (will take 2-4 hours depending on data size)
python extraction_pipeline.py \
    --connection-string "postgresql://username:password@localhost/your_database" \
    --full-pipeline
```

### Option B: Step-by-Step Processing

```bash
# Step 1: Process XBRL files (1-2 hours)
python extraction_pipeline.py \
    --connection-string "postgresql://username:password@localhost/your_database" \
    --process-xbrl \
    --batch-size 50

# Step 2: Enhanced PDF processing (1-2 hours)
python extraction_pipeline.py \
    --connection-string "postgresql://username:password@localhost/your_database" \
    --process-pdf \
    --batch-size 25
```

### Monitor Progress

```sql
-- Check extraction progress
SELECT 
    'financial_metrics' as table_name,
    COUNT(*) as records
FROM financial_metrics
UNION ALL
SELECT 
    'document_chunks' as table_name,
    COUNT(*) as records  
FROM document_chunks;

-- Sample extracted financial metrics
SELECT 
    cm.company_name,
    fm.metric_name,
    fm.metric_value_jpy,
    fm.period_end_date
FROM financial_metrics fm
JOIN company_master cm ON fm.ticker = cm.ticker
ORDER BY fm.period_end_date DESC
LIMIT 10;
```

## Phase 4: Agent Integration (30 minutes)

### 1. Create Agent Configuration

```python
# Create config.py
DATABASE_CONFIG = {
    "connection_string": "postgresql://username:password@localhost/your_database"
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-3-large",  # or your preferred model
    "dimensions": 1024  # Match your existing embeddings
}

AGENT_CONFIG = {
    "max_companies_per_query": 5,
    "max_document_chunks": 20,
    "confidence_threshold": 0.7
}
```

### 2. Create Vector Retriever for Your Data

```python
# Create enhanced_vector_retriever.py
from langchain.schema import BaseRetriever, Document
import psycopg2
import numpy as np
from typing import List

class PostgreSQLVectorRetriever(BaseRetriever):
    def __init__(self, connection_string: str, embedding_model):
        self.connection_string = connection_string
        self.embedding_model = embedding_model
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                # Use document chunks for better precision
                cur.execute("""
                    SELECT 
                        dc.content,
                        dc.content_type,
                        d.title,
                        d.company_name,
                        d.disclosure_date,
                        cm.sector,
                        1 - (dc.embedding <=> %s::vector) as similarity
                    FROM document_chunks dc
                    JOIN disclosures d ON dc.disclosure_id = d.id
                    LEFT JOIN company_master cm ON d.company_code = cm.ticker
                    WHERE 1 - (dc.embedding <=> %s::vector) > 0.7
                    ORDER BY similarity DESC
                    LIMIT 20
                """, (query_embedding.tolist(), query_embedding.tolist()))
                
                results = cur.fetchall()
                
                documents = []
                for row in results:
                    content, content_type, title, company, date, sector, similarity = row
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "title": title,
                            "company_name": company,
                            "disclosure_date": str(date),
                            "sector": sector,
                            "content_type": content_type,
                            "similarity": similarity
                        }
                    )
                    documents.append(doc)
                
                return documents
        finally:
            conn.close()
```

### 3. Initialize and Test Agent

```python
# Create test_agent.py
import asyncio
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI  # or your preferred LLM

from advanced_hybrid_agent import ProductionHybridFinancialAgent
from enhanced_vector_retriever import PostgreSQLVectorRetriever
from config import DATABASE_CONFIG, EMBEDDING_CONFIG

async def test_agent():
    # Initialize components
    sql_db = SQLDatabase.from_uri(DATABASE_CONFIG["connection_string"])
    
    # You'll need to initialize your embedding model here
    # embedding_model = YourEmbeddingModel()
    vector_retriever = PostgreSQLVectorRetriever(
        DATABASE_CONFIG["connection_string"], 
        embedding_model
    )
    
    llm = OpenAI(temperature=0)  # Configure with your API key
    
    # Initialize agent
    agent = ProductionHybridFinancialAgent(sql_db, vector_retriever, llm)
    
    # Test queries
    test_queries = [
        "What was Toyota's revenue in 2023?",
        "Compare Honda and Nissan profitability trends",
        "What are Sony's main risk factors?",
        "Show me automotive sector performance"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = await agent.process_query(query)
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Adequacy: {result.adequacy_score:.2f}")
        print(f"Response: {result.synthesis[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_agent())
```

### 4. Run Initial Tests

```bash
# Test the agent
python test_agent.py
```

## Phase 5: Production Deployment (1 hour)

### 1. Create FastAPI Wrapper

```python
# Create api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from advanced_hybrid_agent import ProductionHybridFinancialAgent
# ... other imports

app = FastAPI(title="Financial Intelligence API")

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

@app.post("/api/v1/chat/query")
async def process_query(request: QueryRequest):
    try:
        # Initialize agent (consider caching this)
        agent = ProductionHybridFinancialAgent(sql_db, vector_retriever, llm)
        
        result = await agent.process_query(request.query)
        
        return {
            "query": result.query,
            "response": result.synthesis,
            "confidence": result.confidence,
            "adequacy_score": result.adequacy_score,
            "execution_time": result.execution_time,
            "companies": [c.name for c in result.intent.companies],
            "sources": result.sources[:5]  # Limit sources in response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Test API

```bash
# Start the API server
python api_server.py

# Test with curl
curl -X POST "http://localhost:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What was Toyota revenue in 2023?"}'
```

## Monitoring and Maintenance

### 1. Database Monitoring

```sql
-- Monitor extraction progress
SELECT 
    table_name,
    COUNT(*) as records,
    MAX(created_at) as last_update
FROM (
    SELECT 'financial_metrics' as table_name, created_at FROM financial_metrics
    UNION ALL
    SELECT 'document_chunks' as table_name, created_at FROM document_chunks
) t
GROUP BY table_name;

-- Check data quality
SELECT 
    fm.metric_name,
    COUNT(*) as count,
    AVG(fm.confidence_score) as avg_confidence
FROM financial_metrics fm
GROUP BY fm.metric_name
ORDER BY count DESC;
```

### 2. Performance Monitoring

```python
# Monitor agent performance
agent = ProductionHybridFinancialAgent(sql_db, vector_retriever, llm)
performance_report = agent.get_performance_report()
print(f"Average response time: {performance_report['metrics']['average_response_time']:.2f}s")
print(f"Average confidence: {performance_report['metrics']['average_confidence']:.2f}")
```

## Troubleshooting

### Common Issues

1. **MeCab Installation Fails**
   ```bash
   # Alternative installation
   pip install fugashi  # Alternative tokenizer
   ```

2. **XBRL Processing Errors**
   ```bash
   # Check Arelle installation
   python -c "from arelle import ModelManager; print('Arelle OK')"
   
   # Process smaller batches
   python extraction_pipeline.py --process-xbrl --batch-size 10
   ```

3. **Memory Issues During Processing**
   ```bash
   # Process in smaller batches
   python extraction_pipeline.py --process-pdf --batch-size 10
   ```

4. **Database Connection Issues**
   ```bash
   # Test connection
   psql "postgresql://username:password@localhost/database" -c "SELECT version();"
   ```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- Analyze tables for better query planning
   ANALYZE financial_metrics;
   ANALYZE document_chunks;
   ANALYZE disclosures;
   
   -- Check index usage
   SELECT schemaname, tablename, indexname, idx_scan 
   FROM pg_stat_user_indexes 
   ORDER BY idx_scan DESC;
   ```

2. **Agent Performance**
   - Cache frequently accessed company data
   - Use connection pooling for database access
   - Consider read replicas for analytics queries

## Next Steps

1. **Enhanced Features**
   - Add real-time document processing webhooks
   - Implement custom report generation
   - Add advanced visualization support

2. **Integration**
   - Connect to external market data APIs
   - Add news sentiment analysis
   - Implement automated alert systems

3. **Scaling**
   - Deploy with Docker containers
   - Add load balancing for multiple agent instances
   - Implement database sharding for very large datasets

Your financial intelligence platform is now ready for production use with comprehensive structured and unstructured data analysis capabilities!