# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Architecture

This is an enhanced financial intelligence system optimized for analyzing Japanese corporate disclosures. The system uses an intelligent retrieval approach with dense embeddings, smart filtering, and multi-stage query processing, backed by PostgreSQL with pgvector.

### Core Components (Enhanced System)

1. **Enhanced Retrieval System** (`enhanced_retrieval_system.py`):
   - **RECOMMENDED**: Production-optimized financial document retrieval
   - Smart query classification and expansion with financial term knowledge base
   - Multi-stage retrieval: filter → rank → rerank with business logic
   - Uses existing 1536-dim embeddings with multilingual-e5-large model
   - Intelligent filtering based on classifications, dates, and companies
   - ~70MB storage (vs 2.5GB for the complex ColBERT approach)

2. **Enhanced Agent Framework** (`enhanced_agent.py`):
   - **RECOMMENDED**: Advanced LangGraph-based agent with 5-stage processing
   - Query analysis and classification (temporal, comparative, analytical)
   - Multi-stage search with refinement and context enrichment
   - Financial pattern recognition and company analysis
   - Comprehensive reasoning trails for transparency

3. **Legacy Systems** (for reference):
   - `agent.py`: Basic agent implementation
   - `retrieval_system.py`: Simple dense retrieval
   - `retrieval_system_new.py`: Complex ColBERT system (not recommended due to storage overhead)
   - `complete_agent_framework.py`: Advanced agent with financial calculations

### Database Schema

The system uses PostgreSQL with the core table:
- `disclosures`: Main table with 1536-dim dense embeddings, company metadata, classifications
- Required extensions: pgvector for similarity search with IVFFlat indexing
- Optional columns: `classification_l1`, `classification_l2` for enhanced filtering

## Development Commands

### Running the Enhanced System (Recommended)

**Enhanced agent (primary interface):**
```bash
python enhanced_agent.py
```

**Test and migrate to enhanced system:**
```bash
python migrate_to_enhanced_system.py
```

### Legacy Commands

**Basic agent:**
```bash
python agent.py
```

**Complex agent framework:**
```bash
python complete_agent_framework.py
```

### Database Setup

**For enhanced system (minimal changes):**
```bash
python migrate_to_enhanced_system.py  # Includes index creation
```

**For legacy complex system (not recommended):**
```bash
psql -f migrate_to_new_system.sql
```

### Environment Setup

Required environment variables:
- `PG_DSN`: PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/tdnet")
- `OPENAI_API_KEY`: For LLM operations

Optional environment variables:
- `REDIS_URL`: Redis connection URL for caching (defaults to disabled if not provided)

## Key Design Patterns

### Enhanced Retrieval Strategy
- **Query Classification**: Automatically detects temporal, comparative, analytical, or simple queries
- **Financial Knowledge Base**: Expands queries with Japanese-English financial term mappings
- **Smart Filtering**: Uses existing classification columns and date ranges intelligently
- **Multi-stage Processing**: Filter candidates → Business logic reranking → Context enrichment
- **Efficient Storage**: Uses existing embeddings, no additional storage overhead

### Agent Workflow (Enhanced)
1. **Query Analysis**: Classify query type and extract financial entities
2. **Initial Search**: Broad search with smart filtering
3. **Refine Search**: Additional searches for comparative queries or temporal diversity
4. **Analyze Results**: Company and temporal pattern analysis
5. **Compose Answer**: Comprehensive financial analysis with evidence citations

### Financial Term Knowledge
The system includes built-in mappings for:
- Earnings terms: 決算, 業績, 売上, 利益, etc.
- Corporate actions: 配当, 合併, M&A, 株式分割, etc.
- Temporal indicators: 四半期, 通期, 最近, 前年同期, etc.
- Classification patterns for automatic filtering

## File Relationships and Recommendations

### **USE THESE (Enhanced System):**
- `enhanced_retrieval_system.py`: Main retrieval engine with smart filtering
- `enhanced_agent.py`: Primary agent interface with multi-stage reasoning
- `migrate_to_enhanced_system.py`: Migration and testing utility

### **REFERENCE ONLY (Legacy):**
- `agent.py`: Basic implementation for comparison
- `retrieval_system.py`: Simple dense retrieval baseline
- `retrieval_system_new.py`: Complex ColBERT system (storage-heavy, not recommended)
- `complete_agent_framework.py`: Advanced calculations (can be integrated if needed)

### **AVOID:**
- `embed_disclosures_new_system_*.py`: Complex embedding scripts with storage overhead
- Complex ColBERT implementations due to 2.5GB+ storage requirements