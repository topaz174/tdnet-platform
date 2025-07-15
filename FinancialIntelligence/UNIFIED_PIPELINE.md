# Unified Extraction Pipeline

A comprehensive extraction pipeline that processes Japanese financial disclosures from the PostgreSQL database, intelligently selecting between XBRL and PDF sources with parallel processing capabilities.

## Overview

The unified extraction pipeline connects directly to the `tdnet.disclosures` table and processes all rows using intelligent file selection logic:

1. **XBRL First**: If `xbrl_path` exists and contains `qualitative.htm`, extract from XBRL
2. **PDF Fallback**: If XBRL is unavailable or lacks qualitative content, extract from PDF
3. **Parallel Processing**: Uses async/await patterns for high-performance extraction
4. **Test Mode**: Process limited rows for validation and testing

## Key Features

- ✅ **Database Integration**: Direct PostgreSQL connection to `disclosures` table
- ✅ **Intelligent File Selection**: XBRL preferred, PDF fallback
- ✅ **Parallel Processing**: Based on proven patterns from `parallel_pdf_extraction_pipeline.py`
- ✅ **XBRL Pipeline Enhancement**: Parallel processing for XBRL extraction
- ✅ **Test Mode**: Process recent days or limited rows for validation
- ✅ **Progress Monitoring**: Real-time progress tracking and ETA
- ✅ **Error Handling**: Robust error handling with detailed logging
- ✅ **Output Management**: Structured JSON output with metadata
- ✅ **Extraction Tracking**: Comprehensive status tracking and resume capability
- ✅ **Multi-Run Support**: Resume processing across multiple sessions
- ✅ **Status Management**: Tools for monitoring and managing extraction progress
- ✅ **Retry Logic**: Automatic retry of failed extractions

## Quick Start

### 1. Set Database Connection

The pipeline automatically loads database configuration from the `.env` file in the root directory. The `.env` file should contain:

```bash
# Option 1: Full connection string (recommended)
PG_DSN=postgresql://postgres:password@127.0.0.1:5432/tdnet

# Option 2: Individual components (automatically combined)
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_NAME=tdnet
DB_PORT=5432
```

**Note**: The pipeline will first try to use `PG_DSN` if available, otherwise it will build the connection string from individual components.

### 2. Setup Extraction Tracking (First Time Only)

```bash
# Setup tracking columns in database
python extraction_status_manager.py setup

# Or run the SQL migration directly
psql $PG_DSN -f add_extraction_tracking.sql
```

### 3. Test the Configuration

```bash
# Test .env file loading
python simple_env_test.py

# Test pipeline configuration and connection
python test_unified_pipeline.py

# Check extraction status
python extraction_status_manager.py status
```

### 4. Run the Pipeline

```bash
# Test mode: process last 7 days
python src/unified_extraction_pipeline.py --test-mode --test-days 7

# Resume mode: only process unprocessed files
python src/unified_extraction_pipeline.py --resume --test-mode --test-days 7

# Status report: check progress
python src/unified_extraction_pipeline.py --status-report
```

### 5. Run Full Pipeline

```bash
# Full pipeline with auto-detected workers
python src/unified_extraction_pipeline.py --full-pipeline

# Full pipeline with custom settings
python src/unified_extraction_pipeline.py --full-pipeline --workers 32 --concurrent-files 16
```

## Configuration Options

### Database Settings
- `--pg-dsn`: PostgreSQL connection string (overrides .env file)
- `--env-file`: Path to .env file (default: searches parent directories)

### Processing Settings
- `--workers`: Number of worker processes (default: auto-detected)
- `--concurrent-files`: Maximum concurrent files (default: 8)
- `--batch-size`: Batch size for processing (default: 4)

### Test Mode Settings
- `--test-mode`: Enable test mode with limited data
- `--test-days`: Number of recent days to process (default: 7)
- `--max-test-rows`: Maximum rows in test mode (default: 100)

### File Selection Settings
- `--prefer-pdf`: Prefer PDF over XBRL (default: prefer XBRL)
- `--allow-no-qualitative`: Allow XBRL files without qualitative.htm

### Output Settings
- `--output-dir`: Output directory (default: unified_output)
- `--no-save-chunks`: Don't save chunks to files (default: save chunks)

### Advanced Settings
- `--gpu-ocr`: Enable GPU-accelerated OCR
- `--disable-ocr`: Disable OCR completely
- `--memory-limit`: Memory limit in GB (default: 16)
- `--debug`: Enable debug logging

## Processing Logic

### File Selection Strategy

The pipeline implements intelligent file selection:

```python
def select_file_for_processing(row):
    if prefer_xbrl and xbrl_available and has_qualitative_htm:
        return xbrl_path, 'xbrl'
    elif pdf_available:
        return pdf_path, 'pdf'
    elif xbrl_available:
        return xbrl_path, 'xbrl'
    else:
        return '', 'none'
```

### Fallback Logic

If XBRL processing fails or lacks qualitative content:
1. Check if PDF is available
2. Automatically fallback to PDF processing
3. Continue with unified chunk output

## Extraction Tracking and Resume Functionality

### Database Tracking

The pipeline includes a comprehensive tracking system that maintains extraction status in the database:

**Tracking Columns:**
- `extraction_status`: Current status (pending, processing, completed, failed, retry)
- `extraction_method`: Method used (xbrl, pdf, pdf_fallback)
- `extraction_date`: When extraction completed
- `extraction_error`: Error message if failed
- `chunks_extracted`: Number of chunks extracted
- `extraction_duration`: Processing time in seconds
- `extraction_file_path`: Path to processed file
- `extraction_metadata`: Additional processing metadata

### Resume Operations

**Resume Processing:**
```bash
# Resume only unprocessed files
python src/unified_extraction_pipeline.py --resume

# Resume and retry failed files
python src/unified_extraction_pipeline.py --resume --retry-failed

# Force reprocess all files (ignore completed status)
python src/unified_extraction_pipeline.py --force-reprocess
```

**Status Management:**
```bash
# Show comprehensive status report
python extraction_status_manager.py status

# Reset failed files to retry
python extraction_status_manager.py reset-failed

# Reset stuck processing entries
python extraction_status_manager.py reset-stuck

# List failed files with errors
python extraction_status_manager.py list-failed
```

### Multi-Day Processing Strategy

For large-scale extraction spanning multiple days:

1. **Initial Run:**
   ```bash
   # Start with test mode
   python src/unified_extraction_pipeline.py --test-mode --test-days 30
   ```

2. **Production Runs:**
   ```bash
   # Full pipeline with tracking
   python src/unified_extraction_pipeline.py --full-pipeline --workers 32
   ```

3. **Resume After Interruption:**
   ```bash
   # Resume from where you left off
   python src/unified_extraction_pipeline.py --resume --workers 32
   ```

4. **Monitor Progress:**
   ```bash
   # Check status periodically
   python src/unified_extraction_pipeline.py --status-report
   python extraction_status_manager.py status
   ```

5. **Handle Failures:**
   ```bash
   # Retry failed extractions
   python src/unified_extraction_pipeline.py --resume --retry-failed
   ```

### Progress Monitoring

**Real-time Monitoring:**
- Processing rate (files/second)
- Chunk extraction rate (chunks/second)
- ETA estimation
- Current file being processed
- Status breakdown

**Status Reports:**
- Overall progress percentage
- Files by extraction method
- Recent activity (last 7 days)
- Error analysis
- Performance statistics

## Database Schema

The pipeline expects the following columns in the `disclosures` table:

```sql
CREATE TABLE disclosures (
    id INTEGER PRIMARY KEY,
    company_code VARCHAR(10),
    company_name TEXT,
    disclosure_date DATE,
    xbrl_path TEXT,                    -- Path to XBRL zip file
    pdf_path TEXT,                     -- Path to PDF file
    title TEXT,
    category VARCHAR(100),
    
    -- Extraction tracking columns (added by migration)
    extraction_status VARCHAR(20) DEFAULT 'pending',
    extraction_method VARCHAR(20),
    extraction_date TIMESTAMP,
    extraction_error TEXT,
    chunks_extracted INTEGER DEFAULT 0,
    extraction_duration FLOAT DEFAULT 0.0,
    extraction_file_path TEXT,
    extraction_metadata JSONB
);

-- Indexes for efficient tracking queries
CREATE INDEX disclosures_extraction_status_idx ON disclosures(extraction_status);
CREATE INDEX disclosures_extraction_date_idx ON disclosures(extraction_date DESC);
CREATE INDEX disclosures_status_date_idx ON disclosures(extraction_status, disclosure_date DESC);
```

## Output Format

### Summary File
- `unified_extraction_summary_YYYYMMDD_HHMMSS.json`

```json
{
  "processing_stats": {
    "total_files": 100,
    "processed_files": 95,
    "failed_files": 5,
    "total_chunks": 2450,
    "total_processing_time": 120.5,
    "files_per_second": 0.79,
    "chunks_per_second": 20.3
  },
  "extraction_method_breakdown": {
    "xbrl": 60,
    "pdf": 30,
    "pdf_fallback": 5,
    "failed": 5
  },
  "failed_rows": [...],
  "results": [...]
}
```

### Individual Chunk Files
- `{company_code}_{disclosure_id}_{method}_chunks.json`

```json
{
  "disclosure_info": {
    "id": 12345,
    "company_code": "7203",
    "company_name": "トヨタ自動車株式会社",
    "disclosure_date": "2024-05-15",
    "extraction_method": "xbrl"
  },
  "chunks": [...],
  "total_chunks": 25,
  "processed_at": "2024-07-03T10:30:00"
}
```

## Performance Optimization

### Recommended Settings

For different system configurations:

**Development/Testing (8-16 cores)**:
```bash
python src/unified_extraction_pipeline.py --test-mode --workers 8 --concurrent-files 4
```

**Production (32+ cores)**:
```bash
python src/unified_extraction_pipeline.py --full-pipeline --workers 32 --concurrent-files 16
```

**Memory-constrained systems**:
```bash
python src/unified_extraction_pipeline.py --workers 8 --memory-limit 8 --disable-ocr
```

### Performance Monitoring

The pipeline provides real-time monitoring:
- Processing rate (rows/second)
- Chunk extraction rate (chunks/second)
- ETA estimation
- Current file being processed
- Memory usage tracking

## Error Handling

### Common Issues and Solutions

**Database Connection Failed**:
```bash
# Check .env file format
cat .env
# Should contain:
PG_DSN=postgresql://username:password@localhost:5432/tdnet
# OR individual components:
DB_USER=username
DB_PASSWORD=password
DB_HOST=localhost
DB_NAME=tdnet
DB_PORT=5432
```

**No Rows Found**:
- Check if `disclosures` table exists
- Verify `xbrl_path` and `pdf_path` columns have data
- Try increasing `--test-days` in test mode

**File Not Found Errors**:
- XBRL/PDF paths in database may be incorrect
- Check file system permissions
- Verify file paths are absolute

**Memory Issues**:
- Reduce `--workers` count
- Lower `--concurrent-files`
- Set `--memory-limit`
- Enable `--disable-ocr`

## Integration with Existing System

The unified pipeline is designed to work seamlessly with the existing financial intelligence system:

### Embeddings Integration
After extraction, run embeddings on the chunks:
```bash
python embed_disclosures_robust.py
```

### Agent Integration
Use with the enhanced agent system:
```bash
python enhanced_agent.py
```

### Database Schema Compatibility
The pipeline works with existing schema and can coexist with:
- `enhanced_retrieval_system.py`
- `complete_agent_framework.py`
- Existing embedding workflows

## Extending the Pipeline

### Adding New Extraction Methods

To add support for additional file formats:

1. Create a new processor class
2. Add to `FileSelectionStrategy`
3. Update `process_disclosure_row` method
4. Add corresponding configuration options

### Custom Filtering Logic

Modify `DatabaseManager.get_disclosure_rows()` to add custom filtering:
- Company-specific processing
- Date range filtering
- Category-based selection
- Custom SQL queries

## Monitoring and Logging

### Log Levels
- `INFO`: General progress and statistics
- `DEBUG`: Detailed processing information
- `ERROR`: Processing failures and issues

### Progress Tracking
Real-time progress updates include:
- Rows processed vs. total
- Current processing rate
- Estimated time remaining
- Current company being processed

## Best Practices

1. **Start with Test Mode**: Always test with a small dataset first
2. **Monitor Resources**: Watch CPU and memory usage during processing
3. **Backup Database**: Ensure database backups before large batch processing
4. **Check File Paths**: Verify file paths in database are accessible
5. **Staged Processing**: Process in batches for very large datasets
6. **Error Review**: Review failed extractions and adjust parameters

## Support and Troubleshooting

For issues or questions:
1. Check the logs for detailed error messages
2. Run in debug mode: `--debug`
3. Test with smaller datasets: `--test-mode --max-test-rows 10`
4. Verify database connectivity: `python test_unified_pipeline.py`
5. Check file system permissions and paths