# Scripts

This directory contains database setup and ETL orchestration scripts.

## XBRL ETL Pipeline

### Quick Start

To load all XBRL filings and their facts:

```bash
python scripts/run_xbrl_etl_pipeline.py
```

### Options

- `--company CODE`: Process only filings for specific company code
- `--limit N`: Process at most N filings (deprecated, processes all available)
- `--batch-size N`: Batch size (currently unused, always processes 1 at a time)

### Examples

```bash
# Process all unprocessed filings (may take many hours for 100k+ filings)
python scripts/run_xbrl_etl_pipeline.py

# Process only Toyota (company code 7203)
python scripts/run_xbrl_etl_pipeline.py --company 7203

# Process all Sony filings
python scripts/run_xbrl_etl_pipeline.py --company 6758
```

### Pipeline Details

The orchestrator script:

1. **Identifies unprocessed disclosures**: Queries for disclosures with `has_xbrl = true` that don't have corresponding entries in `xbrl_filings` table
2. **Sequential processing**: Processes one filing at a time to ensure data consistency
3. **Two-phase loading**: 
   - Phase 1: Load filing metadata into `xbrl_filings` and `filing_sections` tables
   - Phase 2: Load all facts for that filing into `financial_facts` table
4. **Progress tracking**: Shows real-time progress, success rates, and ETA
5. **Error handling**: Failed filings don't block subsequent processing

### Output

The script provides:
- Real-time progress updates every 10 filings
- Processing rate (filings/hour)
- Success/failure statistics
- Detailed logging of errors to timestamped log files

### Log Files

Error logs are saved to:
- `logs/failed_xbrl_filings_YYYYMMDD_HHMMSS.log` (filing errors)
- `logs/failed_facts_loading_YYYYMMDD_HHMMSS.log` (facts loading errors)

## Individual ETL Scripts

You can also run the individual ETL scripts separately:

### Load XBRL Filings

```bash
python src/quantitative/etl/load_xbrl_filings.py
```

### Load Facts

```bash
python src/quantitative/etl/load_facts.py
```

## Database Setup

### Reference Tables

To create/populate reference tables (concepts, units, etc.):

```bash
python scripts/create_reference_tables.py
```

This is a one-time setup that should be run before the XBRL ETL pipeline. 