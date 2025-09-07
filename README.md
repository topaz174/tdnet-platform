# TDnet Financial Intelligence Platform

A comprehensive platform for processing TDnet disclosures, combining quantitative data extraction from XBRLs with qualitative analysis of PDFs and text content.

## Architecture

This platform consists of several modular components:

### Core Modules

- **Scraper** (`src/scraper/`): TDnet data collection and downloading
- **Quantitative** (`src/quantitative/`): XBRL parsing and numeric data extraction
- **Qualitative** (`src/qualitative/`): Text extraction and narrative analysis
- **Classifier** (`src/classifier/`): Disclosure categorization and classification
- **Database** (`src/database/`): Schema management, migrations, and utilities
- **Analytics** (`src/analytics/`): Advanced analytics, AI agents, and retrieval systems

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure database and directories in `config/`

## Usage

### Scraping TDnet Data
```bash
python -m src.scraper.tdnet_search.scraper date 2024-01-15
```

### Processing XBRL Data
```bash
python -m src.quantitative.etl.load_filings
```

### Extracting Qualitative Data
```bash
python -m src.qualitative.pipelines.unified_pipeline
```

### Running Classification
```bash
python -m src.classifier.rules.classifier
```

## Database Management

The platform uses PostgreSQL with a comprehensive migration system:

```bash
# Initialize database
python -m src.database.utils.init_db

# Run migrations
python -m src.database.utils.migrate

# Create backup
python -m src.database.utils.backup
```

## Development

This platform is designed for modularity and scalability. Each component can be developed and tested independently.

## Contributing

When adding new functionality:
1. Follow the modular structure
2. Add appropriate tests
3. Update documentation
4. Follow the existing import patterns
