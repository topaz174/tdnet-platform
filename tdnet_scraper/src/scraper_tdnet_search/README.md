# TDnet Search Historical Scraper

A specialized scraper for retrieving historical financial disclosures from TDnet Search (`https://tdnet-search.appspot.com`).

## Features

- **Historical Data Access**: Scrapes data going back several years
- **Query-Based Search**: Uses query strings like `date<2025-4-01` or `title:決算短信`
- **Smart Pagination**: Automatically handles the 11-page limit by adjusting date queries
- **Duplicate Prevention**: Checks existing database entries to avoid re-downloading
- **PDF/XBRL Downloads**: Downloads both PDF documents and XBRL files when available
- **JPX Mirror URLs**: Uses direct JPX links for reliable PDF access

## Setup

1. **Install Dependencies**:
```bash
pip install requests beautifulsoup4 sqlalchemy psycopg2 python-dateutil
```

2. **Initialize Database**:
```bash
python src/scraper_tdnet_search/setup_tdnet_search.py
```

3. **Configure Directories**: Update `directories.json` with your preferred paths, adding these two values:

    "tdnet_search_pdf_directory": your_path_here,
    "tdnet_search_xbrl_directory": your_path_here

## Usage

### Command Line Options

**Default Historical Scrape** (from 1 month ago):
```bash
python src/scraper_tdnet_search/tdnet_search_scraper.py
```

**Historical from Specific Date**:
```bash
python src/scraper_tdnet_search/tdnet_search_scraper.py historical 2024-01-01
```

**Specific Date Range**:
```bash
python src/scraper_tdnet_search/tdnet_search_scraper.py range 2024-01-01 2024-01-31
```

### Alternative Launcher
Use the launcher script from the project root:
```bash
python scripts/run_tdnet_search_scraper.py [arguments]
```

## Database

- **Database Name**: `tdnet_search`
- **Table**: `disclosures` (identical schema to main TDnet scraper)
- **Configuration**: Uses credentials from `.env` file

## File Organization

- **PDFs**: Saved to `directories.json` configured location, organized by date
- **XBRL**: Saved alongside PDFs in separate directory structure
- **Naming**: `HH-MM_CompanyCode_Title.pdf/zip`

## Key Functions

- `scrape_tdnet_search_historical()`: Main historical scraping function
- `scrape_specific_date_range()`: Targeted date range scraping
- `scrape_tdnet_search_page()`: Single page processing
- `download_file()`: File download with authentication handling

## Technical Details

### Pagination Strategy
The scraper handles TDnet Search's 11-page limit by:
1. Starting with a date query (e.g., `date<2025-04-01`)
2. Processing up to 11 pages
3. Finding the earliest date on the last page
4. Creating a new query starting from that date
5. Repeating until no more data is found

### Authentication
Uses Google authentication via `google_auth.py` for accessing restricted content.

### Error Handling
- Graceful handling of network timeouts
- Duplicate detection and skipping
- Invalid file detection and cleanup
- Comprehensive logging of all operations

## Utilities

- `utils.py`: Database statistics and management functions
- `setup_tdnet_search.py`: Automated environment setup
- `reset_tdnet_search.py`: Database reset functionality

## Dependencies

- `requests`: HTTP client
- `beautifulsoup4`: HTML parsing
- `sqlalchemy`: Database ORM
- `psycopg2`: PostgreSQL driver  
- `python-dateutil`: Date manipulation 