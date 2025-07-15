# TDnet Scraper

A Python-based scraper for collecting disclosure documents from TDnet Search.

## Features

- Scrape disclosures from TDnet Search
- Download associated PDF and XBRL files
- Store disclosure metadata in a database
- Classify disclosures by category and subcategory
- Handle ghost disclosures (results beyond page 11)
- Support for single-day, date range, and historical scraping

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `directories.json` file in the root directory:
   ```json
   {
     "tdnet_search_pdf_directory": "pdfs_tdnet_search",
     "tdnet_search_xbrl_directory": "xbrls_tdnet_search"
   }
   ```

## Usage

### TDnet Search Scraper

```bash
# Scrape a single date
python src/scraper_tdnet_search/tdnet_search_scraper.py date YYYY-MM-DD

# Scrape a date range
python src/scraper_tdnet_search/tdnet_search_scraper.py range START_DATE END_DATE

# Scrape historical data (from a date backwards)
python src/scraper_tdnet_search/tdnet_search_scraper.py historical [START_DATE]

# Default: scrape from one month ago
python src/scraper_tdnet_search/tdnet_search_scraper.py
```

### Debug Scraper

A version of the scraper that doesn't save to database or download files:

```bash
# Scrape a single date
python src/scraper_tdnet_search/debug_scraper.py date YYYY-MM-DD

# Scrape a date range
python src/scraper_tdnet_search/debug_scraper.py range START_DATE END_DATE

# Scrape historical data
python src/scraper_tdnet_search/debug_scraper.py historical [START_DATE]
```

### Reset Tool

```bash
# Reset all data
python scripts/reset_tdnet_search.py

# Reset a single date's data
python scripts/reset_tdnet_search.py date YYYY-MM-DD
```

## Project Structure

```
.
├── src/
│   └── scraper_tdnet_search/
│       ├── tdnet_search_scraper.py  # Main scraper
│       ├── debug_scraper.py         # Debug version (no DB/downloads)
│       ├── init_db_search.py        # Database initialization
│       └── google_auth.py           # Authentication module
├── scripts/
│   └── reset_tdnet_search.py        # Data reset tool
├── directories.json                  # Directory configuration
└── requirements.txt                  # Python dependencies
```

## Features

### Ghost Disclosure Handling

TDnet Search has a limitation where it only shows up to 11 pages of results. The scraper handles this by:
1. Detecting when there are ghost disclosures (results beyond page 11)
2. Using PBR filters to split large result sets
3. Recursively subdividing filters to capture all disclosures

### File Organization

- PDFs and XBRLs are stored in date-based folders
- Files are named with timestamp, company code, and title
- Paths are stored in the database for reference

### Database Schema

The scraper stores the following information:
- Disclosure date and time
- Company code and name
- Disclosure title
- PDF and XBRL file paths
- Category and subcategory
- Exchange information
- Update history

## Contributing

Feel free to submit issues and pull requests.

## License

MIT License 