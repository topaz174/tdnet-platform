#!/bin/bash
# Script to run TDnet full historical scraper (30 days)

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Starting TDnet full historical scraper..."
echo "$(date)" >> full_scrape_log.txt

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the scraper
python tdnet_scraper.py full >> full_scrape_log.txt 2>&1 