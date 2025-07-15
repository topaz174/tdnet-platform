#!/bin/bash
# Script to run TDnet recent scraper (last 5 minutes)

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Starting TDnet recent scraper (last 5 minutes)..."
echo "$(date)" >> scraper_log.txt

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the scraper
python tdnet_scraper.py >> scraper_log.txt 2>&1 