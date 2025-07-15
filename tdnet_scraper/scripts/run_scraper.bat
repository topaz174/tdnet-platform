@echo off
cd /d %~dp0
echo Starting TDnet recent scraper (last 5 minutes)...
echo %date% %time% >> scraper_log.txt
call venv\Scripts\activate
python tdnet_scraper.py >> scraper_log.txt 2>&1 