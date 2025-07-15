@echo off
cd /d %~dp0
echo Starting TDnet full historical scraper...
echo %date% %time% >> full_scrape_log.txt
call venv\Scripts\activate
python tdnet_scraper.py full >> full_scrape_log.txt 2>&1 