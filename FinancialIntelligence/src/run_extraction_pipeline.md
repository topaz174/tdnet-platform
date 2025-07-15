  Start:
  python src/unified_extraction_pipeline.py \
    --full-pipeline \
    --workers 28 \
    --concurrent-files 16 \
    --batch-size 8 \
    --memory-limit 100 \
    --gpu-ocr
  
  To check status first:
  python src/unified_extraction_pipeline.py --status-report

  If interrupted and need to resume later:
  python src/unified_extraction_pipeline.py \
    --resume \
    --full-pipeline \
    --workers 28 \
    --concurrent-files 16 \
    --batch-size 8 \
    --memory-limit 100 \
    --gpu-ocr

  Option 2: Save to log file AND see output in terminal (recommended)
  python src/unified_extraction_pipeline.py \
    --resume \
    --full-pipeline \
    --workers 28 \
    --concurrent-files 16 \
    --batch-size 8 \
    --memory-limit 100 \
    --gpu-ocr \
    2>&1 | tee unified_extraction.log

  Option 1: Status report in another terminal (recommended)
  python src/unified_extraction_pipeline.py --status-report

  Option 2: Monitor the live log file
  # See recent log entries
  tail -f unified_extraction.log

  # See last 50 lines
  tail -n 50 unified_extraction.log

  # Search for progress updates specifically
  grep -i "progress:" unified_extraction.log | tail -10

  Option 3: Database query for current stats
  psql $PG_DSN -c "
  SELECT
      COALESCE(extraction_status, 'pending') as status,
      COUNT(*) as count
  FROM disclosures
  WHERE (xbrl_path IS NOT NULL AND xbrl_path != '')
     OR (pdf_path IS NOT NULL AND pdf_path != '')
  GROUP BY extraction_status
  ORDER BY count DESC;"

  Option 4: Watch progress in real-time
  # Update every 30 seconds
  watch -n 30 "python src/unified_extraction_pipeline.py --status-report"

  Option 5: Check recent completed files
  psql $PG_DSN -c "
  SELECT
      COUNT(*) as completed_today,
      extraction_method,
      AVG(extraction_duration) as avg_duration
  FROM disclosures
  WHERE extraction_date >= CURRENT_DATE
    AND extraction_status = 'completed'
  GROUP BY extraction_method;"

  The status report (Option 1) is the most comprehensive - it shows progress percentage, processing rate, ETA, and method breakdown.  