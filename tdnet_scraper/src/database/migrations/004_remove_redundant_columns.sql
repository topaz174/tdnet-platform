ALTER TABLE reports
  DROP COLUMN IF EXISTS report_role,
  DROP COLUMN IF EXISTS xbrl_path;