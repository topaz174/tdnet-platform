#!/usr/bin/env python3
import sys
import os
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
import psycopg2
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import re

# Add project root to path to import config
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config.config import DB_CONFIG

# Import Arelle API
try:
    from arelle.api.Session import Session
    from arelle.RuntimeOptions import RuntimeOptions
except ModuleNotFoundError:
    print("ERROR: The 'arelle-release' package is required. Install with:\n"
          "       pip install arelle-release")
    sys.exit(1)


class ArelleFinancialTableGenerator:
    """
    Extracts financial data from XBRLs using Arelle, processes it, 
    and generates reports for a given company.
    """
    def __init__(self, company_code, db_config):
        self.company_code = company_code
        self.db_config = db_config
        self.data = []
        self.company_name = ""
        
        # The target metrics we want to extract
        self.target_metrics = {
            '売上高': ['Revenue', 'NetSales', 'Sales', 'OperatingRevenues'],
            '営業利益': ['OperatingIncome', 'OperatingProfit'],
            '経常利益': ['OrdinaryIncome', 'OrdinaryProfit', 'ProfitBeforeTax'],
            '純利益': ['ProfitAttributableToOwnersOfParent', 'NetIncome', 'Profit'],
            '1株当たり純利益': ['NetIncomePerShare', 'EarningsPerShare', 'BasicEarningsPerShare']
        }

    def run(self):
        """Main execution flow."""
        print(f"Starting financial table generation for company code: {self.company_code}")
        disclosures = self._fetch_disclosures()
        if not disclosures:
            print(f"No 'Earnings Report' disclosures found for company code {self.company_code}.")
            return

        self.company_name = disclosures[0]['company_name']
        self._process_disclosures(disclosures)
        self._save_results()
        print("Financial table generation complete.")

    def _fetch_disclosures(self):
        """Fetch relevant disclosures from the PostgreSQL database."""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            print(f"Querying database for company code: {self.company_code}")
            # The company code in db might have a '0' at the end.
            query_code = self.company_code
            if len(query_code) == 4:
                query_code += '0'

            # Fetch disclosures that are earnings reports
            cur.execute(
                """
                SELECT company_name, title, xbrl_path FROM disclosures
                WHERE company_code = %s
                AND subcategory LIKE '%%決算短信%%'
                AND xbrl_path IS NOT NULL
                ORDER BY disclosure_date DESC;
                """,
                (query_code,)
            )
            rows = cur.fetchall()
            print(f"Found {len(rows)} potential earnings reports.")
            return [{'company_name': r[0], 'title': r[1], 'xbrl_path': r[2]} for r in rows]
        except psycopg2.OperationalError as e:
            print(f"Error connecting to the database: {e}")
            print("Please ensure PostgreSQL is running and the .env file with credentials is set up correctly.")
            return []
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _process_disclosures(self, disclosures):
        """Process each disclosure to extract financial data."""
        processed_records = {}

        for disc in disclosures:
            xbrl_zip_path = Path(disc['xbrl_path'])
            if not xbrl_zip_path.exists():
                print(f"  SKIP: XBRL file or directory not found at {xbrl_zip_path}")
                continue

            print(f"Processing: {disc['title']}")

            # Determine the directory that contains the extracted XBRL files
            if xbrl_zip_path.is_dir():
                extraction_dir = xbrl_zip_path
                self._process_extracted_dir(extraction_dir, disc, processed_records)
            else:
                # Extract the ZIP archive to a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        with zipfile.ZipFile(xbrl_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        extraction_dir = Path(temp_dir)
                        self._process_extracted_dir(extraction_dir, disc, processed_records)
                    except (zipfile.BadZipFile, FileNotFoundError, PermissionError) as e:
                        print(f"  ERROR: Could not open or extract zip file: {xbrl_zip_path} - {e}")

        # Update self.data after processing all disclosures
        self.data = sorted(list(processed_records.values()), key=lambda x: (x['year'], x['quarter']), reverse=True)

    def _process_extracted_dir(self, extraction_dir: Path, disc: dict, processed_records: dict):
        """Process an extracted XBRL directory using Arelle."""
        summary_file = self._find_summary_file(extraction_dir)
        if not summary_file:
            print("  SKIP: No summary .htm file found in extracted directory")
            return

        try:
            # Use Arelle to parse the XBRL document
            period_info, financial_data, full_arelle_data = self._extract_data_with_arelle(summary_file)
            
            if not period_info:
                print("  INFO: Falling back to title parsing for period.")
                period_info = self._parse_title(disc['title'])
                if not period_info:
                    print(f"  SKIP: Could not determine period for {disc['title']}")
                    return
            
            year, quarter = period_info['year'], period_info['quarter']
            title_info = self._parse_title(disc['title'])
            correction = title_info['correction'] if title_info else False

            if all(v == 0 for v in financial_data.values()):
                print("  WARN: No financial data extracted. Skipping record.")
                return

            record = {
                'year': year, 
                'quarter': quarter, 
                'correction': correction, 
                **financial_data,
                'full_arelle_data': full_arelle_data
            }

            record_key = (year, quarter)
            if record_key not in processed_records:
                processed_records[record_key] = record
                print(f"  OK: Added data for {year}-Q{quarter}")
            elif correction and not processed_records[record_key].get('correction'):
                processed_records[record_key] = record
                print(f"  OK: Updated with correction for {year}-Q{quarter}")
            else:
                print(f"  INFO: Duplicate for {year}-Q{quarter}, skipping.")

        except Exception as e:
            print(f"  ERROR: Exception in _process_extracted_dir: {e}")

    def _extract_data_with_arelle(self, xbrl_file: Path) -> Tuple[Optional[Dict], Dict[str, float], List[Dict]]:
        """Extract period and financial data using Arelle API."""
        opts = RuntimeOptions(
            entrypointFile=str(xbrl_file.resolve()),
            internetConnectivity="offline",  # rely on cached taxonomy
            keepOpen=True,
            logFormat="[%(messageCode)s] %(message)s - %(file)s",
            logPropagate=False,
        )

        period_info = None
        financial_data = {metric: 0 for metric in self.target_metrics.keys()}
        full_arelle_data = []

        try:
            with Session() as session:
                session.run(opts)
                model_xbrls = session.get_models()
                if not model_xbrls:
                    print(f"  ERROR: Arelle could not parse {xbrl_file}")
                    return None, financial_data, []
                
                model_xbrl = model_xbrls[0]
                
                # Extract period information
                period_info = self._get_document_period_arelle(model_xbrl)
                
                # Extract financial data for our target metrics
                for jp_name, concept_names in self.target_metrics.items():
                    value = self._find_metric_in_model(model_xbrl, concept_names)
                    if value is not None:
                        financial_data[jp_name] = value

                # Store full Arelle data
                for fact in model_xbrl.facts:
                    fact_data = {
                        'concept': fact.concept.name,
                        'value': fact.value,
                        'context': fact.context.id,
                        'period_type': fact.concept.periodType,
                        'labels': {
                            'ja': fact.concept.label(lang='ja'),
                            'en': fact.concept.label(lang='en')
                        }
                    }
                    full_arelle_data.append(fact_data)
                
                return period_info, financial_data, full_arelle_data
        except Exception as e:
            print(f"  ERROR: Arelle extraction failed: {e}")
            return None, financial_data, []

    def _get_document_period_arelle(self, model_xbrl) -> Optional[Dict]:
        """Get period information from Arelle model."""
        try:
            # Look for fiscal year end
            fiscal_year = None
            fy_facts = []
            
            # Find FiscalYearEnd facts
            for fact in model_xbrl.facts:
                if 'FiscalYearEnd' in fact.concept.name:
                    fy_facts.append(fact)
            
            # Prefer CurrentYearInstant context
            for fact in fy_facts:
                if fact.context.id == 'CurrentYearInstant':
                    try:
                        date_text = fact.value
                        fiscal_year = datetime.strptime(date_text, '%Y-%m-%d').year
                        break
                    except (ValueError, TypeError):
                        continue
            
            # If no match, use any FiscalYearEnd fact
            if fiscal_year is None and fy_facts:
                for fact in fy_facts:
                    try:
                        date_text = fact.value
                        fiscal_year = datetime.strptime(date_text, '%Y-%m-%d').year
                        break
                    except (ValueError, TypeError):
                        continue
            
            if fiscal_year is None:
                return None
                
            # Look for quarterly period
            quarter = None
            for fact in model_xbrl.facts:
                if 'QuarterlyPeriod' in fact.concept.name:
                    try:
                        quarter = int(fact.value)
                        break
                    except (ValueError, TypeError):
                        continue
            
            # If QuarterlyPeriod is not found, check if it's annual report
            if quarter is None:
                for fact in model_xbrl.facts:
                    if 'TypeOfCurrentPeriodDEI' in fact.concept.name and fact.value == 'FY':
                        quarter = 4
                        break
                
                # If still not found, assume Q4 for full year reports
                if quarter is None:
                    quarter = 4
            
            return {'year': fiscal_year, 'quarter': quarter}
        except Exception as e:
            print(f"  ERROR: Error getting period: {e}")
            return None
    
    def _find_metric_in_model(self, model_xbrl, concept_names: List[str]) -> Optional[float]:
        """Return the metric value, preferring consolidated contexts and exact matches over partial matches."""

        candidates = []

        # Iterate once over all facts to gather potential matches
        for fact in model_xbrl.facts:
            concept_name = fact.concept.name
            
            # Check if this fact matches any of our target concepts
            matched_alias = None
            for alias in concept_names:
                # Prefer exact matches over substring matches
                if concept_name == alias:
                    matched_alias = alias
                    match_quality = 3  # Exact match
                    break
                elif concept_name.endswith(alias):
                    matched_alias = alias
                    match_quality = 2  # Ends with the alias
                elif alias in concept_name:
                    matched_alias = alias
                    match_quality = 1  # Contains the alias
            
            if not matched_alias:
                continue

            if not self._is_valid_context(fact.context, fact.concept):
                continue

            try:
                val = float(fact.xValue) if hasattr(fact, 'xValue') and fact.xValue is not None else float(str(fact.value).replace(',', ''))
            except (ValueError, TypeError):
                continue

            ctx_id = fact.context.id.lower()
            
            # Determine context quality
            if 'consolidatedmember' in ctx_id:
                context_quality = 2  # Explicitly consolidated
            elif 'member' not in ctx_id:
                context_quality = 1  # No member (implicitly consolidated)
            else:
                context_quality = 0  # Other valid context
            
            # Store this candidate with its quality scores
            candidates.append({
                'value': val,
                'match_quality': match_quality,
                'context_quality': context_quality,
                'concept': concept_name,
                'context': ctx_id
            })
        
        # Sort candidates by match quality (exact > ends with > contains)
        # then by context quality (consolidated > no member > other)
        if candidates:
            candidates.sort(key=lambda x: (x['match_quality'], x['context_quality']), reverse=True)
            return candidates[0]['value']
            
        return None

    def _is_valid_context(self, context, concept) -> bool:
        """Return True only for consolidated (or unspecified) contexts of the current period.

        Any context that contains "NonConsolidatedMember" (case-insensitive) is rejected so
        that standalone parent-company figures are never picked up for consolidated KPIs.
        """

        ctx_id = context.id.lower()

        # Hard reject non-consolidated contexts
        if 'nonconsolidatedmember' in ctx_id:
            return False

        # Duration (flows like revenue, profit)
        if context.isStartEndPeriod and concept.periodType == 'duration':
            return (
                ('current' in ctx_id or 'result' in ctx_id) and
                ('duration' in ctx_id) and
                (
                    'consolidatedmember' in ctx_id  # explicitly consolidated
                    or 'member' not in ctx_id       # or no member specified (implicitly consolidated)
                )
            )

        # Instant (balance-sheet items, EPS usually instant in TD-net taxonomy)
        if context.isInstantPeriod and concept.periodType == 'instant':
            return (
                'current' in ctx_id and
                'instant' in ctx_id and
                (
                    'consolidatedmember' in ctx_id or 'member' not in ctx_id
                )
            )

        return False

    def _find_summary_file(self, folder_path: Path):
        """Find the summary IXBRL file in the extracted folder."""
        summary_files = []
        other_htm_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.htm', '.html')):
                    p = Path(root, file)
                    if 'summary' in str(p).lower():
                        summary_files.append(p)
                    else:
                        other_htm_files.append(p)
        
        if summary_files:
            return summary_files[0]
        if other_htm_files:
            return other_htm_files[0]
        return None

    def _parse_title(self, title):
        """Parse the disclosure title to extract year, quarter, and correction status."""
        try:
            correction = any(corr in title for corr in ['(訂正', '（訂正'])
            year_match = re.search(r'(\d{4})年', title)
            if year_match:
                year = int(year_match.group(1))
            else:
                reiwa_match = re.search(r'令和(\d+)年', title)
                if reiwa_match:
                    year = 2018 + int(reiwa_match.group(1))
                else:
                    heisei_match = re.search(r'平成(\d+)年', title)
                    if heisei_match:
                        year = 1988 + int(heisei_match.group(1))
                    else:
                        return None # Cannot determine year

            quarter = 4
            if '第１四半期' in title or '第1四半期' in title:
                quarter = 1
            elif '第２四半期' in title or '第2四半期' in title or '中間期' in title:
                quarter = 2
            elif '第３四半期' in title or '第3四半期' in title:
                quarter = 3
                
            return {'year': year, 'quarter': quarter, 'correction': correction}
        except Exception:
            return None

    def _save_results(self):
        """Save the processed data to Markdown and full Arelle JSON files."""
        if not self.data:
            print("No data processed, skipping file generation.")
            return

        output_dir = Path('xbrl_tables')
        output_dir.mkdir(exist_ok=True)
        output_basename = f"{self.company_code}_{self.company_name.strip()}_financial_table"
        md_file = output_dir / f"{output_basename}.md"
        arelle_json_file = output_dir / f"{output_basename}_arelle.json"
        
        # Save full Arelle data
        full_arelle_data = []
        for record in self.data:
            if 'full_arelle_data' in record:
                arelle_record = {
                    'year': record['year'],
                    'quarter': record['quarter'],
                    'correction': record.get('correction', False),
                    'facts': record['full_arelle_data']
                }
                full_arelle_data.append(arelle_record)
        
        if full_arelle_data:
            with open(arelle_json_file, 'w', encoding='utf-8') as f:
                json.dump(full_arelle_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved full Arelle data to {arelle_json_file}")
        
        # Save Markdown
        md_content = self._generate_markdown_table()
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Successfully saved Markdown table to {md_file}")

    def _generate_markdown_table(self):
        """Generates a Markdown table from the processed data."""
        if not self.data:
            return "No data available."
            
        title = f"# {self.company_name} ({self.company_code}) Financial Data\n\n"
        headers = ['決算期', '四半期', '売上高', '営業利益', '経常利益', '純利益', '1株当たり純利益']
        
        table = [f"| {' | '.join(headers)} |", f"|{'|'.join(['---'] * len(headers))}|"]
        
        last_year = None
        for item in self.data:
            year = item.get('year')
            display_year = str(year) if year != last_year else ''
            last_year = year
            
            row = [
                display_year,
                item.get('quarter', ''),
                self._fmt(item.get('売上高', 0)),
                self._fmt(item.get('営業利益', 0)),
                self._fmt(item.get('経常利益', 0)),
                self._fmt(item.get('純利益', 0)),
                self._fmt(item.get('1株当たり純利益', 0)),
            ]
            table.append(f"| {' | '.join(map(str, row))} |")
            
        return title + '\n'.join(table)

    @staticmethod
    def _fmt(value):
        """Human-readable formatting that preserves decimals when present."""
        if isinstance(value, (int, float)):
            # Keep two decimal places if the value is not an integer
            if isinstance(value, float) and not value.is_integer():
                return f"{value:,.2f}"
            return f"{value:,.0f}"
        return str(value)


def main():
    """Main function to run the script."""
    db_config = DB_CONFIG
    db_config['database'] = 'tdnet'

    if not all(db_config.values()):
        print("Database configuration is incomplete. Please check your config/config.py and .env file.")
        return

    company_code = input("Enter the company code (e.g., 8783): ").strip()
    if not company_code.isdigit() or not (4 <= len(company_code) <= 5):
        print("Invalid company code. Please enter a 4 or 5-digit code.")
        return

    generator = ArelleFinancialTableGenerator(company_code, db_config)
    generator.run()


if __name__ == "__main__":
    main()