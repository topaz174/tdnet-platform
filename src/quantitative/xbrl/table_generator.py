#!/usr/bin/env python3
"""
Generates a financial data table for a given company code by extracting
data from XBRL files stored in the tdnet_search database.
"""

import os
import re
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
import psycopg2
from datetime import datetime
from typing import Optional, Dict, Set
import sys
import subprocess

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Excel output will not be available. `pip install pandas openpyxl`")

# Correctly set project root to import config
# The script is in src/xbrl_parser, so root is two levels up.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config.config import DB_CONFIG

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    print("Warning: lxml not installed. Period detection will be less reliable. `pip install lxml`")

from dateutil.parser import parse as dt_parse

class FinancialTableGenerator:
    """
    Extracts financial data from XBRLs, processes it, and generates reports.
    """
    def __init__(self, company_code, db_config):
        self.company_code = company_code
        self.db_config = db_config
        self.data = []
        self.company_name = ""
        
        # Load taxonomy from concepts.json
        self.taxonomy = self._load_taxonomy()
        
        # Load contexts from contexts.json
        self.contexts = self._load_contexts()
        
        # Track which columns have values across all processed XBRLs
        self.columns_with_values = set()

        # Metrics that are reported cumulatively and need quarter-on-quarter diff
        self.cumulative_metrics = {
            '売上高', '営業利益', '経常利益・税引前利益', '純利益',
            '営業活動によるキャッシュフロー', '投資活動によるキャッシュフロー', '財務活動によるキャッシュフロー'
        }

    def _load_taxonomy(self) -> Dict[str, list]:
        """Load the taxonomy mapping from concepts.json"""
        concepts_file = Path(__file__).parent / 'concepts.json'
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: concepts.json not found at {concepts_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing concepts.json: {e}")
            return {}

    def _load_contexts(self) -> Dict[str, list]:
        """Load the context mappings from contexts.json"""
        contexts_file = Path(__file__).parent / 'contexts.json'
        try:
            with open(contexts_file, 'r', encoding='utf-8') as f:
                contexts_data = json.load(f)
                # Flatten the structure: combine duration and instant contexts for each quarter
                flattened_contexts = {}
                for quarter, types in contexts_data.items():
                    flattened_contexts[quarter] = types.get('duration', []) + types.get('instant', [])
                return flattened_contexts
        except FileNotFoundError:
            print(f"Warning: contexts.json not found at {contexts_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing contexts.json: {e}")
            return {}

    def run(self):
        """Main execution flow."""
        print(f"Starting financial table generation for company code: {self.company_code}")
        disclosures = self._fetch_disclosures()
        if not disclosures:
            print(f"No 'Earnings Report' disclosures found for company code {self.company_code}.")
            return

        self.company_name = disclosures[0]['company_name']
        self._process_disclosures(disclosures)
        #self._convert_to_quarterly_values() 
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
            else:
                # Treat as ZIP archive
                extraction_dir = None
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        with zipfile.ZipFile(xbrl_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        extraction_dir = Path(temp_dir)
                    except (zipfile.BadZipFile, FileNotFoundError, PermissionError):
                        print(f"  ERROR: Could not open or extract zip file: {xbrl_zip_path}")
                        continue

                    # All further logic executes within this temporary context block
                    self._process_extracted_dir(extraction_dir, disc, processed_records)
                    # Continue to next disclosure after temp dir context ends
                    continue  # We handled processing already

            # If here, we have a directory on disk we can process directly
            self._process_extracted_dir(extraction_dir, disc, processed_records)

        self.data = sorted(list(processed_records.values()), key=lambda x: (x['year'], x['quarter']), reverse=True)

    def _process_extracted_dir(self, extraction_dir: Path, disc: dict, processed_records: dict):
        """Common logic to handle an already-extracted XBRL directory."""
        summary_file = self._find_summary_file(extraction_dir)
        if not summary_file:
            print("  SKIP: No summary .htm file found in extracted directory")
            return

        try:
            with open(summary_file, 'rb') as f:
                xbrl_content_bytes = f.read()
         
            # Parse once with lxml, using a recovering XML parser which is more standard
            # for XBRL/XML files and does not lowercase all tags like the HTML parser.
            if not HAS_LXML:
                print("  FATAL: lxml is required for robust parsing. Skipping file.")
                return
            
            try:
                # Use a recovering XML parser. This is better for XBRL than an HTML parser.
                parser = etree.XMLParser(recover=True, huge_tree=True, encoding='utf-8')
                # We need to feed bytes to the parser when specifying encoding
                root = etree.fromstring(xbrl_content_bytes, parser=parser)
            except Exception as e:
                print(f"  ERROR: lxml failed to parse {summary_file.name}: {e}")
                return

            period_info = self._get_document_period(root)
            
            if not period_info:
                print("  INFO: Falling back to title parsing for period.")
                period_info = self._parse_title(disc['title'])
                if not period_info:
                    print(f"  SKIP: Could not determine period for {disc['title']}")
                    return
            
            year, quarter = period_info['year'], period_info['quarter']
            title_info = self._parse_title(disc['title'])
            correction = title_info['correction'] if title_info else False

            financial_data = self._extract_financial_data(root, quarter)
            
            has_data = any(v is not None for v in financial_data.values())
            if not has_data:
                print("  WARN: No financial data extracted. Skipping record.")
                return

            record = {'year': year, 'quarter': quarter, 'correction': correction, **financial_data}

            record_key = (year, quarter)
            if record_key not in processed_records:
                processed_records[record_key] = record
                print(f"  OK: Added data for {year}-Q{quarter}")
            elif correction and not processed_records[record_key].get('correction') and has_data:
                 processed_records[record_key] = record
                 print(f"  OK: Updated with correction for {year}-Q{quarter}")
            else:
                 print(f"  INFO: Duplicate for {year}-Q{quarter}, skipping.")

        except Exception as e:
            print(f"  ERROR: Exception in _process_extracted_dir: {e}")

    def _get_document_period(self, root: etree._Element) -> Optional[dict]:
        """Parse the period using dedicated FiscalYearEnd and QuarterlyPeriod tags."""
        if not HAS_LXML:
            return None
        try:
            # 1. Fiscal year end (e.g., 2025-03-31) – use the year portion
            fy_elements = root.xpath(".//*[@name and contains(@name, ':FiscalYearEnd')]")
            fiscal_year = None
            for elem in fy_elements:
                # Prefer the element whose contextRef is exactly 'CurrentYearInstant'
                ctx = elem.get('contextRef', '')
                if ctx == 'CurrentYearInstant' or fiscal_year is None:
                    text = (elem.text or '').strip()
                    if text:
                        try:
                            fiscal_year = dt_parse(text).year
                        except Exception:
                            continue
                if ctx == 'CurrentYearInstant' and fiscal_year is not None:
                    break  # Best match obtained

            if fiscal_year is None:
                return None  # Cannot determine year

            # 2. Quarter number (1/2/3/4). Tag may be hidden (ix:hidden) but always carries QuarterlyPeriod.
            q_elements = root.xpath(".//*[@name and contains(@name, ':QuarterlyPeriod')]")
            quarter = None
            for elem in q_elements:
                text = (elem.text or '').strip()
                if text and text.isdigit():
                    try:
                        quarter = int(text)
                        break
                    except ValueError:
                        continue

            # If QuarterlyPeriod tag is not present, assume full-year results (Q4)
            if quarter is None:
                quarter = 4

            return {'year': fiscal_year, 'quarter': quarter}
        except Exception as e:
            print(f"  ERROR: Exception in _get_document_period: {e}")
            return None

    def _parse_title(self, title):
        """(Fallback) Parse the disclosure title to extract year, quarter, and correction status."""
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

    def _extract_financial_data(self, root: etree._Element, detected_quarter: int) -> Dict[str, float]:
        """Extract financial data from a parsed lxml tree using the taxonomy."""
        context_refs = self._get_contexts_for_quarter(detected_quarter)
        data = {}
        
        # Search for the original, case-sensitive attribute name 'contextRef'.
        elements_with_context = root.xpath(".//*[@contextRef]")
        
        for japanese_name, concept_variations in self.taxonomy.items():
            value = self._extract_value_from_xbrl_lxml(elements_with_context, concept_variations, context_refs)
            data[japanese_name] = value  # Keep None for missing values, don't convert to 0
            
            # Track columns that have actual values (not None)
            if value is not None:
                self.columns_with_values.add(japanese_name)
        
        return data

    def _extract_value_from_xbrl_lxml(self, elements: list, field_names: list, context_patterns: list) -> Optional[float]:
        """Extract a specific value from a list of lxml elements.
        context_patterns can be substrings that appear in contextRef."""
        fallback_candidate = None

        for elem in elements:
            context_ref = elem.get('contextRef', '')
            if not any(pat in context_ref for pat in context_patterns):
                continue

            name_attr = elem.get('name', '')
            concept_name = name_attr.split(':')[-1]

            if concept_name not in field_names:
                continue

            if elem.text is None:
                continue

            value_str = elem.text.strip().replace(',', '')
            if not value_str or value_str in ['－', '―', '△']:
                continue

            try:
                value = float(value_str)
                if elem.get('sign') == '-' or '△' in elem.text:
                    value = -value
            except (ValueError, TypeError):
                continue

            # Prefer ConsolidatedMember contexts
            if 'ConsolidatedMember' in context_ref:
                return value

            if fallback_candidate is None:
                fallback_candidate = value

        return fallback_candidate

    def _get_contexts_for_quarter(self, quarter: int) -> list:
        """Get the appropriate context references for the quarter."""
        return self.contexts.get(f'q{quarter}', self.contexts['annual'])

    def _convert_to_quarterly_values(self):
        """Convert cumulative figures to standalone quarterly values."""
        if not self.data:
            return
        # work on ascending order per fiscal year
        data_sorted = sorted(self.data, key=lambda x: (x['year'], x['quarter']))
        data_map = {(d['year'], d['quarter']): d for d in data_sorted}
        for item in data_sorted:
            year, q = item['year'], item['quarter']
            if q == 1:
                continue  # Q1 already standalone
            prev = data_map.get((year, q - 1))
            if not prev:
                continue
            for key in self.cumulative_metrics:
                if key not in item:
                    continue
                curr_val = item.get(key)
                prev_val = prev.get(key)
                if curr_val is None or prev_val is None:
                    continue
                try:
                    item[key] = curr_val - prev_val
                except Exception:
                    continue
        # update self.data preserving original sort (descending by year-quarter)
        self.data = sorted(data_sorted, key=lambda x: (x['year'], x['quarter']), reverse=True)

    def _save_results(self):
        """Save the processed data to Excel and JSON files inside xbrl_tables directory."""
        if not self.data:
            print("No data processed, skipping file generation.")
            return

        output_dir = Path('xbrl_tables')
        output_dir.mkdir(exist_ok=True)
        output_basename = f"{self.company_code}_{self.company_name.strip()}_financial_table"
        excel_file = output_dir / f"{output_basename}.xlsx"
        json_file = output_dir / f"{output_basename}.json"
        
        # Save JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, default=str)
        print(f"Successfully saved data to {json_file}")
        
        # Save Excel
        if HAS_PANDAS:
            success = self._save_excel_file(excel_file)
            if success:
                self._open_excel_file(excel_file)
        else:
            print("Pandas not available. Skipping Excel file generation.")

    def _save_excel_file(self, excel_file):
        """Save the processed data to an Excel file."""
        try:
            # Base headers that are always included
            base_headers = ['決算期', '四半期']
            
            # Only include columns that have values in any of the processed XBRLs
            # Maintain the order from concepts.json
            data_headers = [col for col in self.taxonomy.keys() if col in self.columns_with_values]
            
            if not data_headers:
                print("No financial data found in processed XBRLs. Skipping Excel generation.")
                return
            
            headers = base_headers + data_headers
            
            # Prepare data for DataFrame
            rows = []
            for item in self.data:
                row_data = {
                    '決算期': item.get('year'),
                    '四半期': item.get('quarter')
                }
                
                # Add data values for columns that have values
                for col in data_headers:
                    value = item.get(col)
                    if value is None:
                        row_data[col] = ""  # Use "-" for missing values
                    else:
                        row_data[col] = value
                
                rows.append(row_data)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Save to Excel with formatting
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Financial Data', index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Financial Data']
                
                # Add title
                worksheet.insert_rows(1)
                worksheet.insert_rows(1)
                worksheet['A1'] = f"{self.company_name} ({self.company_code}) Financial Data"
                worksheet['A1'].font = workbook.create_font(size=14, bold=True)
                
                # Auto-adjust column widths to fit content and headers
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            cell_value = str(cell.value) if cell.value is not None else ""
                            if len(cell_value) > max_length:
                                max_length = len(cell_value)
                        except:
                            pass
                    
                    # Ensure minimum width to accommodate column headers properly
                    # Add extra padding for readability
                    adjusted_width = max(max_length + 3, 10)  # Minimum 10 chars, +3 padding
                    adjusted_width = min(adjusted_width, 25)  # Cap at 25 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"Successfully saved Excel file to {excel_file}")
            return True
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            return False

    def _open_excel_file(self, excel_file):
        """Open the Excel file in the default application (like double-clicking in file explorer)."""
        try:
            print(f"Opening Excel file: {excel_file}")
            
            # Use OS-specific commands to open the file with default application
            if os.name == 'nt':  # Windows
                os.startfile(str(excel_file))
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', str(excel_file)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(excel_file)])
            else:
                print("Cannot auto-open file on this operating system")
                return
            
            print("Excel file opened successfully!")
            
        except Exception as e:
            print(f"Could not auto-open Excel file: {e}")
            print(f"You can manually open: {excel_file}")

    @staticmethod
    def _fmt(value):
        """Format value for display - return '-' for None, formatted number for numeric values."""
        if value is None:
            return "-"
        elif isinstance(value, (int, float)):
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

    generator = FinancialTableGenerator(company_code, db_config)
    generator.run()


if __name__ == "__main__":
    main() 