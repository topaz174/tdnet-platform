#!/usr/bin/env python3
"""
Opens IXBRL summary files in the browser based on:
1. Company code, year, and quarter
2. Direct disclosure ID from disclosures table

Provides flexible input parsing for different date formats and quarter specifications.
Uses path_derivation.py to derive XBRL paths dynamically.
"""

import os
import re
import webbrowser
import zipfile
import tempfile
from pathlib import Path
import psycopg2
from typing import Optional, List, Tuple
import sys

# Correctly set project root to import config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.config import DB_CONFIG
from src.utils.path_derivation import derive_xbrl_path

class IXBRLOpener:
    """Opens IXBRL files in the browser based on user input."""
    
    def __init__(self, db_config):
        self.db_config = db_config
    
    def run(self):
        """Main execution flow."""
        print("IXBRL File Opener")
        print("=" * 50)
        
        # Get search mode
        mode = self._get_search_mode()
        
        if mode == "id":
            # Direct disclosure ID search
            disclosure_id = self._get_disclosure_id()
            file_path = self._find_ixbrl_by_id(disclosure_id)
        else:
            # Company code + period search
            company_code = self._get_company_code()
            year, quarter = self._get_period()
            file_path = self._find_ixbrl_file(company_code, year, quarter)
        
        if file_path:
            self._open_in_browser(file_path)
        else:
            print("No matching IXBRL file found.")
    
    def _get_search_mode(self) -> str:
        """Get search mode from user."""
        while True:
            mode = input(
                "Search by:\n"
                "  1. Company code + period (default)\n"
                "  2. Disclosure ID\n"
                "Enter choice (1/2) or press Enter for default: "
            ).strip()
            
            if not mode or mode == "1":
                return "company"
            elif mode == "2":
                return "id"
            else:
                print("Please enter 1 or 2")
    
    def _get_disclosure_id(self) -> int:
        """Get disclosure ID from user."""
        while True:
            try:
                disc_id = input("Enter disclosure ID (from disclosures table): ").strip()
                return int(disc_id)
            except ValueError:
                print("Please enter a valid integer disclosure ID")
    
    def _get_company_code(self) -> str:
        """Get company code from user with flexible input."""
        while True:
            code = input("Enter company code (4-5 digits, e.g., 8783): ").strip()
            if code.isdigit() and 4 <= len(code) <= 5:
                return code
            print("Invalid company code. Please enter a 4 or 5-digit number.")
    
    def _get_period(self) -> Tuple[int, int]:
        """Get year and quarter from user with flexible input parsing."""
        while True:
            period_input = input(
                "Enter period (flexible formats):\n"
                "  Examples: '2024 Q1', '2024-1', '2024年第1四半期', 'FY2024 Q3', '令和6年第2四半期'\n"
                "  Input: "
            ).strip()
            
            result = self._parse_period(period_input)
            if result:
                year, quarter = result
                print(f"Parsed as: FY{year} Q{quarter}")
                return year, quarter
            else:
                print("Could not parse the period. Please try again.")
    
    def _parse_period(self, period_input: str) -> Optional[Tuple[int, int]]:
        """Parse various period input formats."""
        period_input = period_input.upper().replace('　', ' ')  # Replace full-width space
        
        # Extract year
        year = None
        
        # Try different year patterns
        patterns = [
            r'(\d{4})',  # 4-digit year
            r'FY(\d{4})',  # FY2024
            r'(\d{4})年',  # 2024年
            r'令和(\d+)年',  # 令和6年
            r'平成(\d+)年',  # 平成31年
        ]
        
        for pattern in patterns:
            match = re.search(pattern, period_input)
            if match:
                if '令和' in pattern:
                    year = 2018 + int(match.group(1))
                elif '平成' in pattern:
                    year = 1988 + int(match.group(1))
                else:
                    year = int(match.group(1))
                break
        
        if not year:
            return None
        
        # Extract quarter
        quarter = None
        
        quarter_patterns = [
            (r'Q([1-4])', 1),  # Q1, Q2, Q3, Q4
            (r'([1-4])Q', 1),  # 1Q, 2Q, 3Q, 4Q
            (r'-([1-4])', 1),  # 2024-1, 2024-2
            (r'第([1-4])四半期', 1),  # 第1四半期
            (r'第([１-４])四半期', 1),  # 第１四半期 (full-width)
            (r'中間', 2),  # 中間期 = Q2
            (r'INTERIM', 2),  # Interim
            (r'HALF', 2),  # Half year
        ]
        
        for pattern, group_idx in quarter_patterns:
            match = re.search(pattern, period_input)
            if match:
                if isinstance(group_idx, int) and group_idx > 0:
                    quarter_str = match.group(group_idx)
                    # Handle full-width numbers
                    quarter_map = {'１': '1', '２': '2', '３': '3', '４': '4'}
                    quarter_str = quarter_map.get(quarter_str, quarter_str)
                    quarter = int(quarter_str)
                else:
                    quarter = group_idx  # For fixed values like 中間 = 2
                break
        
        # Default to Q4 if no quarter specified
        if quarter is None:
            quarter = 4
        
        return (year, quarter) if year else None
    
    def _find_ixbrl_by_id(self, disclosure_id: int) -> Optional[Path]:
        """Find IXBRL file by disclosure ID."""
        print(f"Searching for disclosure ID: {disclosure_id}")
        
        # Query database for the specific disclosure
        disclosure = self._query_disclosure_by_id(disclosure_id)
        
        if not disclosure:
            print("Disclosure not found in database.")
            return None
        
        print(f"Found: {disclosure['title']}")
        
        # Check if it has XBRL
        if not disclosure['has_xbrl']:
            print("This disclosure does not have XBRL data.")
            return None
        
        # Derive XBRL path using path_derivation
        xbrl_path = derive_xbrl_path(
            disclosure['company_code'],
            disclosure['disclosure_date'],
            disclosure['time'],
            disclosure['title']
        )
        
        # Extract and find summary file
        return self._prepare_xbrl_folder(xbrl_path)
    
    def _find_ixbrl_file(self, company_code: str, year: int, quarter: int) -> Optional[Path]:
        """Find the IXBRL file matching the criteria."""
        print(f"Searching for IXBRL file: Company {company_code}, FY{year} Q{quarter}")
        
        # Query database for matching disclosures
        disclosures = self._query_disclosures(company_code, year, quarter)
        
        if not disclosures:
            print("No matching disclosures found in database.")
            return None
        
        print(f"Found {len(disclosures)} potential matches:")
        for i, disc in enumerate(disclosures, 1):
            print(f"  {i}. {disc['title']}")
        
        # If multiple matches, let user choose
        if len(disclosures) > 1:
            while True:
                try:
                    choice = input(f"Select file (1-{len(disclosures)}) or press Enter for first: ").strip()
                    if not choice:
                        selected = disclosures[0]
                        break
                    idx = int(choice) - 1
                    if 0 <= idx < len(disclosures):
                        selected = disclosures[idx]
                        break
                    print(f"Please enter a number between 1 and {len(disclosures)}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            selected = disclosures[0]
        
        # Derive XBRL path using path_derivation
        xbrl_path = derive_xbrl_path(
            selected['company_code'],
            selected['disclosure_date'],
            selected['time'],
            selected['title']
        )
        
        # Extract and find summary file
        return self._prepare_xbrl_folder(xbrl_path)
    
    def _query_disclosures(self, company_code: str, year: int, quarter: int) -> List[dict]:
        """Query database for matching disclosures."""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Adjust company code format
            query_code = company_code
            if len(query_code) == 4:
                query_code += '0'
            
            # Query for earnings reports
            cur.execute(
                """
                SELECT company_name, title, company_code, disclosure_date, time, has_xbrl
                FROM disclosures
                WHERE company_code = %s
                AND title LIKE '%%決算短信%%'
                AND has_xbrl = true
                ORDER BY disclosure_date DESC;
                """,
                (query_code,)
            )
            
            rows = cur.fetchall()
            disclosures = []
            
            for row in rows:
                title = row[1]
                # Parse title to match year and quarter
                if self._title_matches_period(title, year, quarter):
                    disclosures.append({
                        'company_name': row[0],
                        'title': title,
                        'company_code': row[2],
                        'disclosure_date': row[3],
                        'time': row[4],
                        'has_xbrl': row[5]
                    })
            
            return disclosures
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _query_disclosure_by_id(self, disclosure_id: int) -> Optional[dict]:
        """Query database for a specific disclosure by ID."""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute(
                """
                SELECT company_name, title, company_code, disclosure_date, time, has_xbrl
                FROM disclosures
                WHERE id = %s;
                """,
                (disclosure_id,)
            )
            
            row = cur.fetchone()
            if row:
                return {
                    'company_name': row[0],
                    'title': row[1],
                    'company_code': row[2],
                    'disclosure_date': row[3],
                    'time': row[4],
                    'has_xbrl': row[5]
                }
            return None
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def _title_matches_period(self, title: str, target_year: int, target_quarter: int) -> bool:
        """Check if disclosure title matches the target period."""
        # Parse year from title
        year_match = re.search(r'(\d{4})年', title)
        if year_match:
            title_year = int(year_match.group(1))
        else:
            reiwa_match = re.search(r'令和(\d+)年', title)
            if reiwa_match:
                title_year = 2018 + int(reiwa_match.group(1))
            else:
                heisei_match = re.search(r'平成(\d+)年', title)
                if heisei_match:
                    title_year = 1988 + int(heisei_match.group(1))
                else:
                    return False
        
        if title_year != target_year:
            return False
        
        # Parse quarter from title
        title_quarter = 4  # Default to annual
        if '第１四半期' in title or '第1四半期' in title:
            title_quarter = 1
        elif '第２四半期' in title or '第2四半期' in title or '中間期' in title:
            title_quarter = 2
        elif '第３四半期' in title or '第3四半期' in title:
            title_quarter = 3
        
        return title_quarter == target_quarter
    
    def _prepare_xbrl_folder(self, xbrl_path: str) -> Optional[Path]:
        """Ensure XBRL content is available in a folder and return that folder path for viewing."""
        xbrl_path = Path(xbrl_path)
        
        if not xbrl_path.exists():
            print(f"XBRL file not found: {xbrl_path}")
            return None
        
        # If it's already a directory, return it directly
        if xbrl_path.is_dir():
            return xbrl_path
        
        # Otherwise extract the ZIP
        try:
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(xbrl_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Copy entire extracted contents to a more permanent location for easy browsing
            import shutil
            permanent_dir = Path.home() / 'temp_ixbrl'
            permanent_dir.mkdir(exist_ok=True)
            
            final_dir = permanent_dir / f"ixbrl_temp_{xbrl_path.stem}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(temp_dir, final_dir)
                
            return final_dir
            
        except Exception as e:
            print(f"Error extracting XBRL file: {e}")
        return None
    
    def _open_in_browser(self, file_path: Path):
        """Open the IXBRL file in the default browser (like double-clicking in file explorer)."""
        try:
            print(f"Opening in browser: {file_path}")
            
            # Use os.startfile on Windows (like double-clicking in file explorer)
            if os.name == 'nt':  # Windows
                os.startfile(str(file_path))
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{file_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{file_path}"')
            else:
                # Fallback to webbrowser module
                webbrowser.open(file_path.as_uri())
            
            print("File opened successfully!")
        except Exception as e:
            print(f"Error opening file: {e}")
            print(f"You can manually open: {file_path}")


def main():
    """Main function."""
    db_config = DB_CONFIG.copy()
    db_config['database'] = 'tdnet'
    
    if not all(db_config.values()):
        print("Database configuration is incomplete. Please check your config/config.py and .env file.")
        return
    
    opener = IXBRLOpener(db_config)
    opener.run()


if __name__ == "__main__":
    main() 