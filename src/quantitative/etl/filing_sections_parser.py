"""
Utility for parsing and discovering filing sections from XBRL packages.
"""

import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Union

from .constants import ATTACHMENT_SUFFIX_MAP


def discover_sections_from_path(xbrl_path: Path) -> List[Tuple[str, str, Optional[str], Optional[bool], Optional[int], str, str]]:
    """
    Wrapper to handle directory or zip file path for section discovery.
    
    Returns:
        List of tuples: (statement_role, rel_path, period_prefix, consolidated, layout_code, statement_role_ja, statement_role_en)
    """
    if xbrl_path.is_dir():
        return _discover_sections(xbrl_path)

    # Zip case – extract to temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(xbrl_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return _discover_sections(Path(temp_dir))
        except zipfile.BadZipFile:
            return []  # Return empty list if zip is corrupted


def _discover_sections(folder_path: Path) -> List[Tuple[str, str, Optional[str], Optional[bool], Optional[int], str, str]]:
    """
    Discover statement sections from attachment filenames with full parsing.

    Parses filenames like: 0105010-qcci13-tse-qcediffr-67580-2023-12-31-01-2024-02-14-ixbrl.htm
    Where qcci13 breaks down as: qc + ci + 13
    - qc: period_prefix='q', consolidated=True
    - ci: statement code for Comprehensive Income  
    - 13: layout_code=13

    Returns:
        List of tuples: (statement_role, rel_path, period_prefix, consolidated, layout_code, statement_role_ja, statement_role_en)
    """
    discovered: List[Tuple[str, str, Optional[str], Optional[bool], Optional[int], str, str]] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            fname = file.lower()
            if not fname.endswith(('.htm', '.html')):
                continue

            file_path = Path(root) / file
            full_path_lower = str(file_path).lower()
            
            # Calculate relative path from folder_path
            try:
                rel_path = str(file_path.relative_to(folder_path))
            except ValueError:
                continue  # Skip if file is not under folder_path

            # -----------------------------------------------
            # 1. Summary detection (special case)
            # -----------------------------------------------
            if 'summary' in full_path_lower:
                discovered.append((
                    'Summary', rel_path, None, None, None,
                    'サマリー', 'Summary'
                ))
                continue

            # ------------------------------------------------
            # 2. Revision forecast patterns (rvdf / rvfc files)
            # ------------------------------------------------
            if fname.endswith('ixbrl.htm'):
                if '-rvdf-' in fname:
                    discovered.append((
                        'DividendForecastRevision', rel_path, None, None, None,
                        '配当予想修正', 'Dividend forecast revision'
                    ))
                    continue
                if '-rvfc-' in fname:
                    discovered.append((
                        'EarningsForecastRevision', rel_path, None, None, None,
                        '業績予想修正', 'Earnings forecast revision'
                    ))
                    continue

            # -----------------------------------------------
            # 3. Parse attachment filename pattern
            # -----------------------------------------------
            # Look for pattern like: 0105010-qcci13-tse-...
            # Extract the middle part (qcci13) which contains our codes

            # Pattern to match the attachment code part
            pattern = r'\d+-([a-z]{2,4}\d{2})-tse-'
            match = re.search(pattern, fname)

            if not match:
                # Try qualitative pattern
                if 'qualitative' in full_path_lower:
                    role_info = ATTACHMENT_SUFFIX_MAP['qualitative']
                    discovered.append((
                        role_info['role'], rel_path, None, None, None,
                        role_info['ja'], role_info['en']
                    ))
                continue

            code_part = match.group(1)  # e.g., "qcci13"

            # Parse the code part
            parsed = _parse_attachment_code(code_part)
            if not parsed:
                continue
            
            period_prefix, consolidated, statement_code, layout_code = parsed
            
            # Look up statement role from the statement code
            if statement_code in ATTACHMENT_SUFFIX_MAP:
                role_info = ATTACHMENT_SUFFIX_MAP[statement_code]
                discovered.append((
                    role_info['role'], rel_path, period_prefix, consolidated, layout_code,
                    role_info['ja'], role_info['en']
                ))

    return discovered


def _parse_attachment_code(code_part: str) -> Optional[Tuple[str, bool, str, int]]:
    """
    Parse attachment code like 'qcci13' into components.
    
    Args:
        code_part: Code part from filename (e.g., 'qcci13')
        
    Returns:
        Tuple of (period_prefix, consolidated, statement_code, layout_code) or None if parsing fails
        Example: 'qcci13' -> ('q', True, 'ci', 13)
    """
    if len(code_part) < 4:
        return None
    
    # Extract period/consolidation prefix (first 2 chars)
    prefix = code_part[:2]
    
    # Map period prefix
    period_map = {'ac': 'a', 'an': 'a', 'qc': 'q', 'qn': 'q', 'sc': 's', 'sn': 's'}
    if prefix not in period_map:
        # Handle other patterns like 'bnc', 'inc' - treat as annual consolidated
        period_prefix = 'a'
        consolidated = True
    else:
        period_prefix = period_map[prefix]
        consolidated = prefix[1] == 'c'  # second char 'c' = consolidated, 'n' = non-consolidated
    
    # Extract statement code and layout code
    # Pattern: [prefix][statement_code][layout_code]
    # e.g., qcci13 -> qc + ci + 13
    remainder = code_part[2:]  # Everything after the 2-char prefix
    
    # Extract layout code (trailing digits)
    layout_match = re.search(r'(\d+)$', remainder)
    if not layout_match:
        return None
    
    layout_code = int(layout_match.group(1))
    statement_code = remainder[:layout_match.start()]  # Everything before the digits
    
    return period_prefix, consolidated, statement_code, layout_code 