#!/usr/bin/env python3
"""
Path Derivation Utilities

This module provides functions to derive XBRL and PDF file paths from disclosure metadata.
Since these paths are predictable based on company_code, date, time, and title,
we can reconstruct them on-demand instead of storing them in the database.

Path Format:
- Structure: <base_dir>/<YYYY-MM-DD>/<HH-MM>_<company_code>_<sanitized_title>.<ext>
- Time format: Always HH-MM (e.g., "15-30") since all database time objects have seconds=0
- Both scrapers store time objects without seconds and generate filenames in HH-MM format
"""

import os
import re
import json
from pathlib import Path
from datetime import date, time
from typing import Optional


def load_directories_config(project_root: Optional[str] = None) -> dict:
    """
    Load directory configuration from directories.json file.
    
    Args:
        project_root (str, optional): Project root directory
        
    Returns:
        dict: Dictionary containing directory paths
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    config_path = Path(project_root) / 'directories.json'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Fallback to default directories if config file not found
        return {
            'pdf_directory': str(Path(project_root) / 'downloads' / 'pdfs'),
            'xbrls_directory': str(Path(project_root) / 'downloads' / 'xbrls')
        }
    except Exception as e:
        print(f"Error loading directories.json: {e}")
        # Fallback to default directories
        return {
            'pdf_directory': str(Path(project_root) / 'downloads' / 'pdfs'),
            'xbrls_directory': str(Path(project_root) / 'downloads' / 'xbrls')
        }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename (str): The original filename/title
        
    Returns:
        str: Sanitized filename safe for filesystem use
    """
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores and whitespace
    sanitized = sanitized.strip('_ ')
    
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    return sanitized


def derive_xbrl_path(
    company_code: str,
    disclosure_date: date,
    disclosure_time: time,
    title: str,
    base_dir: Optional[str] = None
) -> str:
    """
    Derive the XBRL file path based on disclosure metadata.
    
    Args:
        company_code (str): Company code (e.g., "12345")
        disclosure_date (date): Date of disclosure
        disclosure_time (time): Time of disclosure
        title (str): Disclosure title
        base_dir (str, optional): Base directory for XBRL files
        
    Returns:
        str: Derived XBRL file path
    """
    if base_dir is None:
        config = load_directories_config()
        base_dir = config.get('xbrls_directory', 'downloads/xbrls')
    
    # Format date and time
    date_str = disclosure_date.strftime('%Y-%m-%d')
    
    # Always use HH-MM format since all database time objects have seconds=0
    # and the scrapers generate filenames in HH-MM format
    time_str = disclosure_time.strftime('%H-%M')
    
    # Sanitize title for filename
    sanitized_title = sanitize_filename(title)
    
    # Construct filename: TIME_COMPANY_TITLE.zip
    filename = f"{time_str}_{company_code}_{sanitized_title}.zip"
    
    # Full path: BASE_DIR/YYYY-MM-DD/FILENAME
    xbrl_path = os.path.join(base_dir, date_str, filename)
    
    return xbrl_path


def derive_pdf_path(
    company_code: str,
    disclosure_date: date,
    disclosure_time: time,
    title: str,
    base_dir: Optional[str] = None
) -> str:
    """
    Derive the PDF file path based on disclosure metadata.
    
    Args:
        company_code (str): Company code (e.g., "12345")
        disclosure_date (date): Date of disclosure
        disclosure_time (time): Time of disclosure
        title (str): Disclosure title
        base_dir (str, optional): Base directory for PDF files
        
    Returns:
        str: Derived PDF file path
    """
    if base_dir is None:
        config = load_directories_config()
        base_dir = config.get('pdf_directory', 'downloads/pdfs')
    
    # Format date and time
    date_str = disclosure_date.strftime('%Y-%m-%d')
    
    # Always use HH-MM format since all database time objects have seconds=0
    # and the scrapers generate filenames in HH-MM format
    time_str = disclosure_time.strftime('%H-%M')
    
    # Sanitize title for filename
    sanitized_title = sanitize_filename(title)
    
    # Construct filename: TIME_COMPANY_TITLE.pdf
    filename = f"{time_str}_{company_code}_{sanitized_title}.pdf"
    
    # Full path: BASE_DIR/YYYY-MM-DD/FILENAME
    pdf_path = os.path.join(base_dir, date_str, filename)
    
    return pdf_path


def check_file_exists(file_path: str, project_root: Optional[str] = None) -> bool:
    """
    Check if a derived file path actually exists on the filesystem.
    
    Args:
        file_path (str): The file path to check
        project_root (str, optional): Project root directory
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    path = Path(file_path)
    
    # Check if path exists as absolute path
    if path.exists():
        return True
    
    # If not absolute, try relative to project root
    relative_path = Path(project_root) / file_path
    if relative_path.exists():
        return True
    
    return False


def derive_and_validate_xbrl_path(
    company_code: str,
    disclosure_date: date,
    disclosure_time: time,
    title: str,
    base_dir: Optional[str] = None,
    project_root: Optional[str] = None
) -> Optional[str]:
    """
    Derive XBRL path and return it only if the file actually exists.
    
    Args:
        company_code (str): Company code
        disclosure_date (date): Date of disclosure
        disclosure_time (time): Time of disclosure
        title (str): Disclosure title
        base_dir (str, optional): Base directory for XBRL files
        project_root (str, optional): Project root directory
        
    Returns:
        str or None: XBRL path if file exists, None otherwise
    """
    xbrl_path = derive_xbrl_path(company_code, disclosure_date, disclosure_time, title, base_dir)
    
    if check_file_exists(xbrl_path, project_root):
        return xbrl_path
    
    return None


def derive_and_validate_pdf_path(
    company_code: str,
    disclosure_date: date,
    disclosure_time: time,
    title: str,
    base_dir: Optional[str] = None,
    project_root: Optional[str] = None
) -> Optional[str]:
    """
    Derive PDF path and return it only if the file actually exists.
    
    Args:
        company_code (str): Company code
        disclosure_date (date): Date of disclosure
        disclosure_time (time): Time of disclosure
        title (str): Disclosure title
        base_dir (str, optional): Base directory for PDF files
        project_root (str, optional): Project root directory
        
    Returns:
        str or None: PDF path if file exists, None otherwise
    """
    pdf_path = derive_pdf_path(company_code, disclosure_date, disclosure_time, title, base_dir)
    
    if check_file_exists(pdf_path, project_root):
        return pdf_path
    
    return None


# Example usage and testing functions
if __name__ == "__main__":
    from datetime import datetime
    
    # Example data
    sample_date = date(2025, 6, 2)
    sample_time = time(8, 30, 0)  # Time from database (always has seconds=0)
    sample_company = "21390"
    sample_title = "（訂正・数値データ訂正） 「2025年３月期 決算短信〔日本基準〕（連結）」の一部訂正に関するお知らせ"
    
    print("Path Derivation Example:")
    print("=" * 50)
    
    xbrl_path = derive_xbrl_path(sample_company, sample_date, sample_time, sample_title)
    pdf_path = derive_pdf_path(sample_company, sample_date, sample_time, sample_title)
    print(f"XBRL: {xbrl_path}")
    print(f"PDF:  {pdf_path}")
    
    print(f"\nFile existence checks:")
    print(f"  XBRL exists: {check_file_exists(xbrl_path)}")
    print(f"  PDF exists:  {check_file_exists(pdf_path)}")
    
    print(f"\nNote: All paths use HH-MM format since database time objects have seconds=0.") 