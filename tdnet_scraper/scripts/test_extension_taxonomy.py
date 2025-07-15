#!/usr/bin/env python3
"""
Test extension taxonomy reading from XSD files.
Takes an XSD path as input and outputs all concepts found with their labels.
"""

import sys
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
from dateutil.parser import parse as dt_parse
from zipfile import ZipFile, Path as ZipPath

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    print("ERROR: lxml not installed. Install with: pip install lxml")
    sys.exit(1)

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.etl.load_concepts import _parse_extension_xsd_content, _parse_label_content, _extract_taxonomy_info


def _find_extension_taxonomy_files_in_package(pkg_root, company_code: str) -> Dict[str, Optional[Path]]:
    """Locate XSD and label linkbase files for a company‐specific extension taxonomy."""
    files: Dict[str, Optional[Path]] = {'xsd': None, 'lab_ja': None, 'lab_en': None}

    # Two naming schemes are observed; accommodate both.
    xsd_pattern1 = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}\.xsd"
    xsd_pattern2 = rf"tse-\w+-{re.escape(company_code)}-\d+\.xsd"

    lab_pattern1 = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}-lab\.xml"
    lab_pattern2 = rf"tse-\w+-{re.escape(company_code)}-\d+-lab\.xml"

    lab_en_pattern1 = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}-lab-en\.xml"
    lab_en_pattern2 = rf"tse-\w+-{re.escape(company_code)}-\d+-lab-en\.xml"

    def _iter_package_files(pkg_root):
        """Yield every file contained in a directory or ZipPath tree."""
        try:
            yield from pkg_root.rglob('*')  # pathlib.Path supports rglob
        except AttributeError:
            stack = [pkg_root]
            while stack:
                node = stack.pop()
                try:
                    if node.is_dir():
                        stack.extend(list(node.iterdir()))
                    else:
                        yield node
                except Exception:
                    continue

    for fp in _iter_package_files(pkg_root):
        fn = fp.name

        if re.match(xsd_pattern1, fn, re.IGNORECASE) or re.match(xsd_pattern2, fn, re.IGNORECASE):
            files['xsd'] = fp
        elif re.match(lab_en_pattern1, fn, re.IGNORECASE) or re.match(lab_en_pattern2, fn, re.IGNORECASE):
            files['lab_en'] = fp
        elif re.match(lab_pattern1, fn, re.IGNORECASE) or re.match(lab_pattern2, fn, re.IGNORECASE):
            files['lab_ja'] = fp

    return files


def test_extension_taxonomy(xsd_path: str, company_code: str = None) -> None:
    """Test extension taxonomy reading from a specific XSD file."""
    pkg_path = Path(xsd_path)
    
    if not pkg_path.exists():
        print(f"ERROR: File does not exist: {xsd_path}")
        return
    
    print(f"Testing extension taxonomy: {xsd_path}")
    print("=" * 80)
    
    # Extract company code from path if not provided
    if not company_code:
        # Try to extract from path
        path_str = str(pkg_path)
        match = re.search(r'(\d{4,5})', path_str)
        if match:
            company_code = match.group(1)
        else:
            company_code = "UNKNOWN"
    
    print(f"Company code: {company_code}")
    
    # Check if this is a direct XSD file or a package
    if pkg_path.suffix.lower() == '.xsd':
        # Direct XSD file
        print("Processing direct XSD file...")
        
        # Parse XSD content
        print("Parsing XSD content...")
        with pkg_path.open('rb') as fh:
            concepts_map = _parse_extension_xsd_content(fh.read())
        
        print(f"Found {len(concepts_map)} concepts in XSD")
        
        # Look for label files in the same directory
        xsd_dir = pkg_path.parent
        labels_ja: Dict[str, str] = {}
        labels_en: Dict[str, str] = {}
        
        # Find label files with similar naming pattern
        xsd_name = pkg_path.stem
        lab_ja_pattern = f"{xsd_name}-lab.xml"
        lab_en_pattern = f"{xsd_name}-lab-en.xml"
        
        lab_ja_file = xsd_dir / lab_ja_pattern
        lab_en_file = xsd_dir / lab_en_pattern
        
        if lab_ja_file.exists():
            print(f"Found Japanese labels: {lab_ja_file}")
            with lab_ja_file.open('rb') as fh:
                labels_ja = _parse_label_content(fh.read())
            print(f"Found {len(labels_ja)} Japanese labels")
        
        if lab_en_file.exists():
            print(f"Found English labels: {lab_en_file}")
            with lab_en_file.open('rb') as fh:
                labels_en = _parse_label_content(fh.read())
            print(f"Found {len(labels_en)} English labels")
        
        # Extract taxonomy info
        tax_prefix, tax_version = _extract_taxonomy_info(pkg_path.name)
        
    else:
        # Package structure (zip or directory)
        if pkg_path.suffix.lower() == '.zip':
            zf = ZipFile(pkg_path, 'r')
            pkg_root = ZipPath(zf)
        else:
            pkg_root = pkg_path
        
        # Find taxonomy files
        taxonomy_files = _find_extension_taxonomy_files_in_package(pkg_root, company_code)
        
        if not taxonomy_files['xsd']:
            print("ERROR: No extension XSD found")
            return
        
        print(f"Found XSD: {taxonomy_files['xsd']}")
        if taxonomy_files['lab_ja']:
            print(f"Found Japanese labels: {taxonomy_files['lab_ja']}")
        if taxonomy_files['lab_en']:
            print(f"Found English labels: {taxonomy_files['lab_en']}")
        
        # Parse XSD content
        print("\nParsing XSD content...")
        with taxonomy_files['xsd'].open('rb') as fh:
            concepts_map = _parse_extension_xsd_content(fh.read())
        
        print(f"Found {len(concepts_map)} concepts in XSD")
        
        # Parse labels if available
        labels_ja: Dict[str, str] = {}
        labels_en: Dict[str, str] = {}
        
        if taxonomy_files['lab_ja']:
            print("Parsing Japanese labels...")
            with taxonomy_files['lab_ja'].open('rb') as fh:
                labels_ja = _parse_label_content(fh.read())
            print(f"Found {len(labels_ja)} Japanese labels")
        
        if taxonomy_files['lab_en']:
            print("Parsing English labels...")
            with taxonomy_files['lab_en'].open('rb') as fh:
                labels_en = _parse_label_content(fh.read())
            print(f"Found {len(labels_en)} English labels")
        
        # Extract taxonomy info
        tax_prefix, tax_version = _extract_taxonomy_info(taxonomy_files['xsd'].name)
        
        if pkg_path.suffix.lower() == '.zip':
            zf.close()
    
    print(f"\nTaxonomy prefix: {tax_prefix}")
    print(f"Taxonomy version: {tax_version}")
    
    # Display results in table format
    print(f"\n{'Concept Name':<50} {'Item Type':<15} {'Japanese Label':<30} {'English Label':<30}")
    print("-" * 125)
    
    for local_name, item_type in concepts_map.items():
        ja_label = labels_ja.get(local_name, '')
        en_label = labels_en.get(local_name, '')
        
        # Truncate labels for display
        ja_display = ja_label[:27] + "..." if len(ja_label) > 30 else ja_label
        en_display = en_label[:27] + "..." if len(en_label) > 30 else en_label
        
        print(f"{local_name:<50} {item_type or 'N/A':<15} {ja_display:<30} {en_display:<30}")
    
    print(f"\nTotal concepts: {len(concepts_map)}")
    print(f"Concepts with Japanese labels: {len([c for c in concepts_map.keys() if c in labels_ja])}")
    print(f"Concepts with English labels: {len([c for c in concepts_map.keys() if c in labels_en])}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python test_extension_taxonomy.py <xsd_path> [company_code]")
        print("Example: python test_extension_taxonomy.py downloads/xbrls/2025-05-08/15-30_45020_2025年３月期　決算短信〔IFRS〕（連結）/XBRLData/Attachment/tse-acediffr-45020-2025-03-31-01-2025-05-08.xsd 45020")
        sys.exit(1)
    
    xsd_path = sys.argv[1]
    company_code = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_extension_taxonomy(xsd_path, company_code)


if __name__ == "__main__":
    main() 