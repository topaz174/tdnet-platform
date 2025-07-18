#!/usr/bin/env python3
"""
Extension taxonomy concept processor for XBRL ETL.

This module handles the processing of company-specific extension taxonomies
found in XBRL filings, extracting concepts and labels from XSD and label files.
"""

import json
import os
import sys
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
from dateutil.parser import parse as dt_parse
from zipfile import ZipFile, Path as ZipPath

try:
    from lxml import etree  # type: ignore
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.config import DB_URL

# Share logger with facts loader so handlers defined there are reused
logger = logging.getLogger('src.etl.load_facts')


# ---------- Generic helpers ----------


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


# ---------- Extension-taxonomy utilities ----------


def _find_extension_taxonomy_files_in_package(pkg_root, company_code: str) -> Dict[str, Optional[Path]]:
    """Locate XSD and label linkbase files for a company‐specific extension taxonomy."""
    files: Dict[str, Optional[Path]] = {'xsd': None, 'lab_ja': None, 'lab_en': None}

    # Pattern for the single attachment XSD file
    xsd_pattern = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}\.xsd"
    
    # Pattern for label files (same naming convention)
    lab_pattern = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}-lab\.xml"
    lab_en_pattern = rf"tse-\w+-{re.escape(company_code)}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}-lab-en\.xml"

    for fp in _iter_package_files(pkg_root):
        # Only check files in Attachment folder
        if 'attachment' not in str(fp).lower():
            continue
            
        fn = fp.name

        if re.match(xsd_pattern, fn, re.IGNORECASE):
            files['xsd'] = fp
        elif re.match(lab_en_pattern, fn, re.IGNORECASE):
            files['lab_en'] = fp
        elif re.match(lab_pattern, fn, re.IGNORECASE):
            files['lab_ja'] = fp

    return files


def _parse_extension_xsd_content(content: bytes) -> Dict[str, Optional[str]]:
    """Return mapping of local_name → item_type for concepts defined in an XSD (vendor-agnostic)."""
    concepts: Dict[str, Optional[str]] = {}

    EXCLUDED_TYPES = {
        'stringitemtype', 'textblockitemtype', 'dateitemtype', 'anyuriitemtype',
        'numberofcompaniesitemtype', 'nonnegativeintegeritemtype', 'booleanitemtype',
    }

    if not HAS_LXML:
        return concepts

    try:
        root = etree.fromstring(content)
        
        # Use namespace-aware XPath for XML Schema elements
        XS_NS = {'xs': 'http://www.w3.org/2001/XMLSchema'}
        elems = root.xpath('.//xs:element[@name]', namespaces=XS_NS)

        for el in elems:
            name = el.get('name')
            raw_type = el.get('type') or ''
            item_type_token = raw_type.split(':')[-1].lower() if raw_type else ''

            if item_type_token in EXCLUDED_TYPES:
                continue

            # Coarse normalisation of item types
            item_type: Optional[str] = None
            if 'monetary' in item_type_token:
                item_type = 'monetary'
            elif any(t in item_type_token for t in ('string', 'text')):
                item_type = 'string'
            elif any(t in item_type_token for t in ('decimal', 'number')):
                item_type = 'decimal'
            elif 'date' in item_type_token:
                item_type = 'date'
            elif 'boolean' in item_type_token:
                item_type = 'boolean'

            if name and item_type is not None:
                concepts[name] = item_type

    except Exception as exc:
        logger.error(f"Extension XSD parsing failed: {exc}")

    return concepts


def _parse_label_content(content: bytes) -> Dict[str, str]:
    """Extract concept → label mapping from a label linkbase (handles multiple vendor formats)."""
    labels: Dict[str, str] = {}

    if not HAS_LXML:
        return labels

    try:
        root = etree.fromstring(content)
        ns_xlink = 'http://www.w3.org/1999/xlink'

        # Map locator labels to concept local names
        loc_to_concept: Dict[str, str] = {}
        for loc in root.xpath('.//*[local-name()="loc"]'):
            href = loc.get(f'{{{ns_xlink}}}href', '')
            loc_label = loc.get(f'{{{ns_xlink}}}label', '')
            if href and loc_label:
                local_name = href.split('#')[-1].split(':')[-1]
                loc_to_concept[loc_label] = local_name

        # Capture every label element once
        label_id_to_text: Dict[str, str] = {}
        for lbl in root.xpath('.//*[local-name()="label"]'):
            label_id = lbl.get(f'{{{ns_xlink}}}label') or lbl.get('id', '')
            role = lbl.get(f'{{{ns_xlink}}}role', '')
            text_raw = (lbl.text or '').strip()
            if not label_id or not text_raw:
                continue
            if 'verbose' in role.lower():
                continue  # skip verbose labels
            label_id_to_text[label_id] = text_raw

            # Heuristic: derive concept name directly from the label identifier pattern
            m = re.match(r'[^_]+_(.+?)(?:_label.*)?$', label_id)
            if m:
                labels.setdefault(m.group(1), text_raw)

        # Use arcs (common in Pronexus/FutureStage) to associate labels
        for arc in root.xpath('.//*[local-name()="labelArc"]'):
            from_lbl = arc.get(f'{{{ns_xlink}}}from')
            to_lbl = arc.get(f'{{{ns_xlink}}}to')
            concept_name = loc_to_concept.get(from_lbl)
            label_text = label_id_to_text.get(to_lbl)
            if concept_name and label_text:
                labels[concept_name] = label_text

    except Exception as exc:
        logger.error(f"Label linkbase parsing failed: {exc}")

    return labels


def _extract_taxonomy_info(filename: str) -> Tuple[str, date]:
    """Return (taxonomy_prefix, version_date) derived from extension XSD filename."""
    m1 = re.match(r'(tse-\w+-\w+)-(\d{4}-\d{2}-\d{2})-\d{2}-(\d{4}-\d{2}-\d{2})\.xsd', filename, re.IGNORECASE)
    if m1:
        prefix, _, filing_date = m1.groups()
        try:
            return prefix, dt_parse(filing_date).date()
        except Exception:
            pass

    m2 = re.match(r'(tse-\w+-\w+)-(\d+)\.xsd', filename, re.IGNORECASE)
    if m2:
        prefix, _ = m2.groups()
        return prefix, date.today()

    return f"tse-extension-{datetime.now().strftime('%Y%m%d')}", date.today()


def _insert_extension_concept(session, taxonomy_prefix: str, taxonomy_version: date, local_name: str,
                              std_label_en: Optional[str], std_label_ja: Optional[str], item_type: Optional[str]) -> Optional[int]:
    """Insert or update a single extension concept and return its ID."""
    try:
        res = session.execute(
            text("""
                INSERT INTO concepts (taxonomy_prefix, taxonomy_version, local_name, std_label_en, std_label_ja, item_type)
                VALUES (:taxonomy_prefix, :taxonomy_version, :local_name, :std_label_en, :std_label_ja, :item_type)
                ON CONFLICT (taxonomy_prefix, local_name) DO UPDATE SET
                    std_label_en = EXCLUDED.std_label_en,
                    std_label_ja = EXCLUDED.std_label_ja,
                    item_type = EXCLUDED.item_type,
                    taxonomy_version = EXCLUDED.taxonomy_version
                RETURNING id
            """),
            {
                'taxonomy_prefix': taxonomy_prefix,
                'taxonomy_version': taxonomy_version,
                'local_name': local_name,
                'std_label_en': std_label_en,
                'std_label_ja': std_label_ja,
                'item_type': item_type,
            }
        )
        row = res.fetchone()
        return row.id if row else None
    except Exception as exc:
        logger.warning(f"Failed to persist concept {local_name}: {exc}")
        return None


# ---------- Public API ----------


def process_extension_taxonomy(pkg_root, company_code: str, session, concept_cache: Dict[Tuple[str, str], int]) -> List[int]:
    """Parse & register concepts for a company‐specific extension taxonomy; return inserted IDs."""
    new_concepts: List[int] = []

    # Identify relevant files
    taxonomy_files = _find_extension_taxonomy_files_in_package(pkg_root, company_code)
    
    if not taxonomy_files['xsd']:
        logger.info(f"No extension XSD found for company {company_code} - skipping extension processing")
        return new_concepts

    # Parse core data
    with taxonomy_files['xsd'].open('rb') as fh:
        concepts_map = _parse_extension_xsd_content(fh.read())

    if not concepts_map:
        logger.error(f"Failed to parse any concepts from XSD for company {company_code}")
        return new_concepts

    labels_ja: Dict[str, str] = {}
    labels_en: Dict[str, str] = {}

    if taxonomy_files['lab_ja']:
        with taxonomy_files['lab_ja'].open('rb') as fh:
            labels_ja = _parse_label_content(fh.read())

    if taxonomy_files['lab_en']:
        with taxonomy_files['lab_en'].open('rb') as fh:
            labels_en = _parse_label_content(fh.read())

    tax_prefix, tax_version = _extract_taxonomy_info(taxonomy_files['xsd'].name)

    for local_name, item_type in concepts_map.items():
        key = (tax_prefix, local_name)
        if key in concept_cache:
            continue

        cid = _insert_extension_concept(
            session,
            taxonomy_prefix=tax_prefix,
            taxonomy_version=tax_version,
            local_name=local_name,
            std_label_en=labels_en.get(local_name),
            std_label_ja=labels_ja.get(local_name),
            item_type=item_type,
        )
        if cid:
            concept_cache[key] = cid
            new_concepts.append(cid)

    return new_concepts 