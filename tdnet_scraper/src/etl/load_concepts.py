#!/usr/bin/env python3
"""
Load concepts from xsd_elements_universal_with_labels.json into the concepts table.
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
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

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


def _parse_extension_xsd_content(content: bytes) -> Dict[str, str]:
    """Return mapping of local_name → item_type for concepts defined in an XSD (vendor-agnostic)."""
    concepts: Dict[str, str] = {}

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

            if name:
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
        logger.error(f"No extension XSD found for company {company_code}")
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


def load_concepts():
    """Load concepts from JSON file into the database."""
    
    # Get the path to the JSON file
    json_file_path = Path(__file__).parent / "xsd_elements_universal_with_labels.json"
    
    if not json_file_path.exists():
        print(f"Error: JSON file not found at {json_file_path}")
        return False
    
    # Load the JSON data
    print(f"Loading concepts from {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    
    # Prepare data for insertion
    concepts_to_insert = []
    for local_name, concept_info in concepts_data.items():
        concept_record = (
            concept_info.get('taxonomy'),      # taxonomy_prefix
            local_name,                        # local_name
            concept_info.get('label_en'),      # std_label_en
            concept_info.get('label_ja'),      # std_label_ja
            concept_info.get('item_type'),     # item_type
            concept_info.get('latest_version') # taxonomy_version
        )
        concepts_to_insert.append(concept_record)
    
    # Connect to database and insert data
    try:
        engine = create_engine(DB_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print(f"Inserting {len(concepts_to_insert)} concepts...")
        
        # Clear existing data
        session.execute(text("DELETE FROM concepts"))
        print("Cleared existing concepts")
        
        # Insert new concepts in batches
        inserted_count = 0
        for concept_record in concepts_to_insert:
            taxonomy_prefix, local_name, std_label_en, std_label_ja, item_type, taxonomy_version = concept_record
            
            result = session.execute(
                text("""
                    INSERT INTO concepts (taxonomy_prefix, local_name, std_label_en, std_label_ja, item_type, taxonomy_version)
                    VALUES (:taxonomy_prefix, :local_name, :std_label_en, :std_label_ja, :item_type, :taxonomy_version)
                    ON CONFLICT (taxonomy_prefix, local_name) DO UPDATE SET
                        std_label_en = EXCLUDED.std_label_en,
                        std_label_ja = EXCLUDED.std_label_ja,
                        item_type = EXCLUDED.item_type,
                        taxonomy_version = EXCLUDED.taxonomy_version
                    RETURNING id
                """),
                {
                    'taxonomy_prefix': taxonomy_prefix,
                    'local_name': local_name,
                    'std_label_en': std_label_en,
                    'std_label_ja': std_label_ja,
                    'item_type': item_type,
                    'taxonomy_version': taxonomy_version
                }
            )
            
            if result.rowcount > 0:
                inserted_count += 1
                if inserted_count % 1000 == 0:
                    print(f"Inserted {inserted_count} concepts...")
        
        # Commit changes
        session.commit()
        
        # Get count of total records
        count_result = session.execute(text("SELECT COUNT(*) FROM concepts"))
        count = count_result.scalar()
        
        print(f"Successfully loaded {count} concepts into the database")
        
        # Show some statistics
        standards_result = session.execute(text("""
            SELECT taxonomy_prefix, COUNT(*) 
            FROM concepts 
            GROUP BY taxonomy_prefix 
            ORDER BY taxonomy_prefix
        """))
        taxonomies = standards_result.fetchall()
        
        print("\nConcepts by taxonomy prefix:")
        for taxonomy_prefix, count in taxonomies:
            print(f"  {taxonomy_prefix}: {count}")
        
        item_types_result = session.execute(text("""
            SELECT item_type, COUNT(*) 
            FROM concepts 
            WHERE item_type IS NOT NULL
            GROUP BY item_type 
            ORDER BY COUNT(*) DESC
        """))
        item_types = item_types_result.fetchall()
        
        print("\nConcepts by item type:")
        for item_type, count in item_types:
            print(f"  {item_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error loading concepts: {e}")
        if 'session' in locals():
            session.rollback()
        return False
    
    finally:
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    success = load_concepts()
    sys.exit(0 if success else 1) 