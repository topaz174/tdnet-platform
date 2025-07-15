#!/usr/bin/env python3
"""
Load facts from XBRL filings into financial_facts table.

This loader:
1. Processes all filing sections (Summary + Attachments)
2. Extracts company-specific extension taxonomies from .xsd/.xml files
3. Uses concepts table as authoritative dictionary for concepts
4. Dynamically creates context_dims entries for encountered contexts (except rejected patterns)
5. Inserts extension concepts and contexts before processing facts
"""

import os
import json
import zipfile
import re
import logging
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from zipfile import Path as ZipPath
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple, Set, Any
import sys
from datetime import datetime, date
from dateutil.parser import parse as dt_parse

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.config import DB_URL

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    logger.warning("lxml not installed. XBRL parsing will not work. `pip install lxml`")

# Configuration
MAX_FILINGS_TO_PROCESS = 99999

# Setup logging
logs_dir = root_dir / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"failed_facts_loading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Only log errors

# Create formatters
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
simple_formatter = logging.Formatter('%(message)s')

# File handler (detailed)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.ERROR)  # Only log errors
file_handler.setFormatter(detailed_formatter)

# Console handler (simple, only ERROR and above)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)  # Only show errors on console
console_handler.setFormatter(simple_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent duplicate logging
logger.propagate = False


@contextmanager
def open_package(pkg_path: Path):
    """
    Yield a Path-like root that allows `/` joins for both cases:
        - directory          → the actual Path object
        - zip archive        → a ZipPath object (zipfile.Path)
    """
    if pkg_path.suffix.lower() == '.zip':
        zf = ZipFile(pkg_path, 'r')
        try:
            yield ZipPath(zf)            # behaves like pathlib within the zip
        finally:
            zf.close()
    else:
        yield pkg_path                   # already a directory


def _iter_package_files(pkg_root):
    """Yield all files (not directories) under pkg_root for both Path and ZipPath."""
    try:
        # pathlib.Path supports rglob
        yield from pkg_root.rglob('*')  # type: ignore[attr-defined]
    except AttributeError:
        # ZipPath on Python <3.11 lacks rglob; manual DFS using iterdir
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


class FactsLoader:
    """Loads financial facts from XBRL files into the financial_facts table."""
    
    def __init__(self):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.concept_cache = {}  # (taxonomy_prefix, local_name) -> concept_id
        self.processed_contexts = set()  # Set of context_ids we've already processed/inserted
        self._load_lookup_caches()
        
    def _load_lookup_caches(self):
        """Load concept lookup cache from database."""
        session = self.Session()
        try:
            # Load concepts mapping
            concepts_query = text("SELECT id, taxonomy_prefix, local_name FROM concepts")
            for row in session.execute(concepts_query):
                key = (row.taxonomy_prefix, row.local_name)
                self.concept_cache[key] = row.id
            
            # Load existing context IDs to avoid duplicates
            contexts_query = text("SELECT DISTINCT context_id FROM context_dims")
            for row in session.execute(contexts_query):
                self.processed_contexts.add(row.context_id)
                
            print(f"Loaded {len(self.concept_cache)} concepts, {len(self.processed_contexts)} existing contexts")
            
        finally:
            session.close()

    def _should_reject_context(self, context_ref: str) -> bool:
        """Check if context should be rejected based on never-want rules."""
        # Reject anything containing "Prior" as it's redundant with already scraped info
        if "Prior" in context_ref:
            return True
        
        # Add other never-want rules here as needed
        # For example, if there are other patterns to avoid
        
        return False

    def _parse_context_components(self, context_id: str) -> Dict[str, Any]:
        """Parse context_id to extract components for context_dims table."""
        # Default values
        components = {
            'context_id': context_id,
            'period_base': None,
            'period_type': None, 
            'fiscal_span': None,
            'consolidated': None,
            'forecast_variant': None
        }
        
        # Extract period_type (Instant or Duration)
        if 'Instant' in context_id:
            components['period_type'] = 'Instant'
            base_part = context_id.replace('Instant', '')
        elif 'Duration' in context_id:
            components['period_type'] = 'Duration'
            base_part = context_id.replace('Duration', '')
        else:
            # If no period type found, assume Duration as default
            components['period_type'] = 'Duration'
            base_part = context_id
        
        # Split by underscore to get main parts and members
        parts = context_id.split('_')
        
        # First part contains the period information
        if parts:
            period_part = parts[0]
            
            # Extract period_base (everything before Instant/Duration)
            if 'Instant' in period_part:
                period_token = period_part.replace('Instant', '')
            elif 'Duration' in period_part:
                period_token = period_part.replace('Duration', '')
            else:
                period_token = period_part
        
        # Extract period_base (just the first word) and fiscal_span
        if period_token:
            # Split by camelCase or word boundaries to get the first word
            words = re.findall(r'[A-Z][a-z]*', period_token)
            if words:
                components['period_base'] = words[0]
            else:
                # Fallback: take first word if no camelCase found
                components['period_base'] = period_token.split()[0] if ' ' in period_token else period_token
            
            # Extract fiscal_span from period_token
            if 'YTD' in period_token:
                components['fiscal_span'] = 0  # Year-to-Date cumulative
            elif 'Year' in period_token:
                components['fiscal_span'] = 99
            elif 'Q1' in period_token:
                components['fiscal_span'] = 1
            elif 'Q2' in period_token or 'HalfYear' in period_token:
                components['fiscal_span'] = 2
            elif 'Q3' in period_token:
                components['fiscal_span'] = 3
        
        # Extract consolidated status from members
        for part in parts:
            if 'NonConsolidated' in part:
                components['consolidated'] = False
            elif 'Consolidated' in part:
                components['consolidated'] = True
        
        # Extract forecast_variant from members
        for part in parts:
            if 'ResultMember' in part:
                components['forecast_variant'] = 'Result'
            elif 'ForecastMember' in part:
                components['forecast_variant'] = 'Forecast'
            elif 'UpperMember' in part:
                components['forecast_variant'] = 'Upper'
            elif 'LowerMember' in part:
                components['forecast_variant'] = 'Lower'
        
        return components

    def _ensure_context_exists(self, context_ref: str, session) -> bool:
        """Ensure context exists in context_dims table, creating if necessary."""
        # Skip if already processed
        if context_ref in self.processed_contexts:
            return True
        
        # Check never-want rules
        if self._should_reject_context(context_ref):
            return False
        
        try:
            # Parse context components
            components = self._parse_context_components(context_ref)
            
            # Insert into context_dims with ON CONFLICT DO NOTHING
            session.execute(text("""
                INSERT INTO context_dims (
                    context_id, period_base, period_type, fiscal_span,
                    consolidated, forecast_variant
                ) VALUES (
                    :context_id, :period_base, :period_type, :fiscal_span,
                    :consolidated, :forecast_variant
                ) ON CONFLICT (context_id) DO NOTHING
            """), components)
            
            # Mark as processed
            self.processed_contexts.add(context_ref)
            return True
            
        except Exception as e:
            logger.error(f"Could not ensure context {context_ref}: {e}")
            session.rollback()  # Clear the failed transaction state
            return False

    def load_facts(self, max_count: int = MAX_FILINGS_TO_PROCESS, company_code: str = None) -> int:
        """Load financial facts from XBRL filings."""
        session = self.Session()
        total_facts_loaded = 0
        error_count = 0
        filings_processed = 0
        
        try:
            # Fetch distinct filings (not sections)
            filings_query = """
                SELECT 
                    f.id           AS filing_id,
                    d.company_code,
                    d.title,
                    d.disclosure_date,
                    d.time
                FROM xbrl_filings f
                JOIN disclosures d ON f.disclosure_id = d.id
                WHERE d.has_xbrl = true
            """

            if company_code:
                filings_query += " AND d.company_code = :company_code"
            filings_query += " ORDER BY d.disclosure_date DESC, d.time DESC LIMIT :max_count"

            params = {'max_count': max_count}
            if company_code:
                params['company_code'] = company_code

            filings = session.execute(text(filings_query), params).fetchall()
            print(f"Found {len(filings)} filings to process")

            for idx, filing in enumerate(filings, 1):
                print(f"\n[{idx}/{len(filings)}] Processing filing {filing.company_code}: {filing.title}")

                try:
                    # Derive XBRL path
                    from src.utils.path_derivation import derive_xbrl_path
                    pkg_path = Path(derive_xbrl_path(
                        filing.company_code,
                        filing.disclosure_date,
                        filing.time,
                        filing.title
                    ))

                    # Open package (directory or zip) using streaming approach
                    with open_package(pkg_path) as pkg_root:
                        # Process extension taxonomy once per filing
                        self._process_extension_taxonomy(pkg_root, filing.company_code, session)

                        # Fetch sections for this filing (skip Narrative)
                        sections = session.execute(text(
                            "SELECT id, statement_role_ja, rel_path FROM filing_sections WHERE filing_id = :fid ORDER BY statement_role_ja"),
                            {'fid': filing.filing_id}
                        ).fetchall()

                        filing_fact_count = 0

                        for sec in sections:
                            if sec.statement_role_ja == 'ナラティブ':
                                continue  # skip narrative sections entirely
                            
                            # Build path from rel_path - works for both ZipPath and Path
                            rel_norm = sec.rel_path.replace('\\', '/')
                            section_file = pkg_root.joinpath(*rel_norm.split('/'))
                            
                            try:
                                with section_file.open('rb') as fh:
                                    content = fh.read()
                                    fh.seek(0)  # Reset to beginning
                                    facts, error_type = self._extract_facts_from_file_handle(
                                        fh,
                                        sec.id,
                                        session
                                    )
                            except FileNotFoundError as e:
                                logger.error(f"File not found for section {sec.statement_role_ja}: {e}")
                                error_count += 1
                                continue
                            except Exception as e:
                                logger.error(f"Exception processing section {sec.statement_role_ja}: {e}")
                                error_count += 1
                                continue
                            
                            if not facts:
                                if error_type != "ERR_NO_FACTS_FOUND":
                                    logger.error(f"No facts found for section {sec.statement_role_ja}: {error_type}")
                                    error_count += 1
                                continue

                            inserted = self._insert_facts(session, facts)
                            filing_fact_count += inserted

                    session.commit()
                    print(f"  ✓ Loaded {filing_fact_count} facts from filing")
                    total_facts_loaded += filing_fact_count
                    filings_processed += 1

                except Exception as e:
                    error_count += 1
                    logger.error(f"ERROR (filing skipped): {e}")
                    session.rollback()
                    continue

        except Exception as e:
            logger.error(f"Database error: {e}")
            session.rollback()
        finally:
            session.close()

        print(f"\nLoad complete: {filings_processed} filings, {total_facts_loaded} facts, {error_count} section errors/skips")
        return total_facts_loaded

    def test_file_path(self, file_path: str, section_role: str = None) -> None:
        """Test processing a specific file path and output results in readable format."""
        pkg_path = Path(file_path)
        
        if not pkg_path.exists():
            print(f"ERROR: File path does not exist: {pkg_path}")
            return
        
        print(f"Testing file: {pkg_path}")
        print("=" * 80)
        
        # Create a mock session for testing
        session = self.Session()
        try:
            # Open package (directory or zip) using streaming approach
            with open_package(pkg_path) as pkg_root:
                # Process extension taxonomy
                print("Processing extension taxonomy...")
                extension_concepts = self._process_extension_taxonomy(pkg_root, "TEST", session)
                print(f"Added {len(extension_concepts)} extension concepts")
            
                # Find all HTML files in the package
                html_files = []
                for file_path in _iter_package_files(pkg_root):
                    if file_path.name.lower().endswith(('.htm', '.html')):
                        html_files.append(file_path)
                
                print(f"\nFound {len(html_files)} HTML files:")
                for i, file_path in enumerate(html_files, 1):
                    print(f"  {i}. {file_path.name}")
            
                # Process each HTML file
                total_facts = 0
                for i, file_path in enumerate(html_files, 1):
                    print(f"\n--- Processing file {i}/{len(html_files)}: {file_path.name} ---")
                    
                    # Determine section role if not specified
                    current_section_role = section_role
                    if not current_section_role:
                        if 'summary' in str(file_path).lower():
                            current_section_role = 'Summary'
                        else:
                            current_section_role = 'Attachment'
                    
                    print(f"Section role: {current_section_role}")

                    try:
                        with file_path.open('rb') as fh:
                            content = fh.read()
                            facts, error_type = self._parse_xbrl_content(content, 0, session)

                            if not facts:
                                print(f"  No facts found. Error: {error_type}")
                                continue

                            print(f"  Found {len(facts)} facts:")
                            print()

                            # Print header
                            print(f"{'Concept':<40} {'Context':<30} {'Value':<15} {'Unit':<10}")
                            print("-" * 95)

                            # Print each fact
                            for fact in facts[:20]:  # Limit to first 20 for readability
                                # Get concept info
                                concept_info = session.execute(
                                    text("SELECT taxonomy_prefix, local_name, std_label_en FROM concepts WHERE id = :id"),
                                    {'id': fact['concept_id']}
                                ).fetchone()

                                concept_name = f"{concept_info.taxonomy_prefix}:{concept_info.local_name}" if concept_info else "Unknown"
                                concept_label = concept_info.std_label_en if concept_info and concept_info.std_label_en else concept_name

                                # Get unit info
                                unit_info = session.execute(
                                    text("SELECT unit_code FROM units WHERE id = :id"),
                                    {'id': fact['unit_id']}
                                ).fetchone()

                                unit_code = unit_info.unit_code if unit_info else "Unknown"

                                # Format value
                                value_str = f"{fact['value']:,.0f}" if isinstance(fact['value'], (int, float)) else str(fact['value'])

                                print(f"{concept_label:<40} {fact['context_id']:<30} {value_str:<15} {unit_code:<10}")

                            if len(facts) > 20:
                                print(f"  ... and {len(facts) - 20} more facts")

                            total_facts += len(facts)

                    except Exception as e:
                        print(f"  ERROR processing file: {e}")
                        continue

                print(f"\n" + "=" * 80)
                print(f"TOTAL: {total_facts} facts found across {len(html_files)} files")
            
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            session.close()

    def _process_extension_taxonomy(self, pkg_root, company_code: str, session) -> List[int]:
        """Process company extension taxonomy and add new concepts to database."""
        from src.etl.load_concepts import process_extension_taxonomy
        return process_extension_taxonomy(pkg_root, company_code, session, self.concept_cache)

    def _extract_facts_from_file_handle(self, file_handle, section_id: int, session) -> Tuple[List[Dict], str]:
        """Extract facts from a file handle (works with both regular files and zip contents)."""
        if not HAS_LXML:
            return [], "ERR_NO_LXML"
        
        try:
            content = file_handle.read()
            return self._parse_xbrl_content(content, section_id, session)
        except Exception:
            return [], "ERR_GENERAL_EXCEPTION"

    def _parse_xbrl_content(self, content: bytes, section_id: int, session) -> Tuple[List[Dict], str]:
        """Parse XBRL content and extract facts."""
        try:
            try:
                root = etree.fromstring(content, etree.XMLParser(recover=True))
            except Exception:
                return [], "ERR_XML_PARSE_FAILED"

            # Some inline-XBRL generators (e.g. certain Sony filings) output lowercase
            # "contextref" instead of the spec-standard camel-case "contextRef".
            # Make the search and extraction case-agnostic so we don't miss those facts.
            elements_with_context = root.xpath(".//*[@contextRef] | .//*[@contextref]")
            
            if not elements_with_context:
                return [], "ERR_NO_CONTEXT_ELEMENTS"

            facts: List[Dict] = []
            context_matches = 0
            concept_matches = 0
            
            for elem in elements_with_context:
                context_ref = elem.get('contextRef') or elem.get('contextref') or ''
                
                # Ensure context exists in context_dims table
                if not self._ensure_context_exists(context_ref, session):
                    continue
                context_matches += 1

                name_attr = elem.get('name', '')
                if ':' in name_attr:
                    taxonomy_prefix, local_name = name_attr.split(':', 1)
                else:
                    # Fallback: derive from element tag if no name attribute (e.g. WizLabo output)
                    if not name_attr:
                        # Element tag is of the form '{namespace}LocalName' – lxml keeps prefix separately
                        local_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag.split(':')[-1]
                        taxonomy_prefix = elem.prefix or 'unknown'
                    else:
                        # Handle cases without namespace prefix
                        taxonomy_prefix = 'unknown'
                        local_name = name_attr

                # Find concept in our cache
                concept_key = (taxonomy_prefix, local_name)
                concept_id = self.concept_cache.get(concept_key)
                
                if not concept_id:
                    # Try common taxonomy prefixes
                    for prefix in ['jppfs', 'jpcrp', 'jpdei']:
                        alt_key = (prefix, local_name)
                        if alt_key in self.concept_cache:
                            concept_id = self.concept_cache[alt_key]
                            break
                
                # NEW: handle prefixes that end with "_cor" (e.g. jpigp_cor → jpigp)
                if not concept_id and taxonomy_prefix.endswith('_cor'):
                    alt_prefix = taxonomy_prefix[:-4]  # strip the suffix
                    alt_key = (alt_prefix, local_name)
                    concept_id = self.concept_cache.get(alt_key)
                
                if not concept_id:
                    logger.warning(f"Concept not found in cache: {taxonomy_prefix}:{local_name}")
                    continue  # Skip unknown concepts
                concept_matches += 1

                if elem.text is None:
                    continue

                value_str = elem.text.strip().replace(',', '').replace('△', '')
                if not value_str or value_str in ['－', '―']:
                    continue

                try:
                    value = float(value_str)
                    if elem.get('sign') == '-' or '△' in elem.text:
                        value = -value
                except (ValueError, TypeError):
                    continue

                unit_ref = elem.get('unitRef', '')
                scale_attr = elem.get('scale', '0')
                try:
                    scale = int(scale_attr)
                except (ValueError, TypeError):
                    scale = 0
                unit_id = self._map_unit_ref_to_unit_id(unit_ref, scale)

                fact = {
                    'section_id': section_id,
                    'concept_id': concept_id,
                    'context_id': context_ref,
                    'unit_id': unit_id,
                    'value': value
                }

                facts.append(fact)

            if not facts:
                return [], "ERR_NO_FACTS_FOUND"

            return facts, "SUCCESS"

        except Exception:
            return [], "ERR_PARSE_EXCEPTION"

    def _map_unit_ref_to_unit_id(self, unit_ref: str, scale: int = 0) -> int:
        """Map XBRL unitRef and scale to database unit_id."""
        unit_ref_lower = unit_ref.lower()
        
        # Handle JPY with scale
        if 'jpy' in unit_ref_lower:
            if scale == 6:
                return 1  # JPY_Mil
            elif scale == 3:
                return 3  # JPY_Thou
            else:
                return 2  # JPY
        elif 'usd' in unit_ref_lower:
            return 4  # USD_Mil
        elif 'shares' in unit_ref_lower or unit_ref == 'Shares':
            return 5  # Shares
        elif 'pure' in unit_ref_lower or unit_ref == 'Pure':
            return 6  # Pure
        elif unit_ref == '' and scale > 0:
            if scale == 6:
                return 1  # JPY_Mil
            elif scale == 3:
                return 3  # JPY_Thou
            else:
                return 2  # JPY
        
        # Default based on scale
        if scale == 6:
            return 1  # JPY_Mil
        elif scale == 3:
            return 3  # JPY_Thou
        else:
            return 2  # JPY

    def _insert_facts(self, session, facts: List[Dict]) -> int:
        """Insert facts into database with per-fact commit and rollback on error."""
        if not facts:
            return 0
        
        inserted_count = 0
        
        for fact in facts:
            try:
                # Check for duplicates quickly via query
                existing_row = session.execute(text("""
                    SELECT 1 FROM financial_facts 
                    WHERE section_id = :section_id 
                      AND concept_id = :concept_id
                      AND context_id = :context_id
                    LIMIT 1
                """), {
                    'section_id': fact['section_id'],
                    'concept_id': fact['concept_id'],
                    'context_id': fact['context_id']
                }).fetchone()
                if existing_row:
                    continue
                
                # Insert fact
                session.execute(text("""
                    INSERT INTO financial_facts (
                        section_id, concept_id, context_id, unit_id, value
                    ) VALUES (
                        :section_id, :concept_id, :context_id, :unit_id, :value
                    )
                """), {
                    'section_id': fact['section_id'],
                    'concept_id': fact['concept_id'],
                    'context_id': fact['context_id'],
                    'unit_id': fact['unit_id'],
                    'value': fact['value']
                })
                session.commit()  # commit per fact to isolate errors
                inserted_count += 1
            except Exception as e:
                logger.error(f"Could not insert fact: {e}")
                session.rollback()  # clear failed state so subsequent queries work
                continue
        
        return inserted_count


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load facts from XBRL filings")
    parser.add_argument("--company", help="Restrict to specific company_code", default=None)
    parser.add_argument("--limit", type=int, help="Max filing sections to process", default=MAX_FILINGS_TO_PROCESS)
    parser.add_argument("--test-file", help="Test a specific file path without inserting into database", default=None)
    parser.add_argument("--section-role", help="Section role for test file (Summary/Attachment)", default=None)
    args = parser.parse_args()
    
    loader = FactsLoader()
    
    if args.test_file:
        # Test mode
        print(f"TEST MODE: Processing file {args.test_file}")
        loader.test_file_path(args.test_file, args.section_role)
    else:
        # Normal loading mode
        print(f"Starting facts loading process...")
        print(f"Maximum filing sections to process: {args.limit}")
        
        if args.company:
            print(f"Company filter: {args.company}")
        
        loaded_count = loader.load_facts(max_count=args.limit, company_code=args.company)
        print(f"\nFacts loading completed. Total loaded: {loaded_count}")


if __name__ == "__main__":
    main()