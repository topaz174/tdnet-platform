#!/usr/bin/env python3
"""
Load XBRL filings from disclosures into the xbrl_filings table.

The schema for xbrl_filings table is as follows:
# CREATE TABLE IF NOT EXISTS xbrl_filings (
#     id                SERIAL PRIMARY KEY,
#     company_id        INTEGER NOT NULL REFERENCES companies(id),
#     disclosure_id     INTEGER NOT NULL REFERENCES disclosures(id),
#     period_start      DATE NOT NULL,
#     period_end        DATE NOT NULL,
#     fiscal_year       SMALLINT GENERATED ALWAYS AS (EXTRACT(YEAR FROM period_end)) STORED,
#     fiscal_quarter    SMALLINT,
#     default_unit_id   INTEGER REFERENCES units(id) DEFAULT 1,  -- assumes 'JPY_Mil'
#     facts_raw         JSONB,
#     taxonomy_version  TEXT,
#     created_at        TIMESTAMP DEFAULT now(),
#     UNIQUE (company_id, period_end)
# );

This script:
- reads the disclosures table and finds all disclosures with has_xbrl = true
- for each of those, it creates a new filing row in the xbrl_filings table
"""

import os
import re
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any
import sys
import json
from collections import OrderedDict
import argparse
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import unified config
from config.config import DB_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

try:
    from lxml import etree  # type: ignore
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    print("Warning: lxml not installed. Period detection will not work. `pip install lxml`")

from dateutil.parser import parse as dt_parse

# Import filing sections parser utility
from src.quantitative.etl.filing_sections_parser import discover_sections_from_path

class XBRLFilingLoader:
    """Loads XBRL filings from disclosures into the xbrl_filings table."""
    
    def __init__(self):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        
    def load_filings(self, company_code: str | None = None) -> int:
        """
        Load XBRL filings from disclosures table.
        
        Args:
            company_code (str | None): Restrict loading to specific company_code
            
        Returns:
            int: Number of filings successfully loaded
        """
        session = self.Session()
        loaded_count = 0
        error_count = 0
        
        try:
            # Query all disclosures with XBRL available (has_xbrl = true)
            base_query = """
                SELECT 
                    d.id as disclosure_id,
                    d.company_code,
                    d.company_name,
                    d.title,
                    d.disclosure_date,
                    d.time,
                    d.has_xbrl,
                    c.id as company_id
                FROM disclosures d
                LEFT JOIN companies c ON (
                    d.company_code = c.ticker OR 
                    (LENGTH(d.company_code) = 5 AND SUBSTRING(d.company_code, 1, 4) = c.ticker)
                )
                WHERE d.has_xbrl = true
            """

            if company_code:
                base_query += " AND d.company_code = :company_code"

            base_query += " ORDER BY d.disclosure_date DESC, d.time DESC"

            disclosures_query = text(base_query)
            
            params: Dict[str, Any] = {}
            if company_code:
                params['company_code'] = company_code
            
            result = session.execute(disclosures_query, params)
            disclosures = result.fetchall()
            
            print(f"Found {len(disclosures)} XBRL disclosures to process")
            
            failed_filings = []
            
            for i, disclosure in enumerate(disclosures, 1):
                print(f"\n[{i}/{len(disclosures)}] Processing {disclosure.company_code}/{disclosure.disclosure_id}: {disclosure.title}")
                
                try:
                    # Derive XBRL path from disclosure metadata
                    from src.shared.utils.path_derivation import derive_xbrl_path
                    xbrl_path = derive_xbrl_path(
                        disclosure.company_code,
                        disclosure.disclosure_date,
                        disclosure.time,
                        disclosure.title
                    )
                    
                    # Extract XBRL information
                    xbrl_info, info_err = self._extract_xbrl_info(Path(xbrl_path))
                    
                    if not xbrl_info:
                        error_reason = info_err or "ERR_EXTRACT_INFO"
                        print(f"  SKIP: {error_reason}")
                        failed_filings.append({
                            'disclosure_id': disclosure.disclosure_id,
                            'company_code': disclosure.company_code,
                            'xbrl_path': str(xbrl_path),
                            'error_reason': error_reason,
                            'error_type': 'SKIP'
                        })
                        error_count += 1
                        continue
                    
                    # Handle case where company_id is None (company not in companies table)
                    company_id = disclosure.company_id
                    if company_id is None:
                        error_reason = "ERR_COMPANY_NOT_FOUND"
                        print(f"  SKIP: {error_reason}")
                        failed_filings.append({
                            'disclosure_id': disclosure.disclosure_id,
                            'company_code': disclosure.company_code,
                            'xbrl_path': str(xbrl_path),
                            'error_reason': error_reason,
                            'error_type': 'SKIP'
                        })
                        error_count += 1
                        continue
                    
                    # ----------------------------------------------
                    # Always insert a new filing (duplicates permitted)
                    # ----------------------------------------------
                    insert_stmt = text("""
                        INSERT INTO xbrl_filings (
                            company_id, 
                            disclosure_id, 
                            period_start, 
                            period_end, 
                            accounting_standard,
                            has_consolidated,
                            industry_code,
                            period_type,
                            submission_no,
                            amendment_flag,
                            report_amendment_flag,
                            xbrl_amendment_flag,
                            fiscal_quarter,
                            parent_filing_id
                        ) VALUES (
                            :company_id, 
                            :disclosure_id, 
                            :period_start, 
                            :period_end, 
                            :accounting_standard,
                            :has_consolidated,
                            :industry_code,
                            :period_type,
                            :submission_no,
                            :amendment_flag,
                            :report_amendment_flag,
                            :xbrl_amendment_flag,
                            :fiscal_quarter,
                            :parent_filing_id
                        ) RETURNING id
                    """)
                    
                    fiscal_quarter = self._map_period_type_to_quarter(xbrl_info.get('period_type'))

                    row = session.execute(insert_stmt, {
                        'company_id': company_id,
                        'disclosure_id': disclosure.disclosure_id,
                        'period_start': xbrl_info['period_start'],
                        'period_end': xbrl_info['period_end'],
                        'accounting_standard': xbrl_info.get('accounting_standard'),
                        'has_consolidated': xbrl_info.get('consolidated_flag'),
                        'industry_code': xbrl_info.get('industry_code'),
                        'period_type': xbrl_info.get('period_type'),
                        'submission_no': xbrl_info.get('submission_no'),
                        'amendment_flag': xbrl_info.get('amendment_flag'),
                        'report_amendment_flag': xbrl_info.get('report_amendment_flag'),
                        'xbrl_amendment_flag': xbrl_info.get('xbrl_amendment_flag'),
                        'fiscal_quarter': fiscal_quarter,
                        'parent_filing_id': None,
                    }).fetchone()

                    filing_id = row.id if row else None
                    was_inserted = True
                    
                    # -------------------------------------------------
                    # Insert all discovered filing sections
                    # -------------------------------------------------
                    sections = discover_sections_from_path(Path(xbrl_path))

                    section_insert_query = text("""
                        INSERT INTO filing_sections (
                            filing_id, rel_path, period_prefix, 
                            consolidated, layout_code, statement_role_ja, statement_role_en
                        )
                        VALUES (
                            :filing_id, :rel_path, :period_prefix, 
                            :consolidated, :layout_code, :statement_role_ja, :statement_role_en
                        )
                        ON CONFLICT (filing_id, statement_role_ja, period_prefix, consolidated, layout_code) DO NOTHING
                    """)
                    
                    for role, rel_path, period_prefix, consolidated, layout_code, role_ja, role_en in sections:
                        session.execute(section_insert_query, {
                            'filing_id': filing_id,
                            'rel_path': rel_path,
                            'period_prefix': period_prefix,
                            'consolidated': consolidated,
                            'layout_code': layout_code,
                            'statement_role_ja': role_ja,
                            'statement_role_en': role_en
                        })
                    
                    # Commit this filing immediately so that a later rollback doesn't undo it
                    session.commit()

                    # -------------------------------------------------
                    # Post-processing: determine corrections/parent linkage
                    # -------------------------------------------------
                    try:
                        self._normalize_parent_and_amendments(
                            session=session,
                            company_id=company_id,
                            period_start=xbrl_info['period_start'],
                            period_end=xbrl_info['period_end'],
                            has_consolidated=xbrl_info.get('consolidated_flag'),
                            accounting_standard=xbrl_info.get('accounting_standard')
                        )
                    except Exception as corr_ex:
                        print(f"  WARN: Normalization failed – {corr_ex}")
                    
                    loaded_count += 1
                    
                    print(f"  ✓ Loaded filing and sections for {disclosure.company_code} Q{fiscal_quarter}-{xbrl_info['fiscal_year']}")
                        
                except Exception as e:
                    error_count += 1
                    print(f"  ERROR: {e}")
                    failed_filings.append({
                        'disclosure_id': disclosure.disclosure_id,
                        'company_code': disclosure.company_code,
                        'xbrl_path': str(xbrl_path) if 'xbrl_path' in locals() else '',
                        'error_reason': str(e),
                        'error_type': 'ERROR'
                    })
                    session.rollback()
                    continue
            
            # Final commit (should be no-op because we already commit per filing)
            session.commit()
            
        except Exception as e:
            print(f"Database error: {e}")
            session.rollback()
        finally:
            session.close()
        
        # Save failed filings log
        self._save_failed_filings_log(failed_filings)
        
        print(f"\n{'='*50}")
        print(f"Load complete:")
        print(f"  - Inserted new: {loaded_count} filings")
        print(f"  - Errors/Skipped: {error_count} filings")
        print(f"{'='*50}")
        
        return loaded_count
    
    def _extract_xbrl_info(self, xbrl_path: Path) -> tuple[Optional[Dict], Optional[str]]:
        """
        Extract fiscal year, quarter, and DEI metadata from the primary IXBRL attachment (no longer using the summary file).
        """
        if not HAS_LXML:
            return None, "ERR_NO_LXML"
        
        if not xbrl_path.exists():
            return None, "ERR_PATH_NOT_FOUND"
        
        try:
            # Handle both directory and zip file cases
            if xbrl_path.is_dir():
                primary_file = self._find_primary_ixbrl_file(xbrl_path)
                dei_data: Optional[Dict] = None
                if primary_file:
                    dei_data = self._parse_dei_file(primary_file)

                # If DEI parsing failed or missing critical dates, fall back to summary parsing
                if not dei_data or 'period_start' not in dei_data:
                    summary_file = self._find_summary_file(xbrl_path)
                    if summary_file:
                        summary_result = self._parse_summary_file(summary_file)
                        if summary_result:
                            # Merge: use DEI fields if any, else summary fields
                            if dei_data:
                                dei_data.update({k: v for k, v in summary_result.items() if k not in dei_data or dei_data[k] is None})
                                return dei_data, None
                            return summary_result, None
                    return None, "ERR_PARSE_FAILED"
                return dei_data, None
            else:
                # Handle zip file
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        with zipfile.ZipFile(xbrl_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        
                        primary_file = self._find_primary_ixbrl_file(Path(temp_dir))
                        dei_result: Optional[Dict] = None
                        if primary_file:
                            dei_result = self._parse_dei_file(primary_file)

                        if not dei_result or 'period_start' not in dei_result:
                            summary_file = self._find_summary_file(Path(temp_dir))
                            if summary_file:
                                summary_result = self._parse_summary_file(summary_file)
                                if summary_result:
                                    if dei_result:
                                        dei_result.update({k: v for k, v in summary_result.items() if k not in dei_result or dei_result[k] is None})
                                        return dei_result, None
                                    return summary_result, None
                            return None, "ERR_PARSE_FAILED"
                        return dei_result, None
                        
                    except (zipfile.BadZipFile, FileNotFoundError, PermissionError) as e:
                        return None, "ERR_ZIP_EXTRACT_FAILED"
        except Exception as e:
            return None, "ERR_GENERAL_EXCEPTION"
    
    def _find_summary_file(self, folder_path: Path) -> Optional[Path]:
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
    
    def _find_primary_ixbrl_file(self, folder_path: Path) -> Optional[Path]:
        """Return the IXBRL attachment with the lexicographically smallest filename (first attachment)."""
        ixbrl_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.htm', '.html')):
                    ixbrl_files.append(Path(root, file))

        if not ixbrl_files:
            return None

        # Sort lexicographically to pick the first attachment (lowest prefix number)
        ixbrl_files.sort(key=lambda p: p.name)
        return ixbrl_files[0]



    def _parse_dei_file(self, file_path: Path) -> Optional[Dict]:
        """Parse the primary IXBRL attachment and extract DEI metadata."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Parse with lxml (HTML parser for inline XBRL)
            root = etree.fromstring(content, etree.XMLParser(recover=True))
            
            ns_ix = {'ix': 'http://www.xbrl.org/2008/inlineXBRL'}

            tag_map = {
                'jpdei_cor:AccountingStandardsDEI': 'accounting_standard',
                'jpdei_cor:WhetherConsolidatedFinancialStatementsArePreparedDEI': 'consolidated_flag',
                'jpdei_cor:IndustryCodeWhenConsolidatedFinancialStatementsArePreparedInAccordanceWithIndustrySpecificRegulationsDEI': 'industry_code',
                'jpdei_cor:CurrentFiscalYearStartDateDEI': 'period_start',
                'jpdei_cor:CurrentPeriodEndDateDEI': 'period_end',
                'jpdei_cor:TypeOfCurrentPeriodDEI': 'period_type',
                'jpdei_cor:NumberOfSubmissionDEI': 'submission_no',
                'jpdei_cor:AmendmentFlagDEI': 'amendment_flag',
                'jpdei_cor:ReportAmendmentFlagDEI': 'report_amendment_flag',
                'jpdei_cor:XBRLAmendmentFlagDEI': 'xbrl_amendment_flag',
            }

            data: Dict[str, Optional[str]] = {}

            # Search for ix:nonNumeric and ix:nonFraction elements that hold DEI values
            nodes = root.xpath('.//ix:nonNumeric | .//ix:nonFraction', namespaces=ns_ix)
            for node in nodes:
                name_attr = node.get('name')
                if not name_attr:
                    continue

                if name_attr in tag_map and tag_map[name_attr] not in data:
                    value = (node.text or '').strip()
                    data[tag_map[name_attr]] = value

            # Convert specific field types - create a new dict with proper types
            result: Dict[str, Any] = {}
            for key, value in data.items():
                if key in ['period_start', 'period_end'] and value:
                    try:
                        result[key] = dt_parse(value).date()
                    except Exception:
                        result[key] = value
                elif key in ['consolidated_flag', 'amendment_flag', 'report_amendment_flag', 'xbrl_amendment_flag'] and value:
                    result[key] = str(value).lower() in {'true', '1', 'yes', 'y'}
                else:
                    result[key] = value

            # Add fiscal_year derived from period_end if available
            if 'period_end' in result and isinstance(result['period_end'], date):
                result['fiscal_year'] = result['period_end'].year

            # Derive fiscal_quarter from period_type for consistency
            result['fiscal_quarter'] = self._map_period_type_to_quarter(result.get('period_type'))

            return result
        except Exception as e:
            print(f"  ERROR: Exception parsing DEI file {file_path}: {e}")
            return None
    
    def _parse_summary_file(self, file_path: Path) -> Optional[Dict]:
        """Parse summary IXBRL to extract period information as fallback."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            root = etree.fromstring(content, etree.XMLParser(recover=True))

            # Find contexts
            contexts = root.xpath('.//ix:resources//xbrli:context', namespaces={
                'ix': 'http://www.xbrl.org/2008/inlineXBRL',
                'xbrli': 'http://www.xbrl.org/2003/instance'
            })
            if not contexts:
                return None
            
            primary_ctx = self._select_longest_duration_context(contexts)
            if primary_ctx is None:
                return None
            
            start_date = dt_parse(primary_ctx.xpath('.//xbrli:startDate', namespaces={'xbrli': 'http://www.xbrl.org/2003/instance'})[0].text).date()
            end_date = dt_parse(primary_ctx.xpath('.//xbrli:endDate', namespaces={'xbrli': 'http://www.xbrl.org/2003/instance'})[0].text).date()

            q_override = self._extract_quarter_from_header(root)
            if q_override is not None:
                fiscal_q = q_override
            else:
                # Infer from context id
                cid = primary_ctx.get('id', '').lower()
                if 'accumulatedq3' in cid:
                    fiscal_q = 3
                elif 'accumulatedq2' in cid:
                    fiscal_q = 2
                elif 'accumulatedq1' in cid:
                    fiscal_q = 1
                else:
                    fiscal_q = 99
            
            return {
                'period_start': start_date,
                'period_end': end_date,
                'fiscal_year': end_date.year,
                'period_type': f'Q{fiscal_q}' if fiscal_q <= 4 else 'Year',
                'fiscal_quarter': fiscal_q
            }
        except Exception as e:
            return None
    
    def _select_longest_duration_context(self, contexts):
        """Return context with longest duration (fallback heuristic)."""
        best = None
        best_duration = -1
        best_end_date = None
        for ctx in contexts:
            try:
                start = dt_parse(ctx.xpath('.//xbrli:startDate', namespaces={'xbrli': 'http://www.xbrl.org/2003/instance'})[0].text).date()
                end = dt_parse(ctx.xpath('.//xbrli:endDate', namespaces={'xbrli': 'http://www.xbrl.org/2003/instance'})[0].text).date()
                duration = (end - start).days
                if duration > best_duration or (duration == best_duration and (best_end_date is None or end > best_end_date)):
                    best_duration = duration
                    best_end_date = end
                    best = ctx
            except Exception:
                continue
        return best

    def _extract_quarter_from_header(self, root):
        """Extract quarter from ix:hidden header nodes."""
        try:
            nodes = root.xpath(
                ".//ix:hidden//*[@name and contains(@name, ':QuarterlyPeriod')]",
                namespaces={'ix': 'http://www.xbrl.org/2008/inlineXBRL'}
            )
            for node in nodes:
                txt = (node.text or '').strip()
                if txt.isdigit():
                    q = int(txt)
                    if 1 <= q <= 4:
                        return q
        except Exception:
            pass
        return None

    def _map_period_type_to_quarter(self, period_type: Optional[str]) -> int:
        """Convert period_type string to fiscal quarter integer (1-4, 99=Year)."""
        if not period_type:
            return 99
        pt = period_type.upper()
        if pt in {"Q1", "FIRSTQUARTER"}:
            return 1
        if pt in {"Q2", "SECONDQUARTER", "HALFYEAR"}:
            return 2
        if pt in {"Q3", "THIRDQUARTER"}:
            return 3
        return 99

    def _save_failed_filings_log(self, failed: list):
        """Save failed/ skipped filings to logs folder with timestamp."""
        if not failed:
            return
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        outfile = logs_dir / f"failed_xbrl_filings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write("Failed XBRL Filings Report\n" + "="*40 + "\n")
            for i, item in enumerate(failed, 1):
                f.write(f"{i}. {item['error_type']}: {item['error_reason']}\n")
                f.write(f"   Disclosure ID: {item['disclosure_id']}\n")
                f.write(f"   Company Code: {item['company_code']}\n")
                f.write(f"   XBRL Path: {item.get('xbrl_path', '')}\n")
                f.write("-"*30 + "\n")
        print(f"Failed filings log saved to {outfile}")



    # -------------------------------------------------
    # Correction normalization helpers
    # -------------------------------------------------

    def _normalize_parent_and_amendments(
        self,
        *,
        session,
        company_id: int,
        period_start: date,
        period_end: date,
        has_consolidated: Optional[bool],
        accounting_standard: Optional[str],
    ) -> None:
        """Identify amendments within a logical filing set and link them to a parent.

        A logical set is defined by (company_id, period_start, period_end, consolidated_flag, accounting_standard).

        Correction candidates are detected when either:
          • xbrl_filings.amendment_flag = TRUE (from IXBRL metadata), OR
          • the associated disclosure subcategory list includes the literal "Earnings Corrections" (case-insensitive)

        The earliest non-correction record (by id) in the set becomes the parent; every correction
        receives its id via parent_filing_id and has amendment_flag enforced TRUE.
        """

        # Fetch the candidate set
        query = text(
            """
            SELECT xf.id,
                   xf.amendment_flag,
                   xf.parent_filing_id,
                   EXISTS (
                       SELECT 1
                         FROM disclosure_labels dl
                         JOIN disclosure_subcategories ds ON ds.id = dl.subcat_id
                        WHERE dl.disclosure_id = xf.disclosure_id
                          AND LOWER(ds.name) = 'earnings corrections'
                   ) AS has_corr_label
              FROM xbrl_filings xf
             WHERE xf.company_id          = :company_id
               AND xf.period_start        = :period_start
               AND xf.period_end          = :period_end
               AND (xf.has_consolidated  IS NOT DISTINCT FROM :has_consolidated)
               AND (xf.accounting_standard IS NOT DISTINCT FROM :accounting_standard)
             ORDER BY xf.id
            """
        )

        rows = session.execute(
            query,
            {
                'company_id': company_id,
                'period_start': period_start,
                'period_end': period_end,
                'has_consolidated': has_consolidated,
                'accounting_standard': accounting_standard,
            },
        ).fetchall()

        processed = []  # [{'id': int, 'is_correction': bool}]
        for r in rows:
            is_corr = bool(r.amendment_flag) or bool(r.has_corr_label)
            processed.append({'id': r.id, 'is_correction': is_corr})

        # Choose parent: first non-correction if exists, otherwise None (no valid parent yet)
        parent_id = next((p['id'] for p in processed if not p['is_correction']), None)

        update_stmt = text(
            """
            UPDATE xbrl_filings
               SET amendment_flag   = :amendment_flag,
                   parent_filing_id = :parent_id
             WHERE id = :id
            """
        )

        for p in processed:
            session.execute(update_stmt, {
                'amendment_flag': p['is_correction'],
                'parent_id': (parent_id if p['is_correction'] and parent_id and p['id'] != parent_id else None),
                'id': p['id'],
            })

        session.commit()


def main():
    """Main function to load XBRL filings."""
    parser = argparse.ArgumentParser(description="Load XBRL filings into database")
    parser.add_argument("--company", help="Restrict loading to specific company_code", default=None)
    parser.add_argument("--limit", type=int, help="Max filings to process (deprecated, processes all)", default=None)
    args = parser.parse_args()

    print(f"Starting XBRL filing loading process...")
    if args.limit:
        print(f"Note: --limit {args.limit} ignored, processing all available filings")
    if args.company:
        print(f"Company filter: {args.company}")

    loader = XBRLFilingLoader()
    loaded_count = loader.load_filings(company_code=args.company)
    
    print(f"\nXBRL filing loading completed. Total loaded: {loaded_count}")


if __name__ == "__main__":
    main()