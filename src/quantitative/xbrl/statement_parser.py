"""
parse_xbrl_statements.py
------------------------

Parse a directory of Japanese XBRL filings (EDINET / TDnet, JP-GAAP or IFRS),
extract granular income-statement, balance-sheet, and cash-flow data,
and build a history for a single company.

Author: 2025-06-08
Requires: lxml, pandas, python-dateutil, tqdm
"""

from __future__ import annotations

import os
import re
import zipfile
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple, Optional
from xml.etree import ElementTree as ET

import pandas as pd
from dateutil.parser import parse as dt_parse
from tqdm import tqdm

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    print("Warning: lxml not available, falling back to xml.etree")

# -----------------------------------------------------------------------------
# Config – adapt if you need finer control
# -----------------------------------------------------------------------------
#   1.   Possible XBRL "ticker/security-code" elements used in Japanese filings.
TICKER_ELEMENT_NAMES = [
    "SecurityCode",  # Most common in Japanese filings
    "TradingSymbol",
    "EntityCommonStockSharesOutstanding"
]

#   2.   Statement type classification based on period type and common patterns
#        We'll extract ALL numeric facts and classify by period type (instant vs duration)
def _classify_statement_by_period_and_concept(period_type: str, concept_name: str) -> str:
    """Classify financial statement type based on period type and concept patterns."""
    
    # Balance sheet items are typically instant (point-in-time)
    if period_type == "instant":
        return "balance"
    
    # Income and cash flow items are typically duration (period-based)
    elif period_type == "duration":
        # Cash flow statement patterns
        cash_flow_patterns = [
            "CashFlow", "CashAndEquivalents", "CashAndCash", 
            "OperatingActivities", "InvestingActivities", "FinancingActivities"
        ]
        
        if any(pattern in concept_name for pattern in cash_flow_patterns):
            return "cashflow"
        else:
            # Default duration items to income statement
            return "income"
    
    return "income"  # Default fallback

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _extract_local_name(tag: str) -> str:
    """Extract the local name from an XML tag (after the namespace)."""
    if '}' in tag:
        return tag.split('}')[1]
    elif ':' in tag:
        return tag.split(':')[1]
    return tag

def _parse_period_from_context(context_elem) -> Optional[Tuple[date, str]]:
    """Parse period information from an XBRL context element."""
    try:
        # Look for period element (try multiple approaches for different formats)
        period = None
        
        # Method 1: Direct period child
        if HAS_LXML:
            periods = context_elem.xpath(".//period | .//*[local-name()='period']")
        else:
            periods = list(context_elem.iter())
            periods = [elem for elem in periods if _extract_local_name(elem.tag) == 'period']
        
        if periods:
            period = periods[0]
        
        if period is None:
            return None
            
        # Check for instant vs duration
        if HAS_LXML:
            instants = period.xpath(".//instant | .//*[local-name()='instant']")
            end_dates = period.xpath(".//endDate | .//*[local-name()='endDate']")
        else:
            instants = [elem for elem in period.iter() if _extract_local_name(elem.tag) == 'instant']
            end_dates = [elem for elem in period.iter() if _extract_local_name(elem.tag) == 'endDate']
        
        if instants and instants[0].text:
            date_val = dt_parse(instants[0].text).date()
            return date_val, "instant"
            
        if end_dates and end_dates[0].text:
            date_val = dt_parse(end_dates[0].text).date()
            return date_val, "duration"
            
    except Exception:
        pass
    return None

def _extract_ticker_from_xml(root) -> Optional[str]:
    """Try to extract ticker/security code from XBRL XML."""
    for ticker_name in TICKER_ELEMENT_NAMES:
        # Search for elements with this local name
        if HAS_LXML:
            elements = root.xpath(f".//*[local-name()='{ticker_name}']")
        else:
            # Fallback for standard xml.etree (less efficient)
            elements = []
            for elem in root.iter():
                if _extract_local_name(elem.tag) == ticker_name:
                    elements.append(elem)
        
        for elem in elements:
            if elem.text and elem.text.strip():
                return elem.text.strip()
    return None

def _extract_xbrl_files_from_zip(zip_path: Path) -> List[Tuple[str, bytes]]:
    """Extract XBRL/XML files from a ZIP archive. Returns list of (filename, content) tuples."""
    xbrl_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                filename = file_info.filename
                # Prioritize files that likely contain actual financial data
                # Skip definition files (.def.xml) and focus on instance documents
                if (filename.lower().endswith('.xbrl') or 
                    ('-ixbrl.htm' in filename.lower()) or
                    (filename.lower().endswith('.xml') and 
                     '-def.xml' not in filename.lower() and
                     '-pre.xml' not in filename.lower() and
                     '-cal.xml' not in filename.lower() and
                     '-lab.xml' not in filename.lower())):
                    try:
                        content = zip_ref.read(filename)
                        xbrl_files.append((filename, content))
                    except Exception as e:
                        print(f"[WARN] Could not read {filename} from {zip_path.name}: {e}")
                        continue
    except Exception as e:
        print(f"[WARN] Could not open ZIP file {zip_path.name}: {e}")
    
    return xbrl_files

# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------
def parse_folder_for_company(folder: str | Path, ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Walk *folder*, parse each XBRL, filter by *ticker*,
    and return {'income': df_IS, 'balance': df_BS, 'cashflow': df_CF}.
    """
    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    print(f"Searching for ticker {ticker} in folder: {folder}")

    # Buffers: {statement -> list of records}
    # Each record: {"ticker": str, "period_end": date, "period_type": str, "concept": value, ...}
    buffers: Dict[str, List[Dict]] = {
        "income": [],
        "balance": [],
        "cashflow": [],
    }
    
    # Temporary buffers to collect facts by period
    period_facts: Dict[Tuple[date, str], Dict[str, float]] = defaultdict(dict)

    # First filter by filename (performance optimization)
    # Look for ZIP files containing the ticker in their name pattern: "time_ticker_description.zip"
    zip_files = []
    
    # Check if we have date-tagged subfolders (like "2017-05-12")
    has_date_subfolders = any(
        d.is_dir() and len(d.name) == 10 and d.name.count('-') == 2
        for d in folder.iterdir()
    )
    
    if has_date_subfolders:
        # Process date-tagged subfolders
        for date_folder in folder.iterdir():
            if date_folder.is_dir():
                for file_path in date_folder.iterdir():
                    if (file_path.suffix.lower() == ".zip" and
                        f"_{ticker}_" in file_path.name):  # ticker appears in filename
                        zip_files.append(file_path)
    else:
        # Fallback: search directly in root folder
        for file_path in folder.rglob("*"):
            if (file_path.suffix.lower() == ".zip" and
                f"_{ticker}_" in file_path.name):
                zip_files.append(file_path)
    
    print(f"Found {len(zip_files)} ZIP files matching ticker {ticker} in filename")
    
    for zip_fp in tqdm(zip_files, desc="Processing ZIP files"):
        # Extract XBRL files from ZIP
        xbrl_files = _extract_xbrl_files_from_zip(zip_fp)
        
        if not xbrl_files:
            print(f"[WARN] No XBRL files found in {zip_fp.name}")
            continue
            
        for filename, content in xbrl_files:
            try:
                # Parse XML content
                if HAS_LXML:
                    root = etree.fromstring(content)
                else:
                    root = ET.fromstring(content)
                    
            except Exception as exc:
                print(f"[WARN] {zip_fp.name}/{filename}: {exc}")
                continue
                
            # Since we already filtered by filename, we can be more lenient here
            # But still verify ticker if available in XBRL content
            doc_ticker = _extract_ticker_from_xml(root)
            if doc_ticker is not None and doc_ticker != str(ticker):
                print(f"[WARN] {zip_fp.name}/{filename}: filename suggests ticker {ticker} but XBRL contains {doc_ticker}")
                continue  # ticker mismatch

            # Build context map for period information
            context_map = {}
            
            # Find contexts - try multiple approaches for different XBRL formats
            contexts = []
            if HAS_LXML:
                # Try multiple XPath patterns for different namespace scenarios
                contexts = (root.xpath(".//*[local-name()='context']") or
                           root.xpath(".//xbrli:context", namespaces={'xbrli': 'http://www.xbrl.org/2003/instance'}) or
                           root.xpath(".//context"))
            else:
                # Fallback for xml.etree
                for elem in root.iter():
                    local_name = _extract_local_name(elem.tag)
                    if local_name == 'context':
                        contexts.append(elem)
                
            for ctx in contexts:
                ctx_id = ctx.get('id')
                if ctx_id:
                    period_info = _parse_period_from_context(ctx)
                    if period_info:
                        context_map[ctx_id] = period_info

            # Iterate all numeric facts
            if HAS_LXML:
                all_elements = root.xpath(".//*[@contextRef]")
            else:
                all_elements = [elem for elem in root.iter() if elem.get('contextRef')]
            
            for elem in all_elements:
                local_name = _extract_local_name(elem.tag)
                
                # Handle iXBRL format: look for 'name' attribute which contains the actual concept
                full_concept_name = local_name
                if elem.get('name'):  # iXBRL format
                    full_concept_name = elem.get('name')  # Keep full namespaced name
                
                # Skip non-financial data elements (these are usually metadata)
                if any(x in full_concept_name.lower() for x in ['textblock', 'abstract', 'title', 'document', 'filing']):
                    continue

                # Get period information from context
                context_ref = elem.get('contextRef')
                if context_ref not in context_map:
                    continue
                period_k = context_map[context_ref]

                # Numeric coercion - only extract numeric values
                try:
                    if elem.text and elem.text.strip():
                        # Handle different numeric formats
                        text_val = elem.text.strip().replace(',', '')
                        val = float(text_val)
                    else:
                        continue
                except (ValueError, AttributeError):
                    continue  # non-numeric

                # Store in temporary period-based buffer
                period_facts[period_k][full_concept_name] = val

    # Convert period-based facts to structured records
    for (period_end, period_type), facts in period_facts.items():
        if not facts:
            continue
            
        # Create base record for this period
        base_record = {
            "ticker": str(ticker),
            "period_end": period_end.strftime("%Y-%m-%d"),
            "period_type": period_type
        }
        
        # Split facts by statement type for duration periods
        if period_type == "instant":
            # All instant items go to balance sheet
            record = {**base_record, **facts}
            buffers["balance"].append(record)
        elif period_type == "duration":
            # Split duration facts into income and cashflow based on concept names
            income_facts = {}
            cashflow_facts = {}
            
            cash_flow_patterns = ["CashFlow", "CashAndEquivalents", "CashAndCash"]
            
            for concept, value in facts.items():
                if any(pattern in concept for pattern in cash_flow_patterns):
                    cashflow_facts[concept] = value
                else:
                    income_facts[concept] = value
            
            # Create separate records for income and cashflow if they have data
            if income_facts:
                income_record = {**base_record, **income_facts}
                buffers["income"].append(income_record)
                
            if cashflow_facts:
                cashflow_record = {**base_record, **cashflow_facts}
                buffers["cashflow"].append(cashflow_record)

    # Convert to DataFrames
    results = {}
    for stmt, records in buffers.items():
        if not records:
            results[stmt] = pd.DataFrame()
            continue
            
        df = pd.DataFrame(records)
        
        # Sort by period_end
        if not df.empty:
            df['period_end'] = pd.to_datetime(df['period_end'])
            df = df.sort_values('period_end').reset_index(drop=True)
            
        results[stmt] = df

    return results

# -----------------------------------------------------------------------------
# Convenience CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Parse Japanese XBRLs for one company")
    parser.add_argument("ticker", help="e.g. 7203 for Toyota")
    parser.add_argument("folder", help="Root folder containing XBRL files")
    parser.add_argument("-o", "--out", default=None,
                        help="Output path for pickle/json (writes three files)")
    args = parser.parse_args()

    data = parse_folder_for_company(args.folder, args.ticker)

    if args.out:
        opath = Path(args.out).expanduser()
        opath.mkdir(parents=True, exist_ok=True)
        for stmt, df in data.items():
            df.to_pickle(opath / f"{args.ticker}_{stmt}.pkl")
            df.reset_index().to_json(opath / f"{args.ticker}_{stmt}.json",
                                     orient="records", force_ascii=False, indent=2)
        print(f"Saved dataframes under {opath}")
    else:
        # Pretty print sizes
        for k, v in data.items():
            print(f"{k.title():9s}: {v.shape[0]} periods × {v.shape[1]} line-items")
