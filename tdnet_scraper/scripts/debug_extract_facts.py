#!/usr/bin/env python3
"""
Debug helper – run the same extraction code that load_facts.py uses but
for two manually-specified XBRL files (the one that failed and a control that
works).  The script prints step-by-step statistics so you can see where the
pipeline diverges.

Edit the TEST_CASES list at the top if you want to try other files.
"""

from __future__ import annotations
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

from lxml import etree

# ---------------------------------------------------------------------------
#  Configuration – add / change paths here
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "label": "problem_case",
        "company_code": "25930",
        "fiscal_quarter": 4,  # Q4 / full-year
        "zip_path": r"F:\\TEMP\\tdnet_xbrls\\2025-06-02\\15-30_25930_2025年4月期決算短信[日本基準]（連結）.zip",
    },
    {
        "label": "control_case",
        "company_code": "31720",
        "fiscal_quarter": 3,  # Q3
        "zip_path": r"F:\\TEMP\\tdnet_xbrls\\<ADD-PATH-HERE>",  # <-- put a known good file here
    },
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTEXTS_JSON = PROJECT_ROOT / "src" / "etl" / "contexts.json"
CONCEPTS_JSON = PROJECT_ROOT / "src" / "xbrl_parser" / "concepts.json"

# ---------------------------------------------------------------------------
#  Helper functions (taken from load_facts.py but without DB dependency)
# ---------------------------------------------------------------------------

def load_context_patterns() -> Dict:
    with open(CONTEXTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def get_context_patterns_for_quarter(quarter: int) -> List[str]:
    data = load_context_patterns()
    quarter_key = f"q{quarter}" if quarter < 4 else "annual"
    section = data.get(quarter_key, {})
    return section.get("duration", []) + section.get("instant", [])

def load_legacy_taxonomy() -> Dict[str, list]:
    try:
        with open(CONCEPTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[WARN] concepts.json not found – taxonomy empty")
        return {}

TAXONOMY = load_legacy_taxonomy()

# ---------------------------------------------------------------------------
#  Core debug routine
# ---------------------------------------------------------------------------

def find_summary_file(z: zipfile.ZipFile) -> str | None:
    """Return first entry that looks like the Summary iXBRL file."""
    for name in z.namelist():
        if "Summary" in name and name.lower().endswith(".htm"):
            return name
    return None

def extract_facts(stats: Dict, xbrl_path: Path, fiscal_quarter: int) -> None:
    """Replicate the extraction pipeline and fill statistics dict."""
    if not xbrl_path.exists():
        stats["status"] = "zip_not_found"
        return

    with zipfile.ZipFile(xbrl_path, "r") as z:
        # try folder vs zip root
        summary_name = find_summary_file(z)
        if not summary_name:
            stats["status"] = "summary_not_found"
            return
        html_bytes = z.read(summary_name)

    stats["summary_entry"] = summary_name

    root = etree.fromstring(html_bytes, etree.XMLParser(recover=True))
    elems = root.xpath(".//*[@contextRef]")
    stats["elem_with_ctx"] = len(elems)

    # ------------------------------------------------------------------
    #  Context filtering (duration / instant patterns)
    # ------------------------------------------------------------------
    ctx_allowed_patterns = get_context_patterns_for_quarter(fiscal_quarter)
    allowed_elems_ctx: List[etree.Element] = []
    for e in elems:
        ctx = e.get("contextRef", "")
        if any(pat in ctx for pat in ctx_allowed_patterns):
            allowed_elems_ctx.append(e)
    stats["after_ctx_filter"] = len(allowed_elems_ctx)

    # quick sample of contexts kept / dropped
    stats["sample_ctx_kept"] = list({e.get("contextRef") for e in allowed_elems_ctx})[:10]
    dropped_ctx = [e.get("contextRef") for e in elems if e not in allowed_elems_ctx]
    stats["sample_ctx_dropped"] = list(dict.fromkeys(dropped_ctx))[:10]

    # ------------------------------------------------------------------
    #  Concept filtering (taxonomy mapping)
    # ------------------------------------------------------------------
    matched = 0
    unmatched_tags: List[str] = []

    for canonical_name, variations in TAXONOMY.items():
        for e in allowed_elems_ctx:
            name_attr = e.get("name", "")
            concept_tag = name_attr.split(":")[-1]
            if concept_tag in variations:
                matched += 1
            else:
                unmatched_tags.append(concept_tag)
    stats["facts_matched"] = matched
    stats["sample_unmatched_tags"] = list(dict.fromkeys(unmatched_tags))[:20]

    stats["status"] = "ok"

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    for case in TEST_CASES:
        label = case["label"]
        print("\n" + "=" * 80)
        print(f"DEBUG – {label}")
        print("=" * 80)

        stats: Dict = {}
        extract_facts(stats, Path(case["zip_path"]), case["fiscal_quarter"])

        for k, v in stats.items():
            print(f"{k:20}: {v}")

if __name__ == "__main__":
    main() 