#!/usr/bin/env python3
"""
XBRL Qualitative Data Extraction Pipeline for Testing

Extract prose and small-table narrative from JPX `qualitative.htm` files 
inside TDnet XBRL zip packages with enhanced chunking and content categorization.

Usage:
    python xbrl_qualitative_extractor.py --xbrl-dir /path/to/xbrls --max-files 5
    python xbrl_qualitative_extractor.py --xbrl-file /path/to/single.zip

Dependencies: 
    pip install beautifulsoup4 lxml pandas jaconv tiktoken langdetect

Database Schema Note:
    When inserting chunks into the database, the 'vectorize' boolean field should be
    added as: ALTER TABLE disclosures ADD COLUMN vectorize BOOLEAN DEFAULT true;
    
    For idempotency, the 'disclosure_hash' field should be added:
    ALTER TABLE disclosures ADD COLUMN disclosure_hash VARCHAR(64);
    
    Re-ingestion workflow should be:
    1. Calculate SHA-256 hash of XBRL zip file
    2. DELETE FROM disclosures WHERE disclosure_hash = :hash 
    3. INSERT new chunks with same disclosure_hash
    
    Embedding jobs should then filter with: WHERE vectorize = true AND embedding IS NULL
    This prevents low-quality chunks (megatable slabs, duplicate blocks, etc.) from 
    being processed by the crontab embedding job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Generator, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import jaconv
    JACONV_AVAILABLE = True
except ImportError:
    print("Warning: jaconv not installed. Install with: pip install jaconv")
    JACONV_AVAILABLE = False
    sys.exit(1)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Initialize the tokenizer for o3 model for accurate token counting
    _tokenizer = tiktoken.encoding_for_model("o3")
except ImportError:
    print("Warning: tiktoken not installed. Token counts will be estimated. Install with: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False
    _tokenizer = None

try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not installed. Language will default to 'ja'. Install with: pip install langdetect")
    LANGDETECT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not installed. Install with: pip install pandas")
    PANDAS_AVAILABLE = False
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print("Warning: BeautifulSoup not installed. Install with: pip install beautifulsoup4 lxml")
    BS4_AVAILABLE = False
    sys.exit(1)

@dataclass
class XBRLChunk:
    disclosure_id: int
    chunk_index: int
    content: str
    content_type: str
    section_code: str
    heading_text: str
    char_length: int
    tokens: int
    vectorize: bool
    is_numeric: bool
    disclosure_hash: str
    source_file: str
    metadata: Dict[str, Any]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, bufsize: int = 131_072) -> str:
    """Return the SHA‑256 of a file (used for idempotency)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def count_tokens(text: str) -> int:
    """
    Count tokens in text using the o3 model tokenizer.
    
    Returns accurate token counts for cost estimation and batch logic.
    Ensures tokens <= character count to prevent downstream issues.
    """
    if not text:
        return 0
    
    char_count = len(text)
    
    if TIKTOKEN_AVAILABLE and _tokenizer:
        token_count = len(_tokenizer.encode(text))
        # Safety check: ensure tokens <= chars to prevent cost/batch logic issues
        # Special Unicode characters can produce more tokens than characters
        return min(token_count, char_count)
    else:
        # Fallback estimation: roughly 2.5 characters per token for mixed Japanese/English
        return max(1, char_count // 3)


def detect_language(text: str) -> str:
    """
    Detect the language of text content.
    
    Returns 'ja' for Japanese, 'en' for English, 'mixed' for mixed content,
    or 'ja' as fallback if detection fails.
    
    Args:
        text: The text content to analyze
        
    Returns:
        str: Language code ('ja', 'en', 'mixed', or 'ja' as fallback)
    """
    if not text or not text.strip():
        return 'ja'  # Default fallback
        
    if not LANGDETECT_AVAILABLE:
        return 'ja'  # Default fallback if langdetect not available
    
    try:
        # Clean text for better detection - remove excessive numbers and symbols
        # that can confuse language detection
        clean_text = re.sub(r'[0-9.,|-△▲\s]+', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Need minimum text length for reliable detection
        if len(clean_text) < 20:
            return 'ja'  # Default for short text
            
        detected_lang = detect(clean_text)
        
        # For financial documents, check for mixed content
        # Count Japanese characters (hiragana, katakana, kanji)
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or  # hiragana
                                                '\u30A0' <= c <= '\u30FF' or  # katakana
                                                '\u4E00' <= c <= '\u9FAF')     # kanji
        
        # Count English letters
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        
        total_alpha = japanese_chars + english_chars
        
        if total_alpha > 0:
            japanese_ratio = japanese_chars / total_alpha
            english_ratio = english_chars / total_alpha
            
            # Mixed content detection
            if japanese_ratio >= 0.3 and english_ratio >= 0.3:
                return 'mixed'
            elif japanese_ratio > 0.7:
                return 'ja'
            elif english_ratio > 0.7:
                return 'en'
        
        # Return detected language if not mixed
        return detected_lang if detected_lang in ['ja', 'en'] else 'ja'
        
    except (LangDetectException, Exception):
        # Fallback to Japanese for any detection errors
        return 'ja'


def is_numeric_content(content: str) -> bool:
    """
    Determine if content is primarily numeric/tabular data.
    
    Returns True for content that is predominantly numbers, financial data,
    and table structures rather than narrative text.
    
    Args:
        content: The text content to analyze
        
    Returns:
        bool: True if content is primarily numeric/tabular, False if narrative
    """
    if not content or len(content.strip()) < 20:
        return False
    
    # Count different types of characters
    total_chars = len(content)
    numeric_chars = sum(1 for c in content if c.isdigit())
    financial_chars = sum(1 for c in content if c in '.,|-△▲%')
    table_chars = sum(1 for c in content if c in '|:()[]')
    japanese_chars = sum(1 for c in content if '\u3040' <= c <= '\u309F' or 
                                            '\u30A0' <= c <= '\u30FF' or 
                                            '\u4E00' <= c <= '\u9FAF')
    
    # Calculate ratios
    numeric_ratio = numeric_chars / total_chars
    financial_ratio = financial_chars / total_chars
    table_ratio = table_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    
    # Classify as numeric if:
    # 1. Moderate numeric content (>15% digits)
    # 2. Financial symbols (>10% financial chars)
    # 3. Combined numeric+financial content is significant (>20%)
    # 4. Table structure present with low narrative content
    # 5. Very low Japanese narrative content (<25%) + any numeric content
    
    combined_numeric_financial = numeric_ratio + financial_ratio
    
    if numeric_ratio > 0.15:  # More than 15% digits
        return True
    
    if financial_ratio > 0.10:  # More than 10% financial symbols
        return True
        
    if combined_numeric_financial > 0.20:  # Combined numeric+financial > 20%
        return True
        
    if table_ratio > 0.05 and japanese_ratio < 0.50:  # Table structure with low narrative
        return True
        
    if japanese_ratio < 0.25 and combined_numeric_financial > 0.15:  # Low narrative, moderate numeric
        return True
    
    # Additional check: look for table-like patterns
    # Count lines that look like table rows (mostly numbers and separators)
    lines = content.split('\n')
    table_lines = 0
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) > 5:
            line_numeric_ratio = sum(1 for c in line_stripped if c.isdigit() or c in '.,|-△▲%|:()[]') / len(line_stripped)
            if line_numeric_ratio > 0.50:
                table_lines += 1
    
    if len(lines) > 1 and table_lines / len(lines) > 0.60:  # More than 60% of lines are table-like
        return True
    
    return False


def should_vectorize_chunk(content: str, content_type: str, section_code: str, char_length: int, tokens: int, is_numeric: bool = None) -> bool:
    """
    Determine if a chunk should be vectorized based on quality criteria.
    
    Returns False for chunks that would be flagged under issues #1-2-4:
    - Issue #1: Megatable slabs (mostly numeric, large chunks)
    - Issue #2: Duplicate heading blocks (detected during filtering)
    - Issue #4: Chunk size creep (oversized chunks, though this is now handled)
    
    Args:
        content: The chunk content text
        content_type: Categorized content type
        section_code: Section classification
        char_length: Length in characters
        tokens: Token count
        
    Returns:
        bool: True if chunk should be vectorized, False otherwise
    """
    # Rule 1: Skip numeric/tabular content (megatable slabs)
    # Use the improved is_numeric detection if available
    if is_numeric is not None and is_numeric:
        return False
    
    # Fallback to legacy numeric detection for compatibility
    if is_numeric is None and content:
        # Count numeric characters (digits, decimals, separators)
        numeric_chars = sum(1 for c in content if c.isdigit() or c in '.,|-△▲')
        numeric_ratio = numeric_chars / len(content)
        
        # More aggressive filtering for large chunks with high numeric content
        # Tightened thresholds based on feedback about digit-heavy chunks
        if ((char_length > 400 and numeric_ratio > 0.25) or  # Large chunks: >25% numeric
            (char_length > 200 and numeric_ratio > 0.35) or  # Medium chunks: >35% numeric  
            (char_length > 100 and numeric_ratio > 0.50) or  # Small chunks: >50% numeric
            (numeric_ratio > 0.60)):                         # Any chunk: >60% numeric
            return False
    
    # Rule 2: Skip very short fragments (likely continuation fragments)
    if char_length < 50 or tokens < 15:
        return False
        
    # Rule 3: Skip table-heavy sections that are primarily numeric
    numeric_sections = ['balance_sheet', 'cash_flow']
    if section_code in numeric_sections and content:
        # Be more restrictive for financial statement sections
        numeric_chars = sum(1 for c in content if c.isdigit() or c in '.,|-△▲')
        if len(content) > 0 and (numeric_chars / len(content)) > 0.25:
            return False
    
    # Rule 4: Skip chunks that are primarily table headers or labels
    if content and len(content.strip()) > 0:
        # Check for table-like patterns: mostly punctuation and separators
        table_chars = sum(1 for c in content if c in '|:0123456789.,()-△▲ \t')
        if (table_chars / len(content)) > 0.80:
            return False
    
    # Rule 5: Consider content type quality
    # Some content types are inherently less valuable for vectorization
    low_value_types = []  # We've improved categorization, so most types are now valuable
    if content_type in low_value_types:
        return False
        
    # Default: vectorize this chunk
    return True


def clean_text(text: str) -> str:
    """Unicode‑normalise & collapse whitespace with full-width punctuation mapping."""
    if not text:
        return text
    
    # Replace full-width spaces with ASCII spaces
    text = text.replace("\u3000", " ")
    
    # Use jaconv if available, otherwise basic normalization
    if JACONV_AVAILABLE:
        text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
    
    # Map remaining full-width punctuation to half-width for BM25 consistency
    zenkaku_to_hankaku = {
        "：": ":",   # colon
        "；": ";",   # semicolon  
        "／": "/",   # slash
        "（": "(",   # left paren
        "）": ")",   # right paren
        "［": "[",   # left bracket
        "］": "]",   # right bracket
        "｛": "{",   # left brace
        "｝": "}",   # right brace
        "「": '"',   # left quote
        "」": '"',   # right quote
        "『": "'",   # left double quote
        "』": "'",   # right double quote
        "．": ".",   # period
        "，": ",",   # comma
        "！": "!",   # exclamation
        "？": "?",   # question
        "＋": "+",   # plus
        "－": "-",   # minus
        "＝": "=",   # equals
        "＜": "<",   # less than
        "＞": ">",   # greater than
        "％": "%",   # percent
        "＆": "&",   # ampersand
        "＊": "*",   # asterisk
        "＃": "#",   # hash
        "＠": "@",   # at sign
        # Full-width enumeration characters
        "①": "1.",   # circled 1
        "②": "2.",   # circled 2
        "③": "3.",   # circled 3
        "④": "4.",   # circled 4
        "⑤": "5.",   # circled 5
        "⑥": "6.",   # circled 6
        "⑦": "7.",   # circled 7
        "⑧": "8.",   # circled 8
        "⑨": "9.",   # circled 9
        "⑩": "10.",  # circled 10
    }
    
    for zenkaku, hankaku in zenkaku_to_hankaku.items():
        text = text.replace(zenkaku, hankaku)
    
    # Remove row-number prefixes throughout the text (aggressive cleaning)
    # This is a multi-pass approach to catch all row-number patterns
    
    # Pass 1: Remove at word boundaries and after common separators
    text = re.sub(r'(\s|\||^)\d{1,2}:\s*', r'\1', text)
    
    # Pass 2: Remove when followed by Japanese characters (labels)
    text = re.sub(r'\d{1,2}:\s*(?=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF])', '', text)
    
    # Pass 3: Remove when followed by opening parenthesis (unit indicators)
    text = re.sub(r'\d{1,2}:\s*(?=\()', '', text)
    
    # Pass 4: Remove when followed by numbers (financial data)
    # But preserve years (4-digit numbers) and times
    text = re.sub(r'\d{1,2}:\s*(?=[-+]?\d{1,3}\s)', '', text)
    
    # Pass 5: VERY CAUTIOUS removal of leading isolated small numbers in table-heavy content
    # Only remove if: starts with small number + space + Japanese text + pipe symbols (clear table pattern)
    # This targets patterns like "1 項目名 | 値 | 値" but preserves "-84 Company..." (legitimate financial data)
    if '|' in text and text.count('|') > 2:  # Only in clearly tabular content
        # Remove small positive integers at start when followed by Japanese and table structure
        text = re.sub(r'^([1-9]|1[0-9])\s+(?=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF].*\|)', '', text)
    
    # Apply financial data cleaning (same as table cells)
    # Convert △ (zenkaku minus) → ASCII "-"
    text = text.replace("△", "-")
    
    # Remove digit group-separators (,) in financial numbers
    # Only remove commas between digits (thousand separators)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_table_cell(cell: str) -> str:
    """Apply the recommended clean-up rules to table cell content."""
    if not cell or pd.isna(cell):
        return ""
    
    cell = str(cell).strip()
    
    # 1. Strip leading "row-number:" pattern (^\d+:)
    cell = re.sub(r"^\d+:", "", cell)
    
    # 2. Replace "nan" or "-" with blank
    cell = cell.replace("nan", "").replace("-", "")
    
    # 3. Convert △ (zenkaku minus) → ASCII "-"
    cell = cell.replace("△", "-")
    
    # 4. Remove digit group-separators (,) after the minus fix
    # Only remove commas that are clearly thousand separators
    cell = re.sub(r"(?<=\d),(?=\d{3})", "", cell)
    
    # 5. Collapse multiple spaces
    cell = re.sub(r"\s+", " ", cell).strip()
    
    return cell

def flatten_tables(soup: BeautifulSoup) -> None:
    """Replace each <table> with pipe‑delimited *row* text."""
    if not PANDAS_AVAILABLE:
        # If pandas is not available, just extract text from tables
        for tbl in soup.find_all("table"):
            table_text = tbl.get_text(" | ", strip=True)
            # Apply basic cleaning even without pandas
            cleaned_text = clean_table_cell(table_text)
            tbl.replace_with(cleaned_text)
        return
    
    for tbl in soup.find_all("table"):
        try:
            # Fix the pandas warning by using StringIO
            from io import StringIO
            df = pd.read_html(StringIO(str(tbl)))[0]
            
            # Clean each cell and create formatted lines
            lines = []
            for _, row in df.iterrows():
                cleaned_cells = []
                for c, v in row.items():
                    # Clean column name and value
                    clean_col = clean_table_cell(str(c))
                    clean_val = clean_table_cell(str(v))
                    
                    # Only add non-empty cells
                    if clean_col and clean_val:
                        cleaned_cells.append(f"{clean_col}: {clean_val}")
                    elif clean_val:  # Value without meaningful column name
                        cleaned_cells.append(clean_val)
                
                if cleaned_cells:  # Only add non-empty rows
                    lines.append(" | ".join(cleaned_cells))
            
            # Replace table with cleaned content
            cleaned_table = "\n".join(lines) if lines else ""
            tbl.replace_with(cleaned_table)
            
        except (ValueError, ImportError):
            # Fallback to simple text extraction with cleaning
            table_text = tbl.get_text(" | ", strip=True)
            cleaned_text = clean_table_cell(table_text)
            tbl.replace_with(cleaned_text)


HEADING_MAP = {
    # Core performance and outlook
    "今後の見通し": "outlook",
    "重要な後発事象": "subsequent_event", 
    "配当方針": "capital_policy",
    "経営成績等の概況": "performance",
    "会計基準の選択に関する基本的な考え方": "accounting_policy",
    "1．経営成績等の概況": "performance",
    "（1）当期の経営成績の概況": "performance",
    "（2）当期の財政状態の概況": "financial_position",
    "（3）今後の見通し": "outlook",
    "2．会計基準の選択に関する基本的な考え方": "accounting_policy",
    
    # Financial statements
    "3．連結財務諸表及び主な注記": "financial_statements",
    "（1）連結財政状態計算書": "balance_sheet",
    "（2）連結損益計算書及び連結包括利益計算書": "income_statement",
    "（3）連結持分変動計算書": "equity_statement",
    "（4）連結キャッシュ・フロー計算書": "cash_flow",
    "（5）連結財務諸表に関する注記事項": "notes",
    
    # Business segments
    "①　国内損害保険事業": "segment_analysis",
    "②　海外保険事業": "segment_analysis",
    "③　国内生命保険事業": "segment_analysis",
    "④　介護事業": "segment_analysis",
    "1.報告セグメントの概要": "segment_analysis",
    "2.報告セグメントごとの収益、利益または損失その他の項目の金額の算定方法": "segment_analysis",
    "3.報告セグメントごとの収益、利益または損失その他の項目の金額に関する情報": "segment_analysis",
    "4.製品およびサービスごとの情報": "segment_analysis",
    "セグメント情報": "segment_analysis",
    
    # Per-share and shareholder information
    "１株当たり情報": "per_share_info",
    "1.基本的1株当たり当期利益": "per_share_info",
    "2.希薄化後 1株当たり当期利益": "per_share_info",
    "(重要な後発事象)": "capital_policy",  # Contains share buyback info
    
    # Geographical breakdown
    "(1) 収益": "geographical_analysis",
    "(2) 非流動資産": "geographical_analysis",
    
    # IFRS transition (very specific category)
    "IFRSへの移行に関する開示": "ifrs_transition",
    "1.遡及適用に対する免除規定": "ifrs_transition",
    "2.IFRS第1号の遡及適用に対する強制的な例外規定": "ifrs_transition",
    "3.日本基準からIFRSへの調整": "ifrs_transition",
    "(1) 企業結合": "ifrs_transition",
    "(3) 移行日前に認識した金融商品の指定": "ifrs_transition",
    "(4) 保険契約": "ifrs_transition",
    "(5) 借手のリース": "ifrs_transition",
    
    # Accounting policies and methods
    "(1) 「現金及び現金同等物」": "accounting_policy",
    "(6) 「金利収益」および「その他の投資損益」": "accounting_policy",
    "(8) 「その他の収益」および「その他の費用」": "accounting_policy",
    "(10) 連結の範囲": "accounting_policy",
    "(11) 報告期間の統一": "accounting_policy",
    "(13) 金融商品の分類および測定": "accounting_policy",
    "(15) 借手のリース": "accounting_policy",
    "(17) 保険契約および再保険契約": "accounting_policy",
    "(18) 確定給付制度に係る確定給付制度債務": "accounting_policy",
    "(19) 株式給付信託(BBT)": "accounting_policy",
    "(21) 繰延税金資産および繰延税金負債": "accounting_policy",
    
    # Risk management
    "(14) ヘッジ会計": "risk_management",
    
    # Corporate governance and compliance
    "継続企業の前提に関する注記": "going_concern",
    "追加情報": "regulatory_compliance",
    "添付資料の目次": "table_of_contents",
}


def classify_heading(raw: str) -> str:
    """Map Japanese heading text ⇒ canonical section_code."""
    if not raw:
        return "unknown"
    
    # Clean the text by removing extra spaces and full-width spaces
    key = re.sub(r"[\s　]+", "", raw.strip())
    
    # Try exact match first
    if key in HEADING_MAP:
        return HEADING_MAP[key]
    
    # Try partial matches for common patterns
    for heading_pattern, section_code in HEADING_MAP.items():
        if key in heading_pattern or heading_pattern in key:
            return section_code
    
    # Enhanced fallback classification using keyword patterns
    key_lower = key.lower()
    
    # IFRS and accounting standards  
    if any(keyword in key_lower for keyword in ['ifrs', '会計', '基準', '測定', '認識', '遡及', '移行', '調整', '企業結合', '連結', '現金及び現金同等物']):
        return "ifrs_transition"
    
    # Segment analysis
    if any(keyword in key_lower for keyword in ['セグメント', '事業', '報告', '収益', '地域']):
        return "segment_analysis"
    
    # Per-share information
    if any(keyword in key_lower for keyword in ['株', '1株', 'eps', '希薄化']):
        return "per_share_info"
    
    # Risk and hedging
    if any(keyword in key_lower for keyword in ['リスク', 'ヘッジ', 'デリバティブ']):
        return "risk_management"
    
    # Corporate actions and capital
    if any(keyword in key_lower for keyword in ['株式', '配当', '自己株式', '資本', '後発']):
        return "capital_policy"
    
    # Accounting policies
    if any(keyword in key_lower for keyword in ['金融商品', 'リース', '保険契約', '給付', '税金']):
        return "accounting_policy"
    
    # Default to general if no match found
    return "general"


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> Generator[str, None, None]:
    """Smart chunking that respects paragraph breaks and bullet points."""
    
    # First, try to split on natural boundaries for long text
    if len(text) > max_tokens * 4:  # ~2000+ chars, definitely needs smart splitting
        chunks = _smart_split_long_text(text, max_tokens)
        for chunk in chunks:
            yield chunk
        return
    
    # For shorter text, use simple word splitting with overlap
    words = text.split()
    if len(words) <= max_tokens:
        yield text
        return
        
    idx = 0
    while idx < len(words):
        yield " ".join(words[idx : idx + max_tokens])
        idx += max_tokens - overlap


def _smart_split_long_text(text: str, max_tokens: int = 400) -> List[str]:
    """Split long text on natural boundaries while respecting max token limits."""
    
    # Special handling for Japanese financial text with bullet points
    # Look for pattern: "主な項目は次のとおり" or "主な差異" followed by bullet points
    bullet_patterns = [
        r'主に次の差異があります。',
        r'主な項目は次のとおり',
        r'以下の.*について',
        r'次のとおり.*あります',
    ]
    
    # Try splitting on technical sections first
    for pattern in bullet_patterns:
        if re.search(pattern, text):
            # Split on bullet-like patterns
            parts = re.split(r'・(?=[^・])', text)  # Split on bullet points
            if len(parts) > 1:
                return _group_parts_into_chunks(parts, max_tokens)
    
    # Try splitting on paragraph breaks
    paragraphs = text.split('\n\n')
    if len(paragraphs) == 1:
        # No paragraph breaks, try splitting on periods + space for Japanese text
        # Look for sentence endings followed by space and capital/section markers
        paragraphs = re.split(r'(?<=。)\s+(?=[IFRS第|日本基準|当社グループ|.*について])', text)
        if len(paragraphs) == 1:
            # Try splitting on periods + space (more general)
            paragraphs = [s.strip() + '.' for s in text.split('. ') if s.strip()]
            if paragraphs[-1].endswith('..'):
                paragraphs[-1] = paragraphs[-1][:-1]  # Remove double period
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if adding this paragraph would exceed limit
        test_chunk = current_chunk + (" " if current_chunk else "") + para
        word_count = len(test_chunk.split())
        
        if word_count <= max_tokens:
            # Can add this paragraph
            current_chunk = test_chunk
        else:
            # Would exceed limit
            if current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Single paragraph is too long, need to force split it
                chunks.extend(_force_split_paragraph(para, max_tokens))
                current_chunk = ""
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _group_parts_into_chunks(parts: List[str], max_tokens: int = 400) -> List[str]:
    """Group text parts into chunks respecting token limits."""
    chunks = []
    current_chunk = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        test_chunk = current_chunk + (" " if current_chunk else "") + part
        word_count = len(test_chunk.split())
        
        if word_count <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                # Single part is too long, force split it
                chunks.extend(_force_split_paragraph(part, max_tokens))
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _force_split_paragraph(text: str, max_tokens: int = 400) -> List[str]:
    """Force split a single long paragraph that can't be split naturally."""
    
    # Try splitting on Japanese sentence patterns first
    sentences = []
    
    # Split on common Japanese sentence endings
    import re
    sentence_endings = r'[。！？]'
    parts = re.split(f'({sentence_endings})', text)
    
    current_sentence = ""
    for i, part in enumerate(parts):
        if re.match(sentence_endings, part):
            sentences.append(current_sentence + part)
            current_sentence = ""
        else:
            current_sentence += part
    
    if current_sentence:
        sentences.append(current_sentence)
    
    # Now group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        word_count = len(test_chunk.split())
        
        if word_count <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Even single sentence is too long, fall back to word splitting
                words = sentence.split()
                for i in range(0, len(words), max_tokens - 50):  # 50 word overlap
                    chunk_words = words[i:i + max_tokens]
                    chunks.append(" ".join(chunk_words))
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _enforce_size_limit(text: str, max_chars: int = 600) -> List[str]:
    """Force split text that exceeds size limit, trying to preserve meaning."""
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    
    # Try splitting on Japanese sentence endings first
    sentences = re.split(r'([。！？])', text)
    
    # Rebuild sentences
    full_sentences = []
    current = ""
    for i, part in enumerate(sentences):
        current += part
        if part in '。！？' or i == len(sentences) - 1:
            if current.strip():
                full_sentences.append(current.strip())
            current = ""
    
    # Group sentences into chunks
    current_chunk = ""
    for sentence in full_sentences:
        if not sentence:
            continue
            
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence too long, split by words
                words = sentence.split()
                words_per_chunk = int(max_chars / 5)  # Rough estimate: 5 chars per word
                for i in range(0, len(words), words_per_chunk):
                    word_chunk = " ".join(words[i:i + words_per_chunk])
                    chunks.append(word_chunk)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def categorize_xbrl_content(text: str, heading: str) -> str:
    """Categorize XBRL content type based on text patterns and heading"""
    text_clean = re.sub(r'\s+', ' ', text.lower())
    heading_clean = re.sub(r'\s+', ' ', heading.lower())
    
    # Use heading-based classification first (most reliable)
    
    # Forward-looking statements and forecasts
    if any(keyword in heading_clean for keyword in ['見通し', '予想', '予測', '見込み', '計画', '目標']) or \
       any(keyword in text_clean for keyword in ['次連結会計年度', '来期', '予想', '見込み', '予測']):
        return 'forecast'
    
    # Capital allocation and shareholder returns
    if any(keyword in heading_clean for keyword in ['配当', '自己株式', '株主還元', '資本政策', '後発事象']) or \
       any(keyword in text_clean for keyword in ['配当', '自己株式', '株主還元', '資本効率', '株式分割']):
        return 'capital_policy'
    
    # Risk management and hedging
    if any(keyword in heading_clean for keyword in ['リスク', 'ヘッジ', 'リスク管理']) or \
       any(keyword in text_clean for keyword in ['ヘッジ', 'リスク', 'デリバティブ', '金利リスク', '為替リスク']):
        return 'risk_management'
    
    # Accounting policies and standards
    if any(keyword in heading_clean for keyword in ['会計', '基準', 'ifrs', '日本基準', '測定', '認識']) or \
       any(keyword in text_clean for keyword in ['会計処理', '測定方法', 'ifrs', '会計基準', '償却', '減価償却']):
        return 'accounting_policy'
    
    # Business segment analysis
    if any(keyword in heading_clean for keyword in ['セグメント', '事業', '損害保険', '生命保険', '介護']) or \
       any(keyword in text_clean for keyword in ['セグメント', '事業部門', '報告セグメント', '保険事業']):
        return 'segment_analysis'
    
    # Per-share metrics and shareholder information
    if any(keyword in heading_clean for keyword in ['1株当たり', '株当たり', 'eps']) or \
       any(keyword in text_clean for keyword in ['1株当たり', '期中平均', '希薄化', '株式数']):
        return 'per_share_metrics'
    
    # Geographical/regional analysis
    if any(keyword in heading_clean for keyword in ['地域', '国内', '海外', '収益']) or \
       any(keyword in text_clean for keyword in ['日本', '海外', '国内', '所在地', '地域別']):
        return 'geographical_analysis'
    
    # Management performance commentary
    if any(keyword in heading_clean for keyword in ['経営成績', '業績', '財政状態']) or \
       any(keyword in text_clean for keyword in ['世界経済', '経営環境', '市場環境', '業績']):
        return 'management_discussion'
    
    # Financial position and cash flows
    if any(keyword in heading_clean for keyword in ['財政状態', 'キャッシュ', '資産', '負債']) or \
       any(keyword in text_clean for keyword in ['資産合計', '資本合計', 'キャッシュフロー', '現金']):
        return 'financial_position'
    
    # Financial performance metrics
    if any(keyword in text_clean for keyword in ['売上', '利益', '収益', '損失', '百万円', '億円']):
        return 'financial_metrics'
    
    # Regulatory and compliance
    if any(keyword in heading_clean for keyword in ['継続企業', '注記', '法人税', '税率']) or \
       any(keyword in text_clean for keyword in ['法律', '規制', '適時開示', '法人税法']):
        return 'regulatory_compliance'
    
    # Tables and structured data (catch remaining numeric content)
    if '|' in text_clean and any(pattern in text_clean for pattern in [r'\d+', '百万円', '千円']):
        return 'financial_table'
    
    # Corporate governance
    if any(keyword in text_clean for keyword in ['取締役', 'ガバナンス', '経営陣', '監査']):
        return 'corporate_governance'
    
    return 'general'

def extract_chunks(html: str, source_file: str, meta: Dict, disclosure_hash: str) -> List[XBRLChunk]:
    """Extract and chunk content from XBRL HTML"""
    soup = BeautifulSoup(html, "lxml")
    flatten_tables(soup)

    records = []
    chunk_index = 0
    
    # Extract content by headings
    for h in soup.select("h1, h2, h3, h4, h5, h6"):
        section = classify_heading(h.get_text())
        heading_text = clean_text(h.get_text(" ", strip=True))
        
        # Skip only if section is truly unknown and the content seems unimportant
        if section == "unknown" and len(heading_text.strip()) < 5:
            continue

        buffer = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                break
            txt = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip()
            if txt:
                buffer.append(txt)
        
        paragraph = clean_text("\n".join(buffer))
        
        if not paragraph or len(paragraph.strip()) < 50:
            continue
        
        # Categorize content
        content_type = categorize_xbrl_content(paragraph, heading_text)
        
        # Chunk the paragraph with size limit enforcement
        for _, chunk in enumerate(chunk_text(paragraph)):
            if len(chunk.strip()) < 30:
                continue
            
            # Enforce strict size limit: if chunk is >650 chars (~430 tokens), force split it
            if len(chunk) > 650:
                sub_chunks = _enforce_size_limit(chunk, 600)  # Target 600 chars max
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) < 30:
                        continue
                    token_count = count_tokens(sub_chunk)
                    is_numeric = is_numeric_content(sub_chunk)
                    vectorize_flag = should_vectorize_chunk(
                        content=sub_chunk,
                        content_type=content_type,
                        section_code=section,
                        char_length=len(sub_chunk),
                        tokens=token_count,
                        is_numeric=is_numeric
                    )
                    detected_language = detect_language(sub_chunk)
                    records.append(
                        XBRLChunk(
                            disclosure_id=meta.get('disclosure_id', 1),
                            chunk_index=chunk_index,
                            content=sub_chunk,
                            content_type=content_type,
                            section_code=section,
                            heading_text=heading_text,
                            char_length=len(sub_chunk),
                            tokens=token_count,
                            vectorize=vectorize_flag,
                            is_numeric=is_numeric,
                            disclosure_hash=disclosure_hash,
                            source_file=meta.get('xbrl_filename', source_file),
                            metadata={
                                'company_code': meta.get('company_code', '000000'),
                                'filing_date': str(meta.get('filing_date', dt.date.today())),
                                'period_end': str(meta.get('period_end', dt.date.today())),
                                'extraction_method': 'xbrl_qualitative',
                                'language': detected_language
                            }
                        )
                    )
                    chunk_index += 1
            else:
                token_count = count_tokens(chunk)
                is_numeric = is_numeric_content(chunk)
                vectorize_flag = should_vectorize_chunk(
                    content=chunk,
                    content_type=content_type,
                    section_code=section,
                    char_length=len(chunk),
                    tokens=token_count,
                    is_numeric=is_numeric
                )
                detected_language = detect_language(chunk)
                records.append(
                XBRLChunk(
                    disclosure_id=meta.get('disclosure_id', 1),
                    chunk_index=chunk_index,
                    content=chunk,
                    content_type=content_type,
                    section_code=section,
                    heading_text=heading_text,
                    char_length=len(chunk),
                    tokens=token_count,
                    vectorize=vectorize_flag,
                    is_numeric=is_numeric,
                    disclosure_hash=disclosure_hash,
                    source_file=meta.get('xbrl_filename', source_file),
                    metadata={
                        'company_code': meta.get('company_code', '000000'),
                        'filing_date': str(meta.get('filing_date', dt.date.today())),
                        'period_end': str(meta.get('period_end', dt.date.today())),
                        'extraction_method': 'xbrl_qualitative',
                        'language': detected_language
                    }
                )
            )
            chunk_index += 1
    
    return records


class XBRLProcessor:
    """Process XBRL files for qualitative data extraction"""
    
    def __init__(self):
        logger.info("Initialized XBRL processor")
    
    def process_xbrl_file(self, xbrl_path: str, disclosure_id: int = 1) -> List[XBRLChunk]:
        """Process single XBRL file and extract chunks"""
        if not os.path.exists(xbrl_path):
            logger.error(f"XBRL file not found: {xbrl_path}")
            return []
        
        logger.info(f"Processing XBRL: {xbrl_path}")
        
        try:
            meta = parse_zip_meta(os.path.basename(xbrl_path))
            meta['disclosure_id'] = disclosure_id
            
            # Calculate hash of XBRL file for idempotency
            disclosure_hash = sha256_file(Path(xbrl_path))
            
            with zipfile.ZipFile(xbrl_path) as z:
                # Find qualitative.htm files
                q_files = [n for n in z.namelist() if n.lower().endswith("qualitative.htm")]
                
                if not q_files:
                    logger.warning(f"No qualitative.htm found in {xbrl_path}")
                    return []
                
                all_chunks = []
                for qf in q_files:
                    try:
                        html = z.read(qf).decode("utf-8", "ignore")
                        chunks = extract_chunks(html, qf, meta, disclosure_hash)
                        all_chunks.extend(chunks)
                        logger.info(f"Extracted {len(chunks)} chunks from {qf}")
                    except Exception as e:
                        logger.error(f"Error processing {qf}: {e}")
                        continue
                
                # Apply deduplication
                all_chunks = self._deduplicate_chunks(all_chunks)
                logger.info(f"Total chunks after deduplication: {len(all_chunks)}")
                
                # Apply intelligent filtering for embedding quality
                all_chunks = self._filter_chunks_for_quality(all_chunks)
                logger.info(f"Total chunks after quality filtering: {len(all_chunks)}")
                
                # Remove duplicate heading blocks (fragments from split tables)
                all_chunks = self._remove_duplicate_heading_blocks(all_chunks)
                logger.info(f"Total chunks after duplicate heading removal: {len(all_chunks)}")
                
                return all_chunks
                
        except Exception as e:
            logger.error(f"Error processing XBRL {xbrl_path}: {e}")
            return []
    
    def _deduplicate_chunks(self, chunks: List[XBRLChunk]) -> List[XBRLChunk]:
        """Remove duplicate chunks based on content"""
        if not chunks:
            return chunks
        
        seen_content = set()
        deduplicated = []
        
        for chunk in chunks:
            content_hash = hash(chunk.content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(chunk)
            else:
                logger.debug(f"Removed duplicate chunk: {chunk.content[:50]}...")
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(deduplicated)} chunks")
        return deduplicated
    
    def _filter_chunks_for_quality(self, chunks: List[XBRLChunk]) -> List[XBRLChunk]:
        """Filter chunks to improve embedding quality by removing noise"""
        if not chunks:
            return chunks
        
        logger.info(f"Starting quality filtering of {len(chunks)} chunks")
        filtered_chunks = []
        removed_reasons = {
            'table_of_contents': 0,
            'mega_tables': 0,
            'very_short': 0,
            'mostly_numeric': 0,
            'continuation_fragments': 0,
            'low_information': 0
        }
        
        # Note: Could extend to detect equity statement fragmentation patterns if needed
        # equity_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if chunk.section_code == 'equity_statement']
        
        for i, chunk in enumerate(chunks):
            content = chunk.content
            section = chunk.section_code
            content_length = len(content)
            should_keep = True
            removal_reason = None
            
            # 1. Remove table of contents - just navigation, no business content
            if section == 'table_of_contents':
                should_keep = False
                removal_reason = 'table_of_contents'
            
            # 2. Remove very short chunks (likely fragments or incomplete thoughts)
            elif content_length < 100:
                should_keep = False
                removal_reason = 'very_short'
            
            # 3. Remove mega tables (balance sheet, cash flow, detailed financials)
            # These are too structured/numeric for good embeddings
            elif section in ['balance_sheet', 'cash_flow'] and content_length > 800:
                should_keep = False
                removal_reason = 'mega_tables'
                
            # 4. Remove equity statement fragments (usually split tables)
            elif section == 'equity_statement' and content_length > 1000:
                should_keep = False
                removal_reason = 'mega_tables'
            
            # 4b. Remove IFRS transition tables and reconciliation tables
            elif (any(pattern in content for pattern in [
                '日本基準', 'IFRS', '表示組替', '認識および測定の差異', '調整額',
                '移行日', '前連結会計年度', '当連結会計年度',
                '資本に対する調整', '包括利益に対する調整'
            ]) and content_length > 800 and content.count('|') > 15):
                should_keep = False
                removal_reason = 'mega_tables'
            
            # 5. Remove mostly numeric content (poor for semantic embeddings)
            elif content_length > 200:
                pipe_count = content.count('|')
                digit_count = sum(1 for c in content if c.isdigit())
                numeric_ratio = (pipe_count + digit_count) / content_length
                
                # More aggressive filtering: 
                # - Large chunks (>1000 chars) with >20% numeric are likely mega-tables
                # - Medium chunks (>500 chars) with >25% numeric are structured tables
                # - Smaller chunks with >35% numeric are table fragments
                if ((content_length > 1000 and numeric_ratio > 0.20) or 
                    (content_length > 500 and numeric_ratio > 0.25) or 
                    (numeric_ratio > 0.35)):
                    should_keep = False
                    removal_reason = 'mostly_numeric'
            
            # 6. Remove continuation fragments (chunks that start with pipes or seem incomplete)
            elif (content.strip().startswith('|') or 
                  content.strip().startswith(('2:', '3:', '4:', '5:', '6:', '7:')) or
                  content.count('|') > 20 and content_length < 500):
                should_keep = False
                removal_reason = 'continuation_fragments'
            
            # 7. Remove low-information content (mostly formatting, units, etc.)
            elif self._is_low_information_content(content):
                should_keep = False
                removal_reason = 'low_information'
            
            # Keep the chunk if it passed all filters
            if should_keep:
                filtered_chunks.append(chunk)
            else:
                removed_reasons[removal_reason] += 1
                logger.debug(f"Removed chunk {i} ({removal_reason}): {content[:100]}...")
        
        # Re-index remaining chunks
        for idx, chunk in enumerate(filtered_chunks):
            chunk.chunk_index = idx
        
        # Log filtering results
        total_removed = len(chunks) - len(filtered_chunks)
        logger.info(f"Quality filtering removed {total_removed} chunks:")
        for reason, count in removed_reasons.items():
            if count > 0:
                logger.info(f"  - {reason}: {count}")
        
        return filtered_chunks
    
    def _is_low_information_content(self, content: str) -> bool:
        """Detect content with low semantic information value"""
        content_lower = content.lower().strip()
        
        # Very repetitive content (same phrase repeated)
        words = content_lower.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return True
        
        # Mostly units and formatting
        formatting_indicators = ['(単位:', '百万円)', '自 2023年', '至 2024年', '至 2025年']
        if any(indicator in content for indicator in formatting_indicators) and len(content) < 200:
            return True
        
        # Just period/date ranges
        if re.match(r'^[（\(]?自?\s*\d{4}年\d{1,2}月\d{1,2}日.*至?\s*\d{4}年\d{1,2}月\d{1,2}日[）\)]?$', content_lower.strip()):
            return True
        
        return False
    
    def _remove_duplicate_heading_blocks(self, chunks: List[XBRLChunk]) -> List[XBRLChunk]:
        """Remove chunks that share the same heading (likely table fragments)"""
        if not chunks:
            return chunks
        
        logger.info(f"Checking for duplicate heading blocks in {len(chunks)} chunks")
        
        # Group chunks by heading
        from collections import defaultdict
        heading_groups = defaultdict(list)
        for i, chunk in enumerate(chunks):
            heading = chunk.heading_text.strip()
            heading_groups[heading].append((i, chunk))
        
        # Identify problematic duplicate headings
        filtered_chunks = []
        removed_count = 0
        
        for heading, chunk_list in heading_groups.items():
            if len(chunk_list) == 1:
                # Single chunk with this heading - keep it
                filtered_chunks.append(chunk_list[0][1])
            else:
                # Multiple chunks with same heading - analyze them
                total_chars = sum(len(chunk.content) for _, chunk in chunk_list)
                _ = total_chars / len(chunk_list)  # avg_chars for potential future use
                
                # Check if this looks like a split table
                is_table_split = False
                
                # Heuristics for table splits:
                # 1. Multiple large chunks (>500 chars) with same heading
                # 2. High numeric content in the chunks
                # 3. Contains table-like patterns
                large_chunks = [(i, chunk) for i, chunk in chunk_list if len(chunk.content) > 500]
                
                if len(large_chunks) > 1:
                    # Check numeric content ratio
                    numeric_ratios = []
                    for _, chunk in large_chunks:
                        content = chunk.content
                        digit_count = sum(1 for c in content if c.isdigit())
                        pipe_count = content.count('|')
                        numeric_ratio = (digit_count + pipe_count) / len(content)
                        numeric_ratios.append(numeric_ratio)
                    
                    avg_numeric_ratio = sum(numeric_ratios) / len(numeric_ratios)
                    
                    # If multiple large chunks with same heading and high numeric content,
                    # likely a split table - remove all
                    if avg_numeric_ratio > 0.15:  # 15% numeric suggests structured data
                        is_table_split = True
                        logger.debug(f"Removing {len(chunk_list)} chunks with duplicate heading '{heading}' (avg {avg_numeric_ratio:.1%} numeric)")
                
                if is_table_split:
                    # Remove all chunks with this heading
                    removed_count += len(chunk_list)
                else:
                    # Keep the longest chunk (most complete)
                    best_chunk = max(chunk_list, key=lambda x: len(x[1].content))[1]
                    filtered_chunks.append(best_chunk)
                    removed_count += len(chunk_list) - 1
                    logger.debug(f"Keeping longest chunk for heading '{heading}', removing {len(chunk_list)-1} duplicates")
        
        # Re-index remaining chunks
        for idx, chunk in enumerate(filtered_chunks):
            chunk.chunk_index = idx
        
        logger.info(f"Removed {removed_count} duplicate heading chunks")
        return filtered_chunks


# ---------------------------------------------------------------------------
# Zip ingestion
# ---------------------------------------------------------------------------


def parse_zip_meta(name: str) -> Dict:
    """Extract company code & filing date from JPX zip filename."""
    # Try pattern: prefix_companycode_date_suffix.zip
    # Example: 15-30_86300_2025年3月期決算短信〔ＩＦＲＳ〕(連結).zip
    m = re.search(r"_(\d{4,6})_", name)
    if m:
        company = m.group(1)
    else:
        # Fallback: try other patterns or extract from filename
        digits = re.findall(r'\d{5,6}', name)
        company = digits[0] if digits else "000000"
    
    # Try to extract date from filename (Japanese format)
    date_match = re.search(r'(\d{4})年(\d{1,2})月', name)
    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
        # Use end of fiscal year
        date = dt.date(year, month, 31) if month == 3 else dt.date(year, month, 30)
    else:
        date = dt.date.today()
    
    return {
        "company_code": company, 
        "filing_date": date, 
        "period_end": date, 
        "disclosure_id": 1,
        "xbrl_filename": name
    }


class XBRLExtractionTester:
    """Test harness for XBRL extraction pipeline"""
    
    def __init__(self, output_dir: str = "test_output_xbrl"):
        self.processor = XBRLProcessor()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def test_single_xbrl(self, xbrl_path: str) -> Dict[str, Any]:
        """Test processing of a single XBRL file"""
        if not os.path.exists(xbrl_path):
            logger.error(f"XBRL file not found: {xbrl_path}")
            return {}
        
        logger.info(f"Testing XBRL: {xbrl_path}")
        
        chunks = self.processor.process_xbrl_file(xbrl_path)
        
        # Generate statistics
        stats = self._generate_stats(chunks, xbrl_path)
        
        # Save results
        self._save_results(chunks, stats, xbrl_path)
        
        return stats
    
    def test_multiple_xbrls(self, xbrl_dir: str, max_files: int = 5) -> Dict[str, Any]:
        """Test processing of multiple XBRL files"""
        if not os.path.exists(xbrl_dir):
            logger.error(f"XBRL directory not found: {xbrl_dir}")
            return {}
        
        # Find XBRL zip files
        xbrl_files = []
        for file in os.listdir(xbrl_dir):
            if file.lower().endswith('.zip'):
                xbrl_files.append(os.path.join(xbrl_dir, file))
        
        if not xbrl_files:
            logger.error(f"No XBRL zip files found in: {xbrl_dir}")
            return {}
        
        # Limit to max_files
        xbrl_files = xbrl_files[:max_files]
        logger.info(f"Testing {len(xbrl_files)} XBRL files")
        
        all_results = {}
        total_chunks = 0
        failed_files = []
        
        for i, xbrl_path in enumerate(xbrl_files, 1):
            filename = os.path.basename(xbrl_path)
            logger.info(f"Processing file {i}/{len(xbrl_files)}: {filename}")
            
            try:
                stats = self.test_single_xbrl(xbrl_path)
                all_results[filename] = stats
                
                if 'error' in stats:
                    failed_files.append({
                        'filename': filename,
                        'error': stats['error']
                    })
                else:
                    total_chunks += stats.get('total_chunks', 0)
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing {xbrl_path}: {error_msg}")
                all_results[filename] = {'error': error_msg}
                failed_files.append({
                    'filename': filename,
                    'error': error_msg
                })
        
        # Generate summary
        summary = {
            'total_files_processed': len([r for r in all_results.values() if 'error' not in r]),
            'total_files_failed': len([r for r in all_results.values() if 'error' in r]),
            'total_chunks_extracted': total_chunks,
            'failed_files': failed_files,
            'results': all_results
        }
        
        # Save summary
        self._save_summary(summary)
        
        return summary
    
    def _generate_stats(self, chunks: List[XBRLChunk], xbrl_path: str) -> Dict[str, Any]:
        """Generate statistics for processed chunks"""
        if not chunks:
            return {
                'xbrl_path': os.path.basename(xbrl_path),
                'total_chunks': 0,
                'error': 'No chunks extracted'
            }
        
        # Content type distribution
        content_types = {}
        for chunk in chunks:
            content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
        
        # Section distribution
        sections = {}
        for chunk in chunks:
            sections[chunk.section_code] = sections.get(chunk.section_code, 0) + 1
        
        # Language distribution
        languages = {}
        for chunk in chunks:
            lang = chunk.metadata.get('language', 'ja')
            languages[lang] = languages.get(lang, 0) + 1
        
        # Chunk size statistics
        chunk_sizes = [chunk.char_length for chunk in chunks]
        token_counts = [chunk.tokens for chunk in chunks]
        
        # Vectorization statistics
        vectorizable_chunks = [chunk for chunk in chunks if chunk.vectorize]
        non_vectorizable_chunks = [chunk for chunk in chunks if not chunk.vectorize]
        
        vectorization_stats = {
            'total_chunks': len(chunks),
            'vectorizable': len(vectorizable_chunks),
            'non_vectorizable': len(non_vectorizable_chunks),
            'vectorizable_percentage': (len(vectorizable_chunks) / len(chunks) * 100) if chunks else 0
        }
        
        # Token stats for vectorizable chunks only
        vectorizable_tokens = [chunk.tokens for chunk in vectorizable_chunks] if vectorizable_chunks else [0]
        
        stats = {
            'xbrl_path': os.path.basename(xbrl_path),
            'total_chunks': len(chunks),
            'content_type_distribution': content_types,
            'section_distribution': sections,
            'language_distribution': languages,
            'chunk_size_stats': {
                'min': min(chunk_sizes) if chunk_sizes else 0,
                'max': max(chunk_sizes) if chunk_sizes else 0,
                'avg': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            },
            'token_stats': {
                'min': min(token_counts) if token_counts else 0,
                'max': max(token_counts) if token_counts else 0,
                'avg': sum(token_counts) / len(token_counts) if token_counts else 0,
                'total': sum(token_counts)
            },
            'vectorization_stats': vectorization_stats,
            'vectorizable_token_stats': {
                'min': min(vectorizable_tokens),
                'max': max(vectorizable_tokens),
                'avg': sum(vectorizable_tokens) / len(vectorizable_tokens),
                'total': sum(vectorizable_tokens)
            } if vectorizable_chunks else {'min': 0, 'max': 0, 'avg': 0, 'total': 0},
            'sample_chunks': [
                {
                    'chunk_index': chunk.chunk_index,
                    'content_type': chunk.content_type,
                    'section_code': chunk.section_code,
                    'heading_text': chunk.heading_text,
                    'char_length': chunk.char_length,
                    'tokens': chunk.tokens,
                    'vectorize': chunk.vectorize,
                    'language': chunk.metadata.get('language', 'ja'),
                    'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                }
                for chunk in chunks[:3]
            ]
        }
        
        return stats
    
    def _save_results(self, chunks: List[XBRLChunk], stats: Dict[str, Any], xbrl_path: str):
        """Save processing results to files"""
        base_name = os.path.splitext(os.path.basename(xbrl_path))[0]
        
        # Save chunks as JSON
        chunks_file = os.path.join(self.output_dir, f"{base_name}_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2, default=str)
        
        # Save stats
        stats_file = os.path.join(self.output_dir, f"{base_name}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Results saved: {chunks_file}, {stats_file}")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save overall summary"""
        summary_file = os.path.join(self.output_dir, f"xbrl_extraction_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Summary saved: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("XBRL EXTRACTION SUMMARY")
        print("="*50)
        print(f"Files processed successfully: {summary['total_files_processed']}")
        print(f"Files failed: {summary['total_files_failed']}")
        print(f"Total chunks extracted: {summary['total_chunks_extracted']}")
        print(f"Results saved to: {self.output_dir}")
        
        if summary['total_files_failed'] > 0:
            print("\n" + "="*50)
            print("FAILED FILES SUMMARY")
            print("="*50)
            for i, failed_file in enumerate(summary.get('failed_files', []), 1):
                print(f"{i}. {failed_file['filename']}")
                print(f"   Error: {failed_file['error']}")
                print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='XBRL Qualitative Data Extraction Pipeline Tester')
    parser.add_argument('--xbrl-file', type=str,
                       help='Path to a single XBRL zip file to process')
    parser.add_argument('--xbrl-dir', type=str,
                       help='Directory containing XBRL zip files to process')
    parser.add_argument('--max-files', type=int, default=5,
                       help='Maximum number of XBRL files to process (default: 5)')
    parser.add_argument('--output-dir', type=str, default='test_output_xbrl',
                       help='Directory to save output files (default: test_output_xbrl)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging for detailed output')
    
    args = parser.parse_args()
    
    # Adjust logging level if debug is requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    if not args.xbrl_file and not args.xbrl_dir:
        print("Error: Please specify either --xbrl-file or --xbrl-dir")
        parser.print_help()
        sys.exit(1)
    
    tester = XBRLExtractionTester(args.output_dir)
    
    try:
        if args.xbrl_file:
            logger.info("Testing single XBRL file...")
            stats = tester.test_single_xbrl(args.xbrl_file)
            print("\nProcessing completed!")
            print(f"Extracted {stats.get('total_chunks', 0)} chunks")
            
        elif args.xbrl_dir:
            logger.info("Testing multiple XBRL files...")
            _ = tester.test_multiple_xbrls(args.xbrl_dir, args.max_files)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
