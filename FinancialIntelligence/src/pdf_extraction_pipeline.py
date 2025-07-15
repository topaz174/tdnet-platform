#!/usr/bin/env python3
"""
Standalone PDF Document Extraction Pipeline for Testing

This script processes PDF files to extract structured text content
with enhanced chunking and content categorization for testing purposes.

Usage:
    python pdf_extraction_pipeline.py --pdf-dir /path/to/pdfs --max-files 5
    python pdf_extraction_pipeline.py --pdf-file /path/to/single.pdf
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import unicodedata

import warnings, logging
# Suppress pdfminer logging warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

# PDF processing
try:
    import pdfplumber
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")
    PDF_PROCESSING_AVAILABLE = False
    sys.exit(1)

# Text processing
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not installed. Japanese text processing will be limited.")
    MECAB_AVAILABLE = False

# Japanese character normalization
try:
    import jaconv
    JACONV_AVAILABLE = True
except ImportError:
    print("Warning: jaconv not installed. Install with: pip install jaconv")
    JACONV_AVAILABLE = False

# OCR processing for image-based PDFs
# Check for Tesseract
try:
    import pytesseract
    from PIL import Image
    import io
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: Tesseract OCR not available. Install with: pip install pytesseract pillow")
    print("Note: You may also need to install tesseract-ocr system package and configure pytesseract.pytesseract.tesseract_cmd")
    TESSERACT_AVAILABLE = False

# Check for Dolphin OCR
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    DOLPHIN_AVAILABLE = True
except ImportError:
    print("Warning: Dolphin OCR not available. Install with: pip install torch transformers")
    DOLPHIN_AVAILABLE = False

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Initialize the tokenizer for o3 model for accurate token counting
    _tokenizer = tiktoken.encoding_for_model("o3")
except ImportError:
    print("Warning: tiktoken not installed. Token counts will be estimated. Install with: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False
    _tokenizer = None

# Set OCR availability
OCR_AVAILABLE = TESSERACT_AVAILABLE or DOLPHIN_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # INFO level to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce verbosity of pdfminer and other libraries
logging.getLogger('pdfminer').setLevel(logging.WARNING)
logging.getLogger('pdfplumber').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Enhanced character normalization mapping - unified with XBRL pipeline
FULL_WIDTH_MAP = str.maketrans({
    # Financial symbols - comprehensive normalization for Japanese financial documents
    "△": "-",    # triangle minus (negative values) - CRITICAL for financial data
    "▲": "-",    # black triangle (negative values)
    "％": "%",    # full-width percent
    "－": "-",    # full-width hyphen-minus (U+FF0D)
    "−": "-",    # minus sign (U+2212)
    "–": "-",    # en dash
    "—": "-",    # em dash
    "～": "~",    # full-width tilde 
    "〜": "~",    # wave dash
    "∼": "~",    # tilde operator
    
    # Full-width digits (commonly seen in Japanese financial docs)
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    
    # Full-width punctuation
    "（": "(",    # full-width parentheses
    "）": ")",
    "［": "[",    # full-width brackets
    "］": "]",
    "｛": "{",    # full-width braces
    "｝": "}",
    "「": '"',    # Japanese corner brackets → quotes (XBRL style)
    "」": '"',    # Japanese corner brackets → quotes (XBRL style)
    "『": "'",    # Japanese white corner brackets → single quotes (XBRL style)
    "』": "'",    # Japanese white corner brackets → single quotes (XBRL style)
    "：": ":",    # full-width colon
    "；": ";",    # full-width semicolon
    "，": ",",    # full-width comma
    "．": ".",    # full-width period
    "？": "?",    # full-width question mark
    "！": "!",    # full-width exclamation mark
    "／": "/",    # full-width slash (ADDED from XBRL)
    
    # Full-width mathematical and financial symbols
    "＋": "+",    # full-width plus
    "＝": "=",    # full-width equals
    "×": "×",    # multiplication sign (keep as-is for Japanese)
    "÷": "÷",    # division sign (keep as-is for Japanese)
    "＜": "<",    # full-width less than
    "＞": ">",    # full-width greater than
    "＊": "*",    # full-width asterisk (ADDED from XBRL)
    "＃": "#",    # full-width hash (ADDED from XBRL)
    "＠": "@",    # full-width at sign (ADDED from XBRL)
    "＆": "&",    # full-width ampersand (ADDED from XBRL)
    
    # Full-width enumeration characters (MAJOR ADDITION from XBRL)
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
    
    # Full-width spaces and other whitespace
    "\u3000": " ", # ideographic space
    "\u00A0": " ", # non-breaking space
    "\u2009": " ", # thin space
    "\u200A": " ", # hair space
    "\u2002": " ", # en space
    "\u2003": " ", # em space
})

def normalize_text(txt: str) -> str:
    """
    Enhanced Japanese financial text normalization unified with XBRL pipeline.
    
    Performs comprehensive character normalization including:
    - Full-width to half-width conversion via jaconv
    - Financial symbol standardization (△ → -)
    - Enumeration character conversion (①②③ → 1.2.3.)
    - Punctuation and space normalization
    """
    if not txt:
        return txt
    
    # 1) Replace full-width spaces first (before jaconv processing)
    txt = txt.replace("\u3000", " ")
    
    # 2) Use jaconv for comprehensive Japanese character normalization (XBRL enhancement)
    if JACONV_AVAILABLE:
        # Convert full-width ASCII and digits to half-width, keep kana as-is
        txt = jaconv.z2h(txt, kana=False, digit=True, ascii=True)
    else:
        # Fallback: basic Unicode normalization
        txt = unicodedata.normalize("NFKC", txt)

    # 3) Apply enhanced full-width character mappings (includes XBRL improvements)
    txt = txt.translate(FULL_WIDTH_MAP)
    
    # 4) Financial data specific cleaning (from XBRL pipeline)
    # Remove digit group-separators (,) in financial numbers
    # Only remove commas between digits (thousand separators)
    txt = re.sub(r"(?<=\d),(?=\d{3})", "", txt)

    # 5) Collapse multiple spaces left by replacements
    txt = re.sub(r"\s+", " ", txt).strip()
    
    return txt

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

def is_numeric_content(content: str) -> bool:
    """
    Determine if content is primarily numeric/tabular data.
    
    Returns True for content that is predominantly numbers, financial data,
    and table structures rather than narrative text.
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
    
    Returns False for chunks that would be flagged under quality issues.
    """
    # Rule 1: Skip numeric/tabular content (megatable slabs)
    if is_numeric is not None and is_numeric:
        return False
    
    # Fallback to legacy numeric detection for compatibility
    if is_numeric is None and content:
        # Count numeric characters (digits, decimals, separators)
        numeric_chars = sum(1 for c in content if c.isdigit() or c in '.,|-△▲')
        numeric_ratio = numeric_chars / len(content)
        
        # More aggressive filtering for large chunks with high numeric content
        if ((char_length > 400 and numeric_ratio > 0.25) or  # Large chunks: >25% numeric
            (char_length > 200 and numeric_ratio > 0.35) or  # Medium chunks: >35% numeric  
            (char_length > 100 and numeric_ratio > 0.50) or  # Small chunks: >50% numeric
            (numeric_ratio > 0.60)):                         # Any chunk: >60% numeric
            return False
    
    # Rule 2: Skip very short fragments (likely continuation fragments)
    if char_length < 50 or tokens < 15:
        return False
        
    # Rule 3: Skip table-heavy sections that are primarily numeric
    numeric_sections = ['table', 'financial_summary']
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
    
    # Default: vectorize this chunk
    return True

def classify_pdf_section(content: str, page_num: int, content_type: str) -> str:
    """Classify PDF content into section codes similar to XBRL headings."""
    content_lower = content.lower()
    
    # Map content types to section codes
    content_type_mapping = {
        'financial_summary': 'performance',
        'management_discussion': 'performance', 
        'table': 'financial_statements',
        'organizational': 'notes',
        'regulatory': 'regulatory_compliance',
        'risk_factors': 'risk_management',
        'business_overview': 'segment_analysis',
        'equity_matters': 'capital_policy',
        'notes': 'notes',
        'header_footer': 'general'
    }
    
    # Enhanced classification using keyword patterns
    if any(keyword in content_lower for keyword in ['見通し', '予想', '予測', '見込み', '計画', '目標']):
        return "outlook"
    
    if any(keyword in content_lower for keyword in ['配当', '自己株式', '株主還元', '資本政策', '後発事象']):
        return "capital_policy"
    
    if any(keyword in content_lower for keyword in ['セグメント', '事業', '報告', '収益', '地域']):
        return "segment_analysis"
    
    if any(keyword in content_lower for keyword in ['株', '1株', 'eps', '希薄化']):
        return "per_share_info"
    
    if any(keyword in content_lower for keyword in ['会計', '基準', 'ifrs', '日本基準', '測定', '認識']):
        return "accounting_policy"
    
    if any(keyword in content_lower for keyword in ['財政状態', 'キャッシュ', '資産', '負債']):
        return "financial_position"
    
    if any(keyword in content_lower for keyword in ['経営成績', '業績', '損益', '売上', '利益']):
        return "performance"
    
    # Use content type mapping as fallback
    return content_type_mapping.get(content_type, 'general')

def extract_pdf_heading(content: str, page_num: int) -> str:
    """Extract or generate heading text for PDF content."""
    lines = content.split('\n')
    
    # Look for heading patterns in first few lines
    for i, line in enumerate(lines[:3]):
        line = line.strip()
        if not line:
            continue
            
        # Japanese heading patterns
        if re.match(r'^[0-9①②③④⑤\(\（].*(について|概況|状況|方針|事項)', line):
            return line
        
        # Title case or formal headings
        if len(line) < 100 and any(char in line for char in '()（）[]【】'):
            return line
        
        # First substantial line as fallback
        if len(line) > 10 and i == 0:
            return line[:50] + '...' if len(line) > 50 else line
    
    # Generate page-based heading as last resort
    return f"Page {page_num} Content"

@dataclass
class DocumentChunk:
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
    page_number: Optional[int]
    metadata: Dict[str, Any]

class PDFProcessor:
    """Enhanced PDF processing for better text chunking and content categorization"""
    
    def __init__(self, ocr_backend: str = "auto", disable_ocr: bool = False, smart_ocr: bool = True):
        """
        Initialize PDF processor with OCR backend selection.
        
        Args:
            ocr_backend: OCR backend to use ('auto', 'tesseract', 'dolphin', 'none')
                        'auto' will prefer dolphin if available, fall back to tesseract
            disable_ocr: If True, completely disable OCR even when text extraction fails
            smart_ocr: If True, automatically detect when OCR is needed vs text-based PDFs
        """
        self.mecab = None
        self.disable_ocr = disable_ocr
        self.smart_ocr = smart_ocr
        self.ocr_backend = self._select_ocr_backend(ocr_backend) if not disable_ocr else "none"
        self._dolphin_model = None
        self._dolphin_tokenizer = None
        
        if MECAB_AVAILABLE:
            # Try different MeCab initialization approaches for Ubuntu/Debian
            mecab_configs = [
                "-r/etc/mecabrc -d/var/lib/mecab/dic/ipadic-utf8",  # Explicit Ubuntu/Debian paths
                "-r/etc/mecabrc",  # Standard Ubuntu/Debian location
                "-r/dev/null -d/var/lib/mecab/dic/ipadic-utf8",  # Empty mecabrc with explicit dictionary
                "-r/dev/null",  # Empty mecabrc (recommended for Ubuntu/Debian)
                "",  # Default configuration
            ]
            
            for config in mecab_configs:
                try:
                    if config:
                        self.mecab = MeCab.Tagger(config)
                        logger.info(f"MeCab initialized successfully with config: '{config}'")
                    else:
                        self.mecab = MeCab.Tagger()
                        logger.info("MeCab initialized successfully with default config")
                    break
                except Exception as e:
                    logger.debug(f"MeCab initialization failed with config '{config}': {e}")
                    continue
            
            if self.mecab is None:
                logger.warning("MeCab initialization failed with all attempted configurations")
                logger.warning("Continuing without MeCab - Japanese text processing will be limited")
            else:
                logger.info("MeCab is ready for Japanese text processing")
    
    def _detect_pdf_type(self, pdf_path: str) -> str:
        """
        Detect if PDF is text-based or image-based by analyzing text extraction success.
        
        Returns:
            'text': PDF has extractable text (OCR not needed)
            'image': PDF is primarily images (needs OCR)
            'mixed': PDF has both text and images (OCR as fallback)
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                sample_pages = min(3, total_pages)  # Check first 3 pages for speed
                
                total_text_length = 0
                successful_extractions = 0
                
                for page_num in range(sample_pages):
                    page = pdf.pages[page_num]
                    
                    # Try to extract text using multiple methods
                    text_methods = []
                    
                    # Method 1: Standard extraction
                    try:
                        text1 = page.extract_text()
                        if text1:
                            text_methods.append(len(text1.strip()))
                    except:
                        pass
                    
                    # Method 2: Layout extraction
                    try:
                        text2 = page.extract_text(layout=True)
                        if text2:
                            text_methods.append(len(text2.strip()))
                    except:
                        pass
                    
                    # Use the best extraction result for this page
                    page_text_length = max(text_methods) if text_methods else 0
                    total_text_length += page_text_length
                    
                    # Consider extraction successful if we got substantial text
                    if page_text_length > 50:
                        successful_extractions += 1
                
                # Calculate metrics
                avg_text_per_page = total_text_length / sample_pages if sample_pages > 0 else 0
                success_rate = successful_extractions / sample_pages if sample_pages > 0 else 0
                
                # Classification based on text extraction success
                if avg_text_per_page > 200 and success_rate > 0.5:
                    return 'text'  # Good text extraction - no OCR needed
                elif avg_text_per_page < 50 and success_rate < 0.3:
                    return 'image'  # Poor text extraction - likely image-based PDF
                else:
                    return 'mixed'  # Mixed or uncertain - use OCR as fallback
                    
        except Exception as e:
            logger.warning(f"PDF type detection failed for {pdf_path}: {e}")
            return 'mixed'  # Default to mixed when uncertain
    
    def _should_use_ocr_for_pdf(self, pdf_path: str) -> bool:
        """
        Determine if this PDF likely needs OCR based on content analysis.
        
        Returns:
            True: PDF appears to be image-based, OCR recommended
            False: PDF appears to be text-based, OCR not needed
        """
        if self.disable_ocr:
            return False
            
        if not self.smart_ocr:
            return True  # Use OCR as fallback if smart detection disabled
            
        pdf_type = self._detect_pdf_type(pdf_path)
        logger.info(f"PDF type detected: {pdf_type}")
        
        # Conservative approach: only use OCR for clearly image-based PDFs
        # For mixed PDFs, still use OCR as fallback but with higher thresholds
        if pdf_type == 'image':
            return True
        elif pdf_type == 'text':
            return False
        else:  # mixed
            return True  # Use OCR as fallback for mixed content
    
    def _select_ocr_backend(self, preferred: str) -> str:
        """Select the best available OCR backend"""
        if preferred == "dolphin" and DOLPHIN_AVAILABLE:
            logger.info("Using Dolphin OCR backend")
            return "dolphin"
        elif preferred == "tesseract" and TESSERACT_AVAILABLE:
            logger.info("Using Tesseract OCR backend")
            return "tesseract"
        elif preferred == "auto":
            if DOLPHIN_AVAILABLE:
                logger.info("Auto-selected Dolphin OCR backend")
                return "dolphin"
            elif TESSERACT_AVAILABLE:
                logger.info("Auto-selected Tesseract OCR backend")
                return "tesseract"
            else:
                logger.warning("No OCR backend available")
                return "none"
        else:
            logger.warning(f"Requested OCR backend '{preferred}' not available, falling back to auto-selection")
            return self._select_ocr_backend("auto")
    
    def _init_dolphin_model(self):
        """Initialize Dolphin model and tokenizer"""
        if self._dolphin_model is None and DOLPHIN_AVAILABLE:
            try:
                # Use the Dolphin model from ByteDance
                model_name = "DOLPHIN-2_8-Llama3-70B"  # This might need adjustment based on actual model naming
                logger.info(f"Loading Dolphin model: {model_name}")
                self._dolphin_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._dolphin_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
                logger.info("Dolphin model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Dolphin model: {e}")
                logger.info("Falling back to Tesseract OCR")
                self.ocr_backend = "tesseract"
    
    def process_pdf_file(self, pdf_path: str, disclosure_id: int = 1) -> List[DocumentChunk]:
        """Process PDF file with enhanced chunking and categorization"""
        
        if not PDF_PROCESSING_AVAILABLE:
            logger.warning(f"pdfplumber not available, skipping PDF processing for {pdf_path}")
            return []
        
        try:
            chunks = []
            chunk_index = 0
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Calculate hash of PDF file for idempotency
            disclosure_hash = sha256_file(Path(pdf_path))
            source_file = os.path.basename(pdf_path)
            
            # Smart OCR detection - determine if this PDF needs OCR
            pdf_needs_ocr = self._should_use_ocr_for_pdf(pdf_path)
            if pdf_needs_ocr:
                logger.info(f"PDF classified as image-based - OCR enabled for {source_file}")
            else:
                logger.info(f"PDF classified as text-based - OCR disabled for {source_file}")
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try multiple text extraction methods
                    text = self._extract_text_from_page(page, page_num)
                    if not text:
                        logger.warning(f"No text found on page {page_num} using any method")
                        continue
                    
                    logger.info(f"Page {page_num} - Raw text length: {len(text)}")
                    logger.debug(f"Page {page_num} - Raw text preview: {text[:200]}...")
                    
                    # Clean and normalize text
                    cleaned_text = self._clean_text(text)
                    logger.info(f"Page {page_num} - Cleaned text length: {len(cleaned_text)}")
                    logger.debug(f"Page {page_num} - Cleaned text preview: {cleaned_text[:200]}...")
                    
                    # Smart OCR: Only use OCR when truly needed based on PDF type and content
                    if len(cleaned_text.strip()) < 3 and OCR_AVAILABLE and not self.disable_ocr and pdf_needs_ocr:
                        logger.info(f"Page {page_num} - No meaningful text extracted ({len(cleaned_text.strip())} chars), attempting OCR...")
                        ocr_text = self._extract_text_with_ocr(page, page_num)
                        if ocr_text and ocr_text.strip():
                            ocr_cleaned = self._clean_text(ocr_text)
                            if len(ocr_cleaned.strip()) > len(cleaned_text.strip()):
                                cleaned_text = ocr_cleaned
                                logger.info(f"Page {page_num} - OCR improved content, new length: {len(cleaned_text)}")
                            else:
                                logger.warning(f"Page {page_num} - OCR did not improve content quality")
                        else:
                            logger.warning(f"Page {page_num} - OCR extraction failed")
                    elif len(cleaned_text.strip()) < 3 and not pdf_needs_ocr:
                        logger.debug(f"Page {page_num} - Minimal text but PDF classified as text-based, skipping OCR")
                    
                    if len(cleaned_text.strip()) < 3:
                        logger.warning(f"Page {page_num} - Text too short after cleaning and OCR attempts, skipping")
                        continue
                    
                    # Extract heading information for enhanced categorization
                    heading_text = self._extract_heading_text(cleaned_text)
                    
                    # Categorize content type with heading-based enhancement
                    content_type = self._categorize_content(cleaned_text, page_num, heading_text)
                    logger.info(f"Page {page_num} - Content type: {content_type}" + 
                               (f", Heading: {heading_text[:50]}..." if heading_text else ""))
                    
                    # Chunk text semantically with enhanced XBRL-style processing
                    max_size = self._get_optimal_chunk_size(content_type, len(cleaned_text))
                    page_chunks = self._chunk_text_enhanced(cleaned_text, max_chunk_size=max_size)
                    logger.info(f"Page {page_num} - Generated {len(page_chunks)} enhanced chunks")
                    
                    for i, chunk_text in enumerate(page_chunks):
                        chunk_length = len(chunk_text.strip())
                        if chunk_length < 15:  # Skip only extremely short chunks
                            logger.debug(f"Page {page_num}, Chunk {i} - Too short ({chunk_length} chars), skipping")
                            continue
                        logger.debug(f"Page {page_num}, Chunk {i} - Length: {chunk_length}, Preview: {chunk_text[:100]}...")
                        
                        # Apply final normalization to ensure consistency
                        normalized_chunk_text = normalize_text(chunk_text)
                        
                        # Calculate chunk properties for unified format
                        char_length = len(normalized_chunk_text)
                        tokens = count_tokens(normalized_chunk_text)
                        is_numeric = is_numeric_content(normalized_chunk_text)
                        section_code = classify_pdf_section(normalized_chunk_text, page_num, content_type)
                        heading_text = extract_pdf_heading(normalized_chunk_text, page_num)
                        vectorize_flag = should_vectorize_chunk(
                            content=normalized_chunk_text,
                            content_type=content_type,
                            section_code=section_code,
                            char_length=char_length,
                            tokens=tokens,
                            is_numeric=is_numeric
                        )
                        
                        chunk = DocumentChunk(
                            disclosure_id=disclosure_id,
                            chunk_index=chunk_index,
                            content=normalized_chunk_text,
                            content_type=content_type,
                            section_code=section_code,
                            heading_text=heading_text,
                            char_length=char_length,
                            tokens=tokens,
                            vectorize=vectorize_flag,
                            is_numeric=is_numeric,
                            disclosure_hash=disclosure_hash,
                            source_file=source_file,
                            page_number=page_num,
                            metadata={
                                'page_number': page_num,
                                'extraction_method': 'pdf_extraction',
                                'language': 'ja',
                                'pdf_path': source_file,
                                'mecab_available': self.mecab is not None
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            
            # Apply duplicate heading block removal (enhanced from XBRL extractor)
            chunks = self._remove_duplicate_heading_blocks(chunks)
            logger.info(f"After heading deduplication: {len(chunks)} chunks")
            
            # Apply deduplication
            chunks = self._deduplicate_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Enhanced cleaning with XBRL-style normalization and row-number removal."""
        # 1) Apply enhanced normalization using the upgraded global function
        text = normalize_text(text)
        
        # 2) Remove row-number prefixes (MAJOR ADDITION from XBRL pipeline)
        # Multi-pass approach to catch all row-number patterns while preserving financial data
        text = self._remove_row_number_prefixes(text)

        # 2) Remove page headers/footers like  '- 24 -' or '24'
        # Remove from start/end of lines (original logic)
        text = re.sub(r'^\s*[-–—]+?\s*\d+\s*[-–—]+\s*$', '', text,
                      flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove embedded page footers in middle of text (e.g., "some text - 5 - more text")
        # Pattern: space/punctuation + dash + space + digits + space + dash + space
        text = re.sub(r'[\s\.\。]\s*[-–—]\s*\d+\s*[-–—]\s+', ' ', text)
        
        # More aggressive pattern for common embedded footers
        text = re.sub(r'\s+[-–—]\s*\d{1,2}\s*[-–—]\s+', ' ', text)
        
        # Additional patterns for persistent page footers
        # Remove page footers at end of content (before company name patterns)
        text = re.sub(r'\s+[-–—]\s*\d{1,2}\s*[-–—]\s+(?=nms|ホールディングス|株式会社)', ' ', text)
        
        # Remove page footers that follow periods or other sentence endings
        text = re.sub(r'[\.\。]\s+[-–—]\s*\d{1,2}\s*[-–—]\s+', '. ', text)

        # 5) REMOVED AGGRESSIVE COMPANY-SPECIFIC PATTERNS
        # These patterns were removing legitimate financial content
        # Keep only minimal document header cleaning
        
        # DISABLED - Enhanced header removal patterns were too aggressive
        # These patterns were removing legitimate financial content

        # 3) Remove redundant dot sequences (table of contents leaders, formatting dots)
        # Remove sequences of 4+ dots/periods (keeping shorter sequences that might be meaningful)
        text = re.sub(r'\.{4,}', '', text)
        
        # Remove lines that are mostly dots with optional spaces
        text = re.sub(r'^\s*[.\s]{10,}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove standalone sequences of dots mixed with spaces
        text = re.sub(r'\s+[.\s]{6,}\s+', ' ', text)

        # 6) Fix dangling parentheses and broken formatting
        # Fix dangling opening parentheses with spaces
        text = re.sub(r'\)\s*の\s*本割当株式', ')の本割当株式', text)
        text = re.sub(r'\)\s*、\s*', ')、', text)
        text = re.sub(r'\s+\(\s*', '(', text)  # Remove spaces before opening parentheses
        text = re.sub(r'\s+\)\s*', ') ', text)  # Normalize spaces after closing parentheses
        
        # Fix scattered numbers and currency formatting
        text = re.sub(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*円', r'\1,\2,\3円', text)
        text = re.sub(r'(\d+)\s+円', r'\1円', text)
        text = re.sub(r'(\d+)\s+株', r'\1株', text)
        text = re.sub(r'(\d+)\s+名', r'\1名', text)
        
        # Remove repeated fragments and normalize spacing
        text = re.sub(r'各\s+位\s+各\s+位', '各位', text)
        text = re.sub(r'株\s*式\s*会\s*社\s+株\s*式\s*会\s*社', '株式会社', text)

        # 4) Collapse excessive whitespace (already done in normalize_text, but ensure)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_row_number_prefixes(self, text: str) -> str:
        """
        Remove row-number prefixes from financial tables (from XBRL pipeline).
        
        This is a multi-pass approach to catch all row-number patterns while
        preserving legitimate financial data like negative values.
        """
        if not text:
            return text
        
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
        
        return text
    
    def _extract_text_from_page(self, page, page_num: int) -> str:
        """Extract text from a page using multiple methods"""
        extracted_texts = []
        
        # Method 1: Standard text extraction
        try:
            text1 = page.extract_text()
            if text1 and text1.strip():
                extracted_texts.append(text1)
        except Exception as e:
            logger.debug(f"Standard text extraction failed for page {page_num}: {e}")
        
        # Method 2: Layout-aware text extraction
        try:
            text2 = page.extract_text(layout=True)
            if text2 and text2.strip() and text2 not in extracted_texts:
                extracted_texts.append(text2)
        except Exception as e:
            logger.debug(f"Layout text extraction failed for page {page_num}: {e}")
        
        # Method 3: Table extraction
        try:
            tables = page.extract_tables()
            if tables:
                for i, table in enumerate(tables):
                    table_rows = []
                    for row in table:
                        if row:
                            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            if any(clean_row):  # Only add non-empty rows
                                table_rows.append(" ".join(clean_row))
                    if table_rows:
                        extracted_texts.append("\n".join(table_rows))
        except Exception as e:
            logger.debug(f"Table extraction failed for page {page_num}: {e}")
        
        # Method 4: Word-by-word extraction
        try:
            words = page.extract_words()
            if words:
                word_text = " ".join([word['text'] for word in words if word.get('text', '').strip()])
                if word_text and word_text not in " ".join(extracted_texts):
                    extracted_texts.append(word_text)
        except Exception as e:
            logger.debug(f"Word extraction failed for page {page_num}: {e}")
        
        # Combine all extracted text
        combined_text = "\n\n".join(extracted_texts) if extracted_texts else ""
        
        # This method is called before PDF-level smart detection, so we need to check
        # if OCR should be used based on available info. For now, keep conservative approach.
        # The main smart detection happens at the PDF level in process_pdf_file()
        if OCR_AVAILABLE and not combined_text.strip() and not self.disable_ocr:
            logger.info(f"Page {page_num} - No text extracted by any method, attempting OCR...")
            ocr_text = self._extract_text_with_ocr(page, page_num)
            if ocr_text and ocr_text.strip():
                combined_text = ocr_text
                logger.info(f"Page {page_num} - OCR extracted text, length: {len(combined_text)}")
            else:
                logger.warning(f"Page {page_num} - OCR also failed to extract text")
        elif not combined_text.strip():
            logger.warning(f"Page {page_num} - No text extracted and OCR not available")
        
        logger.info(f"Page {page_num} - Extracted {len(extracted_texts)} text segments, total length: {len(combined_text)}")
        
        return combined_text.strip() if combined_text else ""
    
    def _extract_text_with_ocr(self, page, page_num: int) -> str:
        """Extract text from page using OCR for image-based PDFs"""
        if not OCR_AVAILABLE:
            logger.warning(f"Page {page_num} - OCR not available")
            return ""
        
        try:
            # Convert page to image
            page_image = page.to_image(resolution=300)  # High resolution for better OCR
            pil_image = page_image.original
            
            # Use the selected OCR backend
            if self.ocr_backend == "dolphin":
                ocr_text = self._extract_with_dolphin(pil_image, page_num)
            elif self.ocr_backend == "tesseract":
                ocr_text = self._extract_with_tesseract(pil_image, page_num)
            else:
                logger.warning(f"Page {page_num} - No valid OCR backend selected")
                return ""
            
            if ocr_text and ocr_text.strip():
                # Clean OCR noise from the extracted text
                cleaned_ocr_text = self._clean_ocr_noise(ocr_text)
                logger.info(f"Page {page_num} - OCR ({self.ocr_backend}) successful, extracted {len(ocr_text)} characters, cleaned to {len(cleaned_ocr_text)} characters")
                return cleaned_ocr_text.strip()
            else:
                logger.warning(f"Page {page_num} - OCR ({self.ocr_backend}) completed but no text found")
                return ""
                
        except Exception as e:
            logger.error(f"Page {page_num} - OCR ({self.ocr_backend}) failed: {e}")
            return ""
    
    def _extract_with_tesseract(self, pil_image, page_num: int) -> str:
        """Extract text using Tesseract OCR"""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available")
        
        # Configure Tesseract for Japanese and English
        # Use both jpn and eng languages for better recognition of mixed content
        custom_config = r'--oem 3 --psm 6 -l jpn+eng'
        
        # Extract text using OCR
        ocr_text = pytesseract.image_to_string(pil_image, config=custom_config)
        logger.debug(f"Page {page_num} - Tesseract extracted {len(ocr_text)} characters")
        return ocr_text
    
    def _extract_with_dolphin(self, pil_image, page_num: int) -> str:
        """Extract text using Dolphin OCR"""
        if not DOLPHIN_AVAILABLE:
            raise RuntimeError("Dolphin not available")
        
        # Initialize Dolphin model if not already done
        if self._dolphin_model is None:
            self._init_dolphin_model()
        
        if self._dolphin_model is None:
            # Fallback to Tesseract if Dolphin failed to load
            logger.warning(f"Page {page_num} - Dolphin model not available, falling back to Tesseract")
            if TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(pil_image, page_num)
            else:
                raise RuntimeError("No OCR backend available")
        
        try:
            # Convert PIL image to format expected by Dolphin
            # Note: This is a placeholder - actual Dolphin integration will depend on their API
            # The actual implementation would need to follow Dolphin's documentation
            
            # For now, this is a conceptual structure - actual Dolphin API may differ
            # We would need to:
            # 1. Prepare the image in the format expected by Dolphin
            # 2. Create appropriate prompts for document parsing
            # 3. Use the model to generate structured document understanding
            
            logger.info(f"Page {page_num} - Processing with Dolphin VLM (placeholder implementation)")
            
            # Placeholder: In real implementation, this would use Dolphin's specific API
            # For now, fall back to Tesseract to maintain functionality
            logger.warning(f"Page {page_num} - Dolphin implementation pending, using Tesseract fallback")
            if TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(pil_image, page_num)
            else:
                return ""
            
        except Exception as e:
            logger.error(f"Page {page_num} - Dolphin processing failed: {e}")
            # Fallback to Tesseract
            if TESSERACT_AVAILABLE:
                logger.info(f"Page {page_num} - Falling back to Tesseract")
                return self._extract_with_tesseract(pil_image, page_num)
            else:
                raise
    
    def _clean_ocr_noise(self, text: str) -> str:
        """Clean OCR-specific noise and artifacts from extracted text"""
        if not text:
            return text
        
        # OCR noise patterns commonly found in financial documents
        ocr_noise_patterns = [
            # Random letter combinations (2-8 chars) that appear isolated
            r'\b[a-zA-Z]{2,8}\s+(?=[0-9億万円千百十])',  # Random letters before numbers/currency
            r'(?<=[0-9億万円千百十])\s+[a-zA-Z]{2,8}\b',  # Random letters after numbers/currency
            r'\b[a-zA-Z]{2,8}\s+(?=[の、。で])',         # Random letters before Japanese particles
            
            # Specific common OCR errors in financial docs
            r'\biRRE\s+BW\b',           # "iRRE BW" noise
            r'\ba\s+pes\b',             # "a pes" noise
            r'\bgar\.\b',               # "gar." noise
            r'\bep\s+i=\b',             # "ep i=" noise
            r'\bnoagl\s+\d+\b',         # "noagl 5" type noise
            r'\bsg\s+器\s+al\b',        # "sg 器 al" mixed noise
            r'\bzisl\s+id\b',           # "zisl id" noise
            r'\beh\s+nish\s+\d+\b',     # "eh nish 12" noise
            r'\biA\s+回\b',             # "iA 回" noise
            r'\bTT\s+\d+\s+[A-Z][a-z]+\b', # "TT 29 Cid" pattern
            r'\bHT\s+\d+\b',            # "HT 420" pattern
            
            # Table border and formatting artifacts
            r'\|\s*[a-zA-Z]{1,3}\s*\|',  # Single letters between pipes
            r'[。\s]+[a-zA-Z]{1,4}[。\s]+', # Random letters between Japanese punctuation
            r'\|\s*\d+\s*\|\s*',        # Single numbers between pipes
            r'\|\s*、\s*\|',            # Comma between pipes
            r'\s*\|\s*\|+\s*',          # Multiple consecutive pipes
            r'=\s*i=\s*',               # "= i=" pattern
            r'\|\s*\d{1,3}\|\s*',       # Short numbers with pipes
            r'開\s+売上高',             # Table header artifacts
            r'有力\s*\|',               # "有力 |" pattern
            r'人回藤想\s*\|',           # Garbled table headers
            r'\@\s*業績',               # "@ 業績" symbol noise
            r'\.{3,}《\.{2,}',          # "...《.." pattern
            r'_\s*\(',                  # "_ (" pattern
            r'増減額\d+-[A-Z]\)',       # "増減額6-A)" incomplete patterns
            r'今回修正予想[A-Z]\)',     # "今回修正予想B)" incomplete
            r'前発想\s+',               # "前発想" typo
            
            # Mixed character noise
            r'[a-zA-Z]\s*器\s*[a-zA-Z]',  # Letters around 器 character
            r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', # Letter-number-letter combinations
            
            # Standalone meaningless letter combinations
            r'\b[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{2,4}\b', # Consonant clusters
        ]
        
        # Apply noise removal patterns
        cleaned_text = text
        for pattern in ocr_noise_patterns:
            cleaned_text = re.sub(pattern, ' ', cleaned_text)
        
        # Clean up specific character-level OCR errors
        char_replacements = {
            '(B-A)': '(B-A)',     # Fix format
            '6-': '(B-A)',        # Common incomplete patterns  
            '増減額6-': '増減額(B-A)',  # Fix specific table patterns
            '今回修正予想B)': '今回修正予想(B)',  # Fix parentheses
            '前発想': '前回発表予想',   # Fix OCR typo
            '20 ': '% ',          # Percent signs often misread
            ' 阿部 ': ' ',        # Remove misread names if generic
        }
        
        for error, correction in char_replacements.items():
            cleaned_text = cleaned_text.replace(error, correction)
        
        # Enhanced table artifact cleaning
        # Remove scattered pipes and numbers that are table borders
        cleaned_text = re.sub(r'\|\s*\d{1,4}\s*\|', ' ', cleaned_text)  # Numbers in pipes
        cleaned_text = re.sub(r'\|\s*\|\s*\|+', ' ', cleaned_text)       # Multiple pipes
        cleaned_text = re.sub(r'\|\s*[、。]\s*\|', ' ', cleaned_text)    # Punctuation in pipes
        cleaned_text = re.sub(r'\s+\|\s+', ' ', cleaned_text)            # Isolated single pipes
        cleaned_text = re.sub(r'^\|\s*', '', cleaned_text)               # Pipes at start
        cleaned_text = re.sub(r'\s*\|$', '', cleaned_text)               # Pipes at end
        
        # Clean up table structure artifacts
        cleaned_text = re.sub(r'[。\s]{3,}', ' ', cleaned_text)          # Multiple dots/spaces
        cleaned_text = re.sub(r'\s+\d{1,3}\s+\|', ' ', cleaned_text)     # Numbers before pipes
        cleaned_text = re.sub(r'\|\s+\d{1,3}\s+', ' ', cleaned_text)     # Numbers after pipes
        
        # Clean up spacing issues caused by removals
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+([、。])', r'\1', cleaned_text)   # Space before punctuation
        
        return cleaned_text.strip()
    
    def _extract_heading_text(self, text: str) -> str:
        """
        Extract heading-like text from content for enhanced categorization.
        
        Uses patterns similar to XBRL pipeline to identify likely heading content
        that can improve content type classification accuracy.
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        heading_text = ""
        
        # Extract first 2-3 lines as potential heading content
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if not line:
                continue
                
            # Skip if line is too long (likely not a heading)
            if len(line) > 100:
                continue
                
            # Look for heading-like patterns
            heading_patterns = [
                r'^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\s]+[\u3002\uff1a\uff1f\uff01]?$',  # Japanese text
                r'^[\d\u3002\.\)\uff09\u3001]+\s*[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+',  # Numbered headings
                r'^\([^\)]+\)\s*[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+',  # Parenthetical headings
                r'^[\u3010\u3011\u300c\u300d\u300e\u300f]+.*[\u3011\u300d\u300f]+',  # Bracketed headings
            ]
            
            if any(re.match(pattern, line) for pattern in heading_patterns):
                heading_text += line + " "
                
        return heading_text.strip()
    
    def _is_numeric_content(self, text: str) -> bool:
        """
        Advanced numeric content detection with multi-criteria analysis.
        
        Enhanced from XBRL pipeline with comprehensive financial table detection
        using character analysis, pattern matching, and line-by-line evaluation.
        """
        if not text or len(text.strip()) < 10:
            return False
            
        # Character counting for analysis
        total_chars = len(text)
        digits = sum(1 for c in text if c.isdigit())
        financial_symbols = sum(1 for c in text if c in '.,|-△▲%')
        table_chars = sum(1 for c in text if c in '|:()[]')
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
        
        # Calculate ratios for analysis
        digit_ratio = digits / total_chars
        financial_ratio = financial_symbols / total_chars
        table_ratio = table_chars / total_chars
        japanese_ratio = japanese_chars / total_chars
        numeric_financial_ratio = (digits + financial_symbols) / total_chars
        
        # Multi-criteria decision logic (from XBRL pipeline)
        
        # Rule 1: High digit content
        if digit_ratio > 0.15:  # >15% digits
            return True
            
        # Rule 2: High financial symbol content  
        if financial_ratio > 0.10:  # >10% financial symbols
            return True
            
        # Rule 3: Combined numeric and financial content
        if numeric_financial_ratio > 0.20:  # >20% combined
            return True
            
        # Rule 4: Table structure with low Japanese content
        if table_ratio > 0.05 and japanese_ratio < 0.50:
            return True
            
        # Rule 5: Low Japanese with moderate numeric content
        if japanese_ratio < 0.25 and numeric_financial_ratio > 0.15:
            return True
            
        # Line-by-line analysis for table detection
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) >= 3:  # Need at least 3 lines for table analysis
            table_lines = 0
            for line in lines:
                line_chars = len(line)
                if line_chars == 0:
                    continue
                    
                line_digits = sum(1 for c in line if c.isdigit())
                line_financial = sum(1 for c in line if c in '.,|-△▲%')
                line_numeric_ratio = (line_digits + line_financial) / line_chars
                
                # Count as table line if >50% numeric content
                if line_numeric_ratio > 0.50:
                    table_lines += 1
                    
            # If >60% of lines are table-like, entire content is numeric
            if table_lines / len(lines) > 0.60:
                return True
                
        return False
    
    def _should_vectorize_chunk(self, content: str, content_type: str, section_code: str, 
                                char_length: int, tokens: int, is_numeric: bool) -> tuple[bool, str]:
        """
        Comprehensive quality filtering to determine if chunk should be vectorized.
        
        Enhanced from XBRL pipeline with multi-stage filtering including mega-table detection,
        IFRS transition removal, low-information detection, and duplicate filtering.
        
        Returns: (should_vectorize: bool, filter_reason: str)
        """
        # Rule 1: Fragment detection - too short chunks
        if char_length < 50 or tokens < 15:
            return False, "fragment"
            
        # Rule 2: Advanced numeric content detection
        if self._is_numeric_content(content):
            # Tiered filtering by chunk size (from XBRL pipeline)
            if char_length > 400 and is_numeric:  # Large chunks with confirmed numeric
                return False, "mostly_numeric_large"
            elif char_length > 200:  # Medium chunks
                # More lenient for medium chunks with financial data
                numeric_ratio = sum(1 for c in content if c.isdigit()) / len(content)
                if numeric_ratio > 0.35:
                    return False, "mostly_numeric_medium"
            elif char_length > 100:  # Small chunks
                numeric_ratio = sum(1 for c in content if c.isdigit()) / len(content)
                if numeric_ratio > 0.50:
                    return False, "mostly_numeric_small"
            else:  # Very small chunks
                numeric_ratio = sum(1 for c in content if c.isdigit()) / len(content)
                if numeric_ratio > 0.60:
                    return False, "mostly_numeric_tiny"
                    
        # Rule 3: Section-specific filtering
        if section_code in ['balance_sheet', 'cash_flow'] and is_numeric:
            numeric_ratio = sum(1 for c in content if c.isdigit()) / len(content)
            if numeric_ratio > 0.25:
                return False, "section_specific_numeric"
                
        # Rule 4: Table pattern detection
        table_chars = sum(1 for c in content if c in '|:0123456789.,()-△▲ \t')
        if table_chars / len(content) > 0.80:
            return False, "table_pattern"
            
        # Rule 5: Mega-table detection (balance sheets, cash flows >800 chars)
        if content_type in ['financial_position', 'financial_table'] and char_length > 800:
            return False, "mega_table"
            
        # Rule 6: Large equity statements
        if content_type == 'capital_policy' and char_length > 1000:
            pipe_count = content.count('|')
            if pipe_count > 15:  # Structured table
                return False, "mega_equity_table"
                
        # Rule 7: IFRS transition table removal
        if self._is_ifrs_transition_table(content, char_length):
            return False, "ifrs_transition"
            
        # Rule 8: Low-information content detection
        if self._is_low_information_content(content, char_length):
            return False, "low_information"
            
        # Rule 9: Continuation fragments (starts with table markers)
        content_clean = content.strip()
        if (content_clean.startswith('|') or 
            any(content_clean.startswith(f'{i}:') for i in range(2, 10))):
            return False, "continuation_fragment"
            
        # Rule 10: Very short chunks (final check)
        if char_length < 100:
            return False, "very_short"
            
        return True, "passed"
    
    def _is_ifrs_transition_table(self, content: str, char_length: int) -> bool:
        """
        Detect IFRS transition tables that should be filtered out.
        
        Enhanced from XBRL pipeline - detects specific IFRS migration content
        that typically contains large tables with limited semantic value.
        """
        if char_length < 800:  # Only check large content
            return False
            
        pipe_count = content.count('|')
        if pipe_count < 15:  # Need significant table structure
            return False
            
        # IFRS transition indicators (from XBRL pipeline)
        ifrs_indicators = [
            '日本基準', 'IFRS', '表示組替', '認識および測定の差異', 
            '調整額', '移行日', '国際財務報告基準', '会計基準の変更',
            '移行前', '移行後', '基準変更', '会計方針の変更'
        ]
        
        indicator_count = sum(1 for indicator in ifrs_indicators if indicator in content)
        
        # High density of IFRS terms + large table structure = transition table
        return indicator_count >= 3
    
    def _is_low_information_content(self, content: str, char_length: int) -> bool:
        """
        Detect low-information content that adds little semantic value.
        
        Enhanced from XBRL pipeline with pattern detection for repetitive content,
        formatting indicators, and pure date range content.
        """
        # Rule 1: Repetitive content detection
        words = content.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.30:  # <30% unique words
                return True
                
        # Rule 2: Formatting indicators for short content
        if char_length < 200:
            formatting_patterns = [
                '(単位:', '百万円)', '千円)', '億円)', 
                '前連結会計年度', '当連結会計年度',
                '自 ', '至 ', '期間'
            ]
            
            pattern_count = sum(1 for pattern in formatting_patterns if pattern in content)
            if pattern_count >= 2:
                return True
                
        # Rule 3: Pure date range content
        import re
        date_pattern = r'\d{4}年\d{1,2}月\d{1,2}日.*\d{4}年\d{1,2}月\d{1,2}日'
        if re.search(date_pattern, content) and char_length < 150:
            return True
            
        return False
    
    def _categorize_content(self, text: str, page_num: int, heading_text: str = "") -> str:
        """Categorize content type based on text patterns with enhanced financial granularity
        
        Enhanced categorization system upgraded from XBRL pipeline with 12 financial-specific types.
        Uses dual heading-based and content-based classification for better accuracy.
        """
        # Clean text and heading for analysis
        text_clean = re.sub(r'\s+', ' ', text.lower())
        heading_clean = re.sub(r'\s+', ' ', heading_text.lower()) if heading_text else ""
        
        # PHASE 2 ENHANCEMENT: 12 Financial-Specific Content Types (from XBRL pipeline)
        
        # 1. Forward-looking statements and forecasts
        if any(keyword in heading_clean for keyword in ['見通し', '予想', '予測', '見込み', '計画', '目標']) or \
           any(keyword in text_clean for keyword in ['次連結会計年度', '来期', '予想', '見込み', '予測', '次期']):
            return 'forecast'
        
        # 2. Capital allocation and shareholder returns
        if any(keyword in heading_clean for keyword in ['配当', '自己株式', '株主還元', '資本政策', '後発事象']) or \
           any(keyword in text_clean for keyword in ['配当', '自己株式', '株主還元', '資本効率', '株式分割']):
            return 'capital_policy'
        
        # 3. Risk management and hedging
        if any(keyword in heading_clean for keyword in ['リスク', 'ヘッジ', 'リスク管理']) or \
           any(keyword in text_clean for keyword in ['ヘッジ', 'リスク', 'デリバティブ', '金利リスク', '為替リスク']):
            return 'risk_management'
        
        # 4. Accounting policies and standards
        if any(keyword in heading_clean for keyword in ['会計', '基準', 'ifrs', '日本基準', '測定', '認識']) or \
           any(keyword in text_clean for keyword in ['会計処理', '測定方法', 'ifrs', '会計基準', '償却', '減価償却']):
            return 'accounting_policy'
        
        # 5. Business segment analysis
        if any(keyword in heading_clean for keyword in ['セグメント', '事業', '損害保険', '生命保険', '介護']) or \
           any(keyword in text_clean for keyword in ['セグメント', '事業部門', '報告セグメント', '保険事業']):
            return 'segment_analysis'
        
        # 6. Per-share metrics and shareholder information
        if any(keyword in heading_clean for keyword in ['1株当たり', '株当たり', 'eps']) or \
           any(keyword in text_clean for keyword in ['1株当たり', '期中平均', '希薄化', '株式数']):
            return 'per_share_metrics'
        
        # 7. Geographical/regional analysis
        if any(keyword in heading_clean for keyword in ['地域', '国内', '海外', '収益']) or \
           any(keyword in text_clean for keyword in ['日本', '海外', '国内', '所在地', '地域別']):
            return 'geographical_analysis'
        
        # 8. Management performance commentary
        if any(keyword in heading_clean for keyword in ['経営成績', '業績', '財政状態']) or \
           any(keyword in text_clean for keyword in ['世界経済', '経営環境', '市場環境', '業績']):
            return 'management_discussion'
        
        # 9. Financial position and cash flows
        if any(keyword in heading_clean for keyword in ['財政状態', 'キャッシュ', '資産', '負債']) or \
           any(keyword in text_clean for keyword in ['資産合計', '資本合計', 'キャッシュフロー', '現金']):
            return 'financial_position'
        
        # 10. Financial performance metrics
        if any(keyword in text_clean for keyword in ['売上', '利益', '収益', '損失', '百万円', '億円']):
            return 'financial_metrics'
        
        # 11. Regulatory and compliance
        if any(keyword in heading_clean for keyword in ['継続企業', '注記', '法人税', '税率']) or \
           any(keyword in text_clean for keyword in ['法律', '規制', '適時開示', '法人税法']):
            return 'regulatory_compliance'
        
        # 12. Tables and structured financial data (enhanced detection)
        table_indicators = ['株式', '割当', '譲渡制限', '取締役', '監査等委員', '報酬', '新株予約権']
        table_patterns = [r'株\s*数', r'金\s*額', r'[0-9,]+\s*円', r'[0-9,]+\s*株']
        if ('|' in text_clean and any(pattern in text_clean for pattern in [r'\d+', '百万円', '千円'])) or \
           (any(indicator in text_clean for indicator in table_indicators) and 
            any(re.search(pattern, text_clean) for pattern in table_patterns)):
            return 'financial_table'
        
        # 13. Corporate governance (enhanced from organizational)
        if any(keyword in text_clean for keyword in ['取締役', 'ガバナンス', '経営陣', '監査', '組織変更', '人事異動']):
            return 'corporate_governance'
        
        # Legacy fallback categories for backward compatibility
        
        # Business overview and operations
        business_keywords = ['事業', '業務', '製品', 'サービス', '市場', '顧客', '販売']
        if any(keyword in text_clean for keyword in business_keywords):
            return 'business_overview'
        
        # Notes and supplementary information
        note_keywords = ['注記', '補足', '説明', '詳細', '参考']
        if any(keyword in text_clean for keyword in note_keywords) or page_num > 10:
            return 'notes'
        
        # Header/footer content (should be mostly removed by cleaning)
        if (len(text_clean) < 100 and 
            any(pattern in text_clean for pattern in ['各位', '会社名', '代表者', 'コード'])):
            return 'header_footer'
        
        return 'general'
    
    def _get_optimal_chunk_size(self, content_type: str, text_length: int) -> int:
        """Determine optimal chunk size based on content type and text length"""
        
        # Base sizes for different content types
        size_map = {
            'table': 800,                    # Tables can be larger to preserve structure
            'financial_summary': 600,        # Financial data benefits from larger chunks
            'organizational': 400,           # Org charts are usually concise
            'regulatory': 500,               # Legal text is moderately sized
            'management_discussion': 700,    # Strategy discussions can be longer
            'risk_factors': 500,             # Risk factors are usually itemized
            'business_overview': 600,        # Business descriptions can be detailed
            'equity_matters': 500,           # Stock-related content is structured
            'notes': 400,                    # Notes are usually brief
            'header_footer': 200,            # Headers/footers should be small
            'general': 500                   # Default size
        }
        
        base_size = size_map.get(content_type, 500)
        
        # Adjust based on total text length
        if text_length < 200:
            # Very short text - don't split unnecessarily
            return text_length
        elif text_length < 800:
            # Short text - use smaller chunks
            return min(base_size, 400)
        elif text_length > 3000:
            # Very long text - use larger chunks to reduce fragmentation
            return int(min(base_size * 1.3, 900))
        
        return base_size
    
    def _chunk_text_semantically(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Chunk text semantically using sentence boundaries with overlap prevention"""
        
        logger.debug(f"Chunking text of length {len(text)}")
        
        # Japanese sentence boundary patterns
        sentence_endings = ['。', '！', '？', '．']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        logger.debug(f"Found {len(sentences)} sentences")
        
        # If no sentence boundaries found, treat as one large text block and split by length
        if len(sentences) <= 1:
            logger.debug("No sentence boundaries found, splitting by paragraphs and length")
            # Try splitting by paragraphs or line breaks
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                sentences = paragraphs
                logger.debug(f"Split into {len(sentences)} paragraphs")
            else:
                # Last resort: split by character count with no overlap
                sentences = []
                for i in range(0, len(text), max_chunk_size):
                    chunk = text[i:i+max_chunk_size].strip()
                    if chunk:
                        sentences.append(chunk)
                logger.debug(f"Split into {len(sentences)} character-based chunks")
        
        # Combine sentences into chunks with improved overlap prevention
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would create a chunk that's too large
            potential_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Post-process to ensure no overlaps and minimum chunk sizes
        filtered_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= 30:  # Minimum meaningful chunk size
                # Check for substantial overlap with previous chunk
                if filtered_chunks:
                    last_chunk = filtered_chunks[-1]
                    # Simple overlap check - avoid chunks that are largely contained in previous
                    if not (chunk in last_chunk or last_chunk in chunk):
                        filtered_chunks.append(chunk)
                    else:
                        logger.debug(f"Skipped overlapping chunk during chunking: {chunk[:50]}...")
                else:
                    filtered_chunks.append(chunk)
        
        logger.debug(f"Generated {len(filtered_chunks)} final chunks (removed {len(chunks) - len(filtered_chunks)} overlaps)")
        return filtered_chunks
    
    def _chunk_text_enhanced(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Enhanced chunking with XBRL-style financial pattern recognition and bullet point handling"""
        
        logger.debug(f"Enhanced chunking text of length {len(text)} with max size {max_chunk_size}")
        
        # Convert character limit to approximate token limit for better consistency with XBRL
        # Rough estimate: 1 token ≈ 3-4 chars for mixed Japanese/English financial text
        max_tokens = int(max_chunk_size / 3.5)
        
        # Use enhanced chunking logic adapted from XBRL
        if len(text) > max_tokens * 4:  # ~2000+ chars, use smart splitting
            logger.debug("Using smart split for long text")
            chunks = self._smart_split_long_text(text, max_tokens)
        else:
            logger.debug("Using simple chunking for shorter text")
            chunks = self._simple_chunk_with_overlap(text, max_tokens)
        
        # Apply size limit enforcement (from XBRL)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                logger.debug(f"Enforcing size limit on {len(chunk)}-char chunk")
                sub_chunks = self._enforce_size_limit_chunking(chunk, max_chunk_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Filter out very short chunks
        filtered_chunks = [chunk for chunk in final_chunks if len(chunk.strip()) >= 30]
        
        logger.debug(f"Enhanced chunking: Generated {len(filtered_chunks)} final chunks (size enforcement: {len(chunks)} → {len(final_chunks)}, filtered: {len(final_chunks)} → {len(filtered_chunks)})")
        
        return filtered_chunks
    
    def _smart_split_long_text(self, text: str, max_tokens: int = 150) -> List[str]:
        """Enhanced from XBRL: Split long text on natural boundaries with financial pattern recognition"""
        
        # Enhanced bullet point patterns for Japanese financial documents
        bullet_patterns = [
            r'主に次の差異があります。',
            r'主な項目は次のとおり',
            r'以下の.*について',
            r'次のとおり.*あります',
            r'主な内容は以下の.*です',
            r'詳細は以下の.*となります',
            r'具体的には.*以下の.*です'
        ]
        
        # Try splitting on financial bullet point patterns first
        for pattern in bullet_patterns:
            if re.search(pattern, text):
                logger.debug(f"Found financial bullet pattern: {pattern}")
                # Split on bullet-like patterns (enhanced from XBRL)
                parts = re.split(r'・(?=[^・])', text)  # Split on bullet points
                if len(parts) > 1:
                    logger.debug(f"Split into {len(parts)} bullet-point parts")
                    return self._group_parts_into_chunks(parts, max_tokens)
        
        # Try splitting on paragraph breaks
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # Enhanced Japanese sentence boundary detection (from XBRL)
            # Look for sentence endings followed by space and financial/section markers
            paragraphs = re.split(r'(?<=。)\s+(?=[IFRS第|日本基準|当社グループ|.*について|.*項目|.*事項])', text)
            if len(paragraphs) == 1:
                # Try splitting on periods + space (more general)
                paragraphs = [s.strip() + '.' for s in text.split('. ') if s.strip()]
                if paragraphs and paragraphs[-1].endswith('..'):
                    paragraphs[-1] = paragraphs[-1][:-1]  # Remove double period
        
        # Group paragraphs into chunks
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
                    sub_chunks = self._force_split_paragraph(para, max_tokens)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _simple_chunk_with_overlap(self, text: str, max_tokens: int = 150) -> List[str]:
        """Simple chunking with overlap for shorter texts"""
        
        overlap = min(50, max_tokens // 4)  # 25% overlap, max 50 words
        
        words = text.split()
        if len(words) <= max_tokens:
            return [text]
            
        chunks = []
        idx = 0
        while idx < len(words):
            chunk_words = words[idx : idx + max_tokens]
            chunks.append(" ".join(chunk_words))
            idx += max_tokens - overlap
            
        return chunks
    
    def _group_parts_into_chunks(self, parts: List[str], max_tokens: int = 150) -> List[str]:
        """Group text parts into chunks respecting token limits (from XBRL)"""
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
                    sub_chunks = self._force_split_paragraph(part, max_tokens)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _force_split_paragraph(self, text: str, max_tokens: int = 150) -> List[str]:
        """Force split a long paragraph using Japanese sentence patterns (from XBRL)"""
        
        # Enhanced Japanese sentence boundary detection
        sentences = []
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
        
        # Group sentences into chunks
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
                    overlap = min(20, max_tokens // 4)  # 25% overlap, max 20 words
                    for i in range(0, len(words), max_tokens - overlap):
                        chunk_words = words[i:i + max_tokens]
                        chunks.append(" ".join(chunk_words))
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _enforce_size_limit_chunking(self, text: str, max_chars: int = 600) -> List[str]:
        """Force split text that exceeds size limit, preserving meaning (from XBRL)"""
        
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
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.80) -> List[DocumentChunk]:
        """Remove duplicate and near-duplicate chunks based on content with enhanced detection"""
        if not chunks:
            return chunks
        
        logger.info(f"Starting deduplication of {len(chunks)} chunks with threshold {similarity_threshold}")
        
        # Step 0: Remove self-duplicates within chunks (repeated paragraphs in same chunk)
        for chunk in chunks:
            chunk.content = self._remove_self_duplicates(chunk.content)
        
        # Step 1: Remove exact duplicates using hash
        seen_content = set()
        exact_deduplicated = []
        exact_removed = 0
        
        for chunk in chunks:
            content_hash = hash(chunk.content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                exact_deduplicated.append(chunk)
            else:
                exact_removed += 1
                logger.debug(f"Removed exact duplicate chunk {chunk.chunk_index} (page {chunk.page_number}): \"{chunk.content[:50]}...\"")
        
        if exact_removed > 0:
            logger.info(f"Exact deduplication: Removed {exact_removed} duplicate chunks")
        
        # Step 2: Enhanced near-duplicate detection with substring detection
        import difflib
        
        final_chunks = []
        near_removed = 0
        substring_removed = 0
        
        # Sort chunks by length (longer first) to prioritize keeping longer, more complete chunks
        sorted_chunks = sorted(exact_deduplicated, key=lambda x: len(x.content), reverse=True)
        
        for i, chunk in enumerate(sorted_chunks):
            is_duplicate = False
            duplicate_reason = ""
            
            # Compare with already accepted chunks
            for accepted_chunk in final_chunks:
                # Method 1: Check for substring containment (one chunk completely contains another)
                chunk_content = chunk.content.strip()
                accepted_content = accepted_chunk.content.strip()
                
                # Skip very short content to avoid false positives
                if len(chunk_content) < 30 or len(accepted_content) < 30:
                    continue
                
                # Check if current chunk is completely contained in accepted chunk
                if chunk_content in accepted_content and len(chunk_content) < len(accepted_content) * 0.95:
                    is_duplicate = True
                    substring_removed += 1
                    duplicate_reason = f"substring (contained in chunk {accepted_chunk.chunk_index})"
                    logger.debug(f"Removed substring duplicate chunk {chunk.chunk_index} (page {chunk.page_number}): contained in chunk {accepted_chunk.chunk_index}")
                    break
                
                # Check if accepted chunk is completely contained in current chunk (keep current, remove accepted)
                elif accepted_content in chunk_content and len(accepted_content) < len(chunk_content) * 0.95:
                    # Remove the accepted chunk and add current chunk instead
                    final_chunks.remove(accepted_chunk)
                    substring_removed += 1
                    logger.debug(f"Replaced chunk {accepted_chunk.chunk_index} with larger chunk {chunk.chunk_index} (page {chunk.page_number})")
                    # Continue checking against remaining accepted chunks
                    continue
                
                # Method 2: Calculate similarity for near-duplicates
                # Skip if lengths are very different (quick filter)
                len_ratio = min(len(chunk_content), len(accepted_content)) / max(len(chunk_content), len(accepted_content))
                if len_ratio < 0.6:
                    continue
                
                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, chunk_content, accepted_content).ratio()
                
                # Use adaptive thresholds based on content type and length
                adaptive_threshold = self._get_adaptive_threshold(chunk, accepted_chunk, similarity_threshold)
                
                if similarity > adaptive_threshold:
                    is_duplicate = True
                    near_removed += 1
                    duplicate_reason = f"similarity {similarity:.3f} > {adaptive_threshold:.3f}"
                    logger.debug(f"Removed near-duplicate chunk {chunk.chunk_index} (page {chunk.page_number}, {duplicate_reason}): \"{chunk.content[:50]}...\"")
                    break
            
            if not is_duplicate:
                final_chunks.append(chunk)
        
        # Step 3: Final pass to remove any remaining overlaps within same page
        final_chunks = self._remove_intra_page_overlaps(final_chunks)
        
        # Re-index the remaining chunks to maintain sequential order
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
        
        total_removed = exact_removed + near_removed + substring_removed
        if total_removed > 0:
            logger.info(f"Total deduplication: Removed {total_removed} chunks ({exact_removed} exact, {near_removed} similar, {substring_removed} substring), kept {len(final_chunks)} unique chunks")
        else:
            logger.info("Deduplication: No duplicate chunks found")
        
        return final_chunks
    
    def _remove_self_duplicates(self, text: str) -> str:
        """Remove repeated paragraphs and sentences within the same text"""
        if not text or len(text) < 100:
            return text
        
        # Split into sentences for analysis
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in ['。', '！', '？', '．']:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        if len(sentences) < 3:
            return text
        
        # Remove exact duplicate sentences
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence_clean = re.sub(r'\s+', ' ', sentence.strip())
            if len(sentence_clean) < 20:  # Skip very short sentences
                unique_sentences.append(sentence)
                continue
                
            if sentence_clean not in seen_sentences:
                seen_sentences.add(sentence_clean)
                unique_sentences.append(sentence)
            else:
                logger.debug(f"Removed self-duplicate sentence: {sentence_clean[:50]}...")
        
        # Additional check for repeated blocks (common in tables)
        result_text = " ".join(unique_sentences)
        
        # Look for repeated blocks of 50+ characters
        import difflib
        
        # Split into chunks for block comparison
        text_chunks = [result_text[i:i+100] for i in range(0, len(result_text), 50)]
        unique_blocks = []
        
        for i, chunk in enumerate(text_chunks):
            is_duplicate = False
            for j, existing_block in enumerate(unique_blocks):
                if len(chunk) > 50 and len(existing_block) > 50:
                    similarity = difflib.SequenceMatcher(None, chunk, existing_block).ratio()
                    if similarity > 0.9:
                        is_duplicate = True
                        logger.debug(f"Removed repeated block: {chunk[:30]}...")
                        break
            
            if not is_duplicate:
                unique_blocks.append(chunk)
        
        # Reconstruct if significant duplicates were found
        if len(unique_blocks) < len(text_chunks) * 0.8:
            # Attempt reconstruction from unique blocks
            reconstructed = "".join(unique_blocks)
            # Clean up potential boundary issues
            reconstructed = re.sub(r'\s{2,}', ' ', reconstructed)
            return reconstructed.strip()
        
        return result_text
    
    def _get_adaptive_threshold(self, chunk1: DocumentChunk, chunk2: DocumentChunk, base_threshold: float) -> float:
        """Calculate adaptive similarity threshold based on content characteristics"""
        
        # Lower threshold for financial summaries (more likely to have similar structure)
        if chunk1.content_type == 'financial_summary' and chunk2.content_type == 'financial_summary':
            return base_threshold - 0.05
        
        # Higher threshold for notes and general content (more diverse)
        if chunk1.content_type in ['notes', 'general'] or chunk2.content_type in ['notes', 'general']:
            return base_threshold + 0.05
        
        # Lower threshold for very short chunks (more likely to be duplicated)
        avg_length = (len(chunk1.content) + len(chunk2.content)) / 2
        if avg_length < 100:
            return base_threshold - 0.1
        
        return base_threshold
    
    def _remove_intra_page_overlaps(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove overlapping chunks within the same page"""
        if not chunks:
            return chunks
        
        # Group chunks by page
        page_groups = {}
        for chunk in chunks:
            page_num = chunk.page_number
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(chunk)
        
        final_chunks = []
        overlap_removed = 0
        
        for page_num, page_chunks in page_groups.items():
            if len(page_chunks) <= 1:
                final_chunks.extend(page_chunks)
                continue
            
            # Sort chunks by their original order (chunk_index)
            page_chunks.sort(key=lambda x: x.chunk_index)
            
            # Check for significant overlaps between consecutive chunks
            filtered_page_chunks = [page_chunks[0]]  # Always keep first chunk
            
            for i in range(1, len(page_chunks)):
                current_chunk = page_chunks[i]
                previous_chunk = filtered_page_chunks[-1]
                
                # Check for overlap between consecutive chunks
                overlap_ratio = self._calculate_overlap_ratio(previous_chunk.content, current_chunk.content)
                
                if overlap_ratio > 0.6:  # Significant overlap
                    # Keep the longer chunk
                    if len(current_chunk.content) > len(previous_chunk.content):
                        filtered_page_chunks[-1] = current_chunk
                    overlap_removed += 1
                    logger.debug(f"Removed overlapping chunk {current_chunk.chunk_index} on page {page_num} (overlap: {overlap_ratio:.3f})")
                else:
                    filtered_page_chunks.append(current_chunk)
            
            final_chunks.extend(filtered_page_chunks)
        
        if overlap_removed > 0:
            logger.info(f"Intra-page overlap removal: Removed {overlap_removed} overlapping chunks")
        
        return final_chunks
    
    def _calculate_overlap_ratio(self, text1: str, text2: str) -> float:
        """Calculate the overlap ratio between two text chunks"""
        import difflib
        
        # Use longest common subsequence to find overlap
        matcher = difflib.SequenceMatcher(None, text1, text2)
        matching_blocks = matcher.get_matching_blocks()
        
        total_overlap = sum(block.size for block in matching_blocks)
        min_length = min(len(text1), len(text2))
        
        if min_length == 0:
            return 0.0
        
        return total_overlap / min_length
    
    def _remove_duplicate_heading_blocks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove chunks that share the same heading (likely table fragments)
        
        Enhanced from XBRL extractor to handle PDF-specific patterns:
        - Balance sheet fragments split across pages
        - Cash flow statement continuations  
        - Income statement breakdowns with identical headings
        - Financial table fragments with same section headers
        """
        if not chunks:
            return chunks
        
        logger.info(f"Checking for duplicate heading blocks in {len(chunks)} chunks")
        
        # Group chunks by normalized heading
        from collections import defaultdict
        heading_groups = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            # Normalize heading for better matching
            heading = self._normalize_heading_for_comparison(chunk.heading_text)
            heading_groups[heading].append((i, chunk))
        
        # Process each heading group
        filtered_chunks = []
        removed_count = 0
        
        for heading, chunk_list in heading_groups.items():
            if len(chunk_list) == 1:
                # Single chunk with this heading - keep it
                filtered_chunks.append(chunk_list[0][1])
            else:
                # Multiple chunks with same heading - analyze for quality
                decision = self._analyze_duplicate_heading_group(heading, chunk_list)
                
                if decision['action'] == 'remove_all':
                    # Remove all chunks (likely table fragments)
                    removed_count += len(chunk_list)
                    logger.debug(f"Removing {len(chunk_list)} chunks with heading '{heading}' ({decision['reason']})")
                    
                elif decision['action'] == 'keep_best':
                    # Keep the best chunk, remove others
                    best_chunk = decision['best_chunk']
                    filtered_chunks.append(best_chunk)
                    removed_count += len(chunk_list) - 1
                    logger.debug(f"Keeping best chunk for heading '{heading}', removing {len(chunk_list)-1} duplicates ({decision['reason']})")
                    
                elif decision['action'] == 'keep_all':
                    # Keep all chunks (different contexts)
                    for _, chunk in chunk_list:
                        filtered_chunks.append(chunk)
                    logger.debug(f"Keeping all {len(chunk_list)} chunks for heading '{heading}' ({decision['reason']})")
        
        # Re-index remaining chunks to maintain sequence
        for idx, chunk in enumerate(filtered_chunks):
            chunk.chunk_index = idx
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate heading chunks")
        else:
            logger.info("No duplicate heading blocks found")
            
        return filtered_chunks
    
    def _normalize_heading_for_comparison(self, heading: str) -> str:
        """Normalize heading text for better duplicate detection"""
        if not heading:
            return "no_heading"
        
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', heading.strip())
        
        # Remove page numbers and common prefixes/suffixes
        normalized = re.sub(r'\s*\(?\d+\)?\s*$', '', normalized)  # Remove trailing page numbers
        normalized = re.sub(r'^[\d\s\-\.\(\)]+', '', normalized)  # Remove leading numbers/formatting
        
        # Normalize common financial terms for better matching
        financial_normalizations = {
            '貸借対照表': 'balance_sheet',
            'バランスシート': 'balance_sheet', 
            '損益計算書': 'income_statement',
            'キャッシュフロー': 'cash_flow',
            'キャッシュ・フロー': 'cash_flow',
            '株主資本': 'equity',
            '資本': 'equity',
            '連結': 'consolidated',
            '単体': 'standalone'
        }
        
        for japanese, english in financial_normalizations.items():
            normalized = normalized.replace(japanese, english)
        
        return normalized.lower()
    
    def _analyze_duplicate_heading_group(self, heading: str, chunk_list: List[tuple]) -> Dict[str, Any]:
        """Analyze a group of chunks with the same heading to decide what to do
        
        Returns:
            {
                'action': 'remove_all' | 'keep_best' | 'keep_all',
                'reason': str,
                'best_chunk': DocumentChunk (if action is 'keep_best')
            }
        """
        # Extract chunks for analysis
        chunks = [chunk for _, chunk in chunk_list]
        
        # Calculate group statistics
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chars = total_chars / len(chunks)
        
        # Analyze content characteristics
        numeric_ratios = []
        page_numbers = set()
        section_codes = set()
        
        for chunk in chunks:
            # Calculate numeric content ratio
            content = chunk.content
            digit_count = sum(1 for c in content if c.isdigit())
            pipe_count = content.count('|')
            numeric_ratio = (digit_count + pipe_count) / len(content) if content else 0
            numeric_ratios.append(numeric_ratio)
            
            # Track diversity
            page_numbers.add(chunk.page_number)
            section_codes.add(chunk.section_code)
        
        avg_numeric_ratio = sum(numeric_ratios) / len(numeric_ratios)
        
        # Decision logic based on content analysis
        
        # Rule 1: High numeric content suggests table fragments - remove all
        if len(chunks) > 2 and avg_numeric_ratio > 0.20 and avg_chars > 300:
            return {
                'action': 'remove_all',
                'reason': f'table_fragments (avg {avg_numeric_ratio:.1%} numeric, {avg_chars:.0f} chars)'
            }
        
        # Rule 2: Very similar large chunks across different pages - likely duplicated content
        if len(chunks) > 1 and avg_chars > 500 and len(page_numbers) > 1:
            # Check content similarity between chunks
            similarities = []
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    overlap = self._calculate_overlap_ratio(chunks[i].content, chunks[j].content)
                    similarities.append(overlap)
            
            if similarities and max(similarities) > 0.80:
                return {
                    'action': 'remove_all',
                    'reason': f'cross_page_duplicates (max similarity {max(similarities):.1%})'
                }
        
        # Rule 3: Multiple small chunks with same heading - keep the most informative one
        if len(chunks) > 2 and avg_chars < 200:
            # Find chunk with highest information content
            best_chunk = max(chunks, key=lambda c: len(c.content) * (1 - self._is_numeric_content(c.content)))
            return {
                'action': 'keep_best',
                'reason': f'small_fragments (keeping most informative: {len(best_chunk.content)} chars)',
                'best_chunk': best_chunk
            }
        
        # Rule 4: Different section codes suggest different contexts - keep all
        if len(section_codes) > 1:
            return {
                'action': 'keep_all',
                'reason': f'different_contexts ({len(section_codes)} section codes)'
            }
        
        # Rule 5: Chunks from same page with high numeric content - likely table splits
        if len(page_numbers) == 1 and avg_numeric_ratio > 0.15:
            return {
                'action': 'remove_all',
                'reason': f'same_page_table_splits (page {list(page_numbers)[0]}, {avg_numeric_ratio:.1%} numeric)'
            }
        
        # Default: Keep the longest/most complete chunk
        best_chunk = max(chunks, key=lambda c: len(c.content))
        return {
            'action': 'keep_best',
            'reason': f'default_dedup (keeping longest: {len(best_chunk.content)} chars)',
            'best_chunk': best_chunk
        }

class PDFExtractionTester:
    """Test harness for PDF extraction pipeline"""
    
    def __init__(self, output_dir: str = "test_output", ocr_backend: str = "auto", disable_ocr: bool = False, smart_ocr: bool = True):
        self.processor = PDFProcessor(ocr_backend=ocr_backend, disable_ocr=disable_ocr, smart_ocr=smart_ocr)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def test_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Test processing of a single PDF file"""
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {}
        
        logger.info(f"Testing PDF: {pdf_path}")
        
        # Extract raw text for debugging
        raw_text_by_page = self.extract_raw_text(pdf_path)
        self.save_raw_text(raw_text_by_page, pdf_path)
        
        chunks = self.processor.process_pdf_file(pdf_path)
        
        # Generate summary statistics
        stats = self._generate_stats(chunks, pdf_path)
        
        # Save results
        self._save_results(chunks, stats, pdf_path)
        
        return stats
    
    def extract_raw_text(self, pdf_path: str) -> Dict[int, str]:
        """Extract raw text from PDF using multiple methods for debugging"""
        raw_text_by_page = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try multiple extraction methods
                    extracted_texts = []
                    
                    # Method 1: Standard text extraction
                    text1 = page.extract_text()
                    if text1:
                        extracted_texts.append(f"[STANDARD METHOD]\n{text1}")
                    
                    # Method 2: Extract text with layout preservation
                    text2 = page.extract_text(layout=True)
                    if text2 and text2 != text1:
                        extracted_texts.append(f"[LAYOUT METHOD]\n{text2}")
                    
                    # Method 3: Try extracting tables
                    try:
                        tables = page.extract_tables()
                        if tables:
                            table_text = []
                            for i, table in enumerate(tables):
                                table_text.append(f"[TABLE {i+1}]")
                                for row in table:
                                    if row:
                                        clean_row = [str(cell) if cell is not None else "" for cell in row]
                                        table_text.append(" | ".join(clean_row))
                            extracted_texts.append("\n".join(table_text))
                    except Exception as e:
                        logger.debug(f"Table extraction failed for page {page_num}: {e}")
                    
                    # Method 4: Extract words with positions (for complex layouts)
                    try:
                        words = page.extract_words()
                        if words:
                            word_text = " ".join([word['text'] for word in words])
                            if word_text and word_text not in str(extracted_texts):
                                extracted_texts.append(f"[WORDS METHOD]\n{word_text}")
                    except Exception as e:
                        logger.debug(f"Word extraction failed for page {page_num}: {e}")
                    
                    # Combine all extracted text
                    if extracted_texts:
                        raw_text_by_page[page_num] = "\n\n".join(extracted_texts)
                    else:
                        raw_text_by_page[page_num] = "[NO TEXT EXTRACTED BY ANY METHOD]"
                        
                        # Try to get some basic info about the page
                        try:
                            page_info = {
                                'width': page.width,
                                'height': page.height,
                                'objects': len(page.objects),
                                'chars': len(page.chars) if hasattr(page, 'chars') else 0,
                                'images': len([obj for obj in page.objects if obj.get('object_type') == 'image'])
                            }
                            raw_text_by_page[page_num] += f"\n[PAGE INFO: {page_info}]"
                        except Exception as e:
                            logger.debug(f"Could not get page info: {e}")
                            
        except Exception as e:
            logger.error(f"Error extracting raw text: {e}")
        
        return raw_text_by_page
    
    def test_multiple_pdfs(self, pdf_dir: str, max_files: int = 5) -> Dict[str, Any]:
        """Test processing of multiple PDF files"""
        
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF directory not found: {pdf_dir}")
            return {}
        
        # Find PDF files
        pdf_files = []
        for file in os.listdir(pdf_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(pdf_dir, file))
        
        if not pdf_files:
            logger.error(f"No PDF files found in: {pdf_dir}")
            return {}
        
        # Limit to max_files
        pdf_files = pdf_files[:max_files]
        logger.info(f"Testing {len(pdf_files)} PDF files")
        
        all_results = {}
        total_chunks = 0
        failed_files = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            logger.info(f"Processing file {i}/{len(pdf_files)}: {filename}")
            
            try:
                stats = self.test_single_pdf(pdf_path)
                all_results[filename] = stats
                
                # Check if the stats indicate an error
                if 'error' in stats:
                    failed_files.append({
                        'filename': filename,
                        'error': stats['error']
                    })
                else:
                    total_chunks += stats.get('total_chunks', 0)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing {pdf_path}: {error_msg}")
                all_results[filename] = {'error': error_msg}
                failed_files.append({
                    'filename': filename,
                    'error': error_msg
                })
        
        # Generate overall summary
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
    
    def _generate_stats(self, chunks: List[DocumentChunk], pdf_path: str) -> Dict[str, Any]:
        """Generate statistics for processed chunks"""
        
        if not chunks:
            return {
                'pdf_path': os.path.basename(pdf_path),
                'total_chunks': 0,
                'error': 'No chunks extracted'
            }
        
        # Content type distribution
        content_types = {}
        for chunk in chunks:
            content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
        
        # Page distribution
        pages = set(chunk.page_number for chunk in chunks if chunk.page_number)
        
        # Chunk size statistics
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        stats = {
            'pdf_path': os.path.basename(pdf_path),
            'pdf_name': os.path.basename(pdf_path),
            'total_chunks': len(chunks),
            'total_pages_with_content': len(pages),
            'content_type_distribution': content_types,
            'chunk_size_stats': {
                'min': min(chunk_sizes) if chunk_sizes else 0,
                'max': max(chunk_sizes) if chunk_sizes else 0,
                'avg': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            },
            'sample_chunks': [
                {
                    'chunk_index': chunk.chunk_index,
                    'content_type': chunk.content_type,
                    'page_number': chunk.page_number,
                    'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                }
                for chunk in chunks[:3]  # First 3 chunks as samples
            ]
        }
        
        return stats
    
    def _save_results(self, chunks: List[DocumentChunk], stats: Dict[str, Any], pdf_path: str):
        """Save processing results to files"""
        
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Save chunks as JSON
        chunks_file = os.path.join(self.output_dir, f"{base_name}_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2, default=str)
        
        # Save stats
        stats_file = os.path.join(self.output_dir, f"{base_name}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Results saved: {chunks_file}, {stats_file}")
    
    def save_raw_text(self, raw_text_by_page: Dict[int, str], pdf_path: str):
        """Save raw extracted text for debugging"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        raw_text_file = os.path.join(self.output_dir, f"{base_name}_raw_text.txt")
        
        with open(raw_text_file, 'w', encoding='utf-8') as f:
            for page_num, text in raw_text_by_page.items():
                f.write(f"=== PAGE {page_num} ===\n")
                f.write(text)
                f.write(f"\n\n=== END PAGE {page_num} ===\n\n")
        
        logger.info(f"Raw text saved: {raw_text_file}")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save overall summary"""
        
        summary_file = os.path.join(self.output_dir, f"extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Summary saved: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Files processed successfully: {summary['total_files_processed']}")
        print(f"Files failed: {summary['total_files_failed']}")
        print(f"Total chunks extracted: {summary['total_chunks_extracted']}")
        print(f"Results saved to: {self.output_dir}")
        
        # Display failed files and errors clearly
        if summary['total_files_failed'] > 0:
            print("\n" + "="*50)
            print("FAILED FILES SUMMARY")
            print("="*50)
            failed_files = summary.get('failed_files', [])
            for i, failed_file in enumerate(failed_files, 1):
                print(f"{i}. {failed_file['filename']}")
                print(f"   Error: {failed_file['error']}")
                print()
            
            # Also save failed files to separate file for easy reference
            failed_files_file = os.path.join(self.output_dir, f"failed_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(failed_files_file, 'w', encoding='utf-8') as f:
                f.write("FAILED FILES REPORT\n")
                f.write("=" * 50 + "\n\n")
                for i, failed_file in enumerate(failed_files, 1):
                    f.write(f"{i}. {failed_file['filename']}\n")
                    f.write(f"   Error: {failed_file['error']}\n\n")
            
            print(f"Failed files report also saved to: {failed_files_file}")

def main():
    parser = argparse.ArgumentParser(description='PDF Document Extraction Pipeline Tester')
    parser.add_argument('--pdf-file', type=str,
                       help='Path to a single PDF file to process')
    parser.add_argument('--pdf-dir', type=str,
                       help='Directory containing PDF files to process')
    parser.add_argument('--max-files', type=int, default=5,
                       help='Maximum number of PDF files to process (default: 5)')
    parser.add_argument('--output-dir', type=str, default='test_output',
                       help='Directory to save output files (default: test_output)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging for detailed output')
    parser.add_argument('--ocr-backend', type=str, default='auto',
                       choices=['auto', 'tesseract', 'dolphin', 'none'],
                       help='OCR backend to use (default: auto - prefers dolphin if available)')
    parser.add_argument('--disable-ocr', action='store_true',
                       help='Completely disable OCR, even when text extraction fails')
    parser.add_argument('--disable-smart-ocr', action='store_true',
                       help='Disable smart OCR detection (use OCR for all PDFs as fallback)')
    
    args = parser.parse_args()
    
    # Adjust logging level if debug is requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    if not args.pdf_file and not args.pdf_dir:
        print("Error: Please specify either --pdf-file or --pdf-dir")
        parser.print_help()
        sys.exit(1)
    
    tester = PDFExtractionTester(
        args.output_dir, 
        ocr_backend=args.ocr_backend, 
        disable_ocr=args.disable_ocr,
        smart_ocr=not args.disable_smart_ocr
    )
    
    try:
        if args.pdf_file:
            logger.info("Testing single PDF file...")
            stats = tester.test_single_pdf(args.pdf_file)
            print("\nProcessing completed!")
            print(f"Extracted {stats.get('total_chunks', 0)} chunks")
            
        elif args.pdf_dir:
            logger.info("Testing multiple PDF files...")
            summary = tester.test_multiple_pdfs(args.pdf_dir, args.max_files)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 