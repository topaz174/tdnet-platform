#!/usr/bin/env python3
"""
Debug script to test text cleaning and identify overly aggressive patterns.
"""

import sys
sys.path.append('.')

# Sample text from the extracted chunks to test cleaning behavior
sample_text = """(3)発行済株式数(普通株式) 1 期末発行済株式数(自己株式を含む) 2023年3月期 21,611,000株 2022年3月期 21,611,000株 2 期末自己株式数 2023年3月期 6,067,959株 2022年3月期 6,067,959株 3 期中平均株式数 2023年3月期 15,543,041株 2022年3月期 16,341,383株(参考)個別業績の概要 1.2023年3月期の個別業績(2022年4月1日~2023年3月31日)(1)個別経営成績(%表示は対前期増減率) 売上高 営業利益 経常利益 当期純利益 百万円 % 百万円 % 百万円 % 百万円 % 2023年3月期 645 0.0 132 -19.9 225 -12.7 156 -12.2 2022年3月期 645 -0.9 165 -2.6 258 3.5 178 2.3 1株当たり 潜在株式調整後 当期純利益 1株当たり当期純利益 円 銭 円 銭 2023年3月期 10.08 - 2022年3月期 10.91 -(2)個別財政状態 総資産 純資産 自己資本比率 1株当たり純資産 百万円 百万円 % 円 銭 2023年3月期 20,538 1,061 5.2 68.29 2022年3月期 18,511 982 5.3 63.21(参考)自己資本 2023年3月期 1,061百万円 2022年3月期 982百万円"""

# Test the cleaning function
from src.pdf_extraction_pipeline import normalize_text
import re
import unicodedata

# Full-width map from the pipeline
FULL_WIDTH_MAP = str.maketrans({
    "△": "-",    # triangle minus (negative values)
    "▲": "-",    # black triangle (negative values)
    "％": "%",    # full-width percent
    "－": "-",    # full-width hyphen-minus (U+FF0D) - main culprit
    "‐": "-",    # hyphen (U+2010)
    "‑": "-",    # non-breaking hyphen (U+2011)
    "–": "-",    # en dash (U+2013)
    "—": "-",    # em dash (U+2014)
    "―": "-",    # horizontal bar (U+2015)
    "（": "(",   # full-width parentheses
    "）": ")",
    "［": "[",   # full-width brackets
    "］": "]",
    "｛": "{",   # full-width braces
    "｝": "}",
    # Common Japanese symbols in financial documents
    "円": "円",   # keep yen symbol as-is
    "株": "株",   # keep kanji for "stock/share" as-is
    "：": ":",   # full-width colon
    "；": ";",   # full-width semicolon
    "？": "?",   # full-width question mark
    "！": "!",   # full-width exclamation mark
    "。": ".",   # Japanese period → period
    "、": ",",   # Japanese comma → comma
})

def debug_clean_text(text: str) -> str:
    """Debug version of _clean_text to see what's being removed"""
    print(f"=== DEBUGGING TEXT CLEANING ===")
    print(f"Original text length: {len(text)}")
    print(f"Original text preview: {text[:200]}...")
    
    # 1) Apply comprehensive normalization
    text = normalize_text(text)
    print(f"After normalize_text: {len(text)} chars")
    
    # 2) Remove page headers/footers - these patterns look suspicious
    original_len = len(text)
    text = re.sub(r'^\s*[-–—]+?\s*\d+\s*[-–—]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    if len(text) != original_len:
        print(f"After page headers removal: {original_len} -> {len(text)} chars")
    
    # Remove embedded page footers in middle of text
    original_len = len(text)
    text = re.sub(r'[\s\.\。]\s*[-–—]\s*\d+\s*[-–—]\s+', ' ', text)
    text = re.sub(r'\s+[-–—]\s*\d{1,2}\s*[-–—]\s+', ' ', text)
    text = re.sub(r'\s+[-–—]\s*\d{1,2}\s*[-–—]\s+(?=nms|ホールディングス|株式会社)', ' ', text)
    text = re.sub(r'[\.\。]\s+[-–—]\s*\d{1,2}\s*[-–—]\s+', '. ', text)
    if len(text) != original_len:
        print(f"After embedded footers removal: {original_len} -> {len(text)} chars")

    # 5) Remove document headers and company information patterns - VERY SUSPICIOUS
    original_len = len(text)
    text = re.sub(r'\s*nms\s*ホールディングス株式会社\([0-9]+\)[0-9]+年[0-9]+月期\s*決算短信[^\n]*', '', text)
    text = re.sub(r'\s*nms\s*ホールディングス株式会社[^\n]*決算短信[^\n]*', '', text)
    text = re.sub(r'\([0-9]{4}\)[0-9]+年[0-9]+月期\s*決算短信', '', text)
    text = re.sub(r'\s+決算短信\s+', ' ', text)
    text = re.sub(r'\s+nms\s+(?=ホールディングス|株式会社)', ' ', text)
    text = re.sub(r'\s+ホールディングス株式会社\s+', ' ', text)
    if len(text) != original_len:
        print(f"After company headers removal: {original_len} -> {len(text)} chars - THIS IS SUSPICIOUS!")
    
    # Enhanced header removal patterns - ALSO SUSPICIOUS
    header_patterns = [
        r'^\s*[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日\s*$',
        r'^\s*各\s*位\s*$',
        r'^\s*会\s*社\s*名\s+[^\n]*株\s*式\s*会\s*社\s*$',
        r'^\s*代表者名\s+[^\n]*$',
        r'^\s*問合せ先\s+[^\n]*TEL[^\n]*$',
        r'^\s*\(コード[:\d\s]*\)[^\n]*$',
        r'^\s*\(TEL[^\n]*\)\s*$',
        r'^\s*譲渡制限付株式報酬としての新株式発行に関するお知らせ\s*$',
        r'^\s*業績予想修正に関するお知らせ\s*$',
        r'^\s*決算短信\s*$',
        r'^\s*各\s+位\s*$',
        r'^\s*会\s+社\s+名\s*$',
        r'^\s*代表者名\s*$',
        r'^\s*問合せ先\s*$',
    ]
    
    original_len = len(text)
    for pattern in header_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    if len(text) != original_len:
        print(f"After header patterns removal: {original_len} -> {len(text)} chars")

    # 3) Remove redundant dot sequences
    original_len = len(text)
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'^\s*[.\s]{10,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+[.\s]{6,}\s+', ' ', text)
    if len(text) != original_len:
        print(f"After dot sequences removal: {original_len} -> {len(text)} chars")

    # 6) Fix dangling parentheses and broken formatting
    original_len = len(text)
    text = re.sub(r'\)\s*の\s*本割当株式', ')の本割当株式', text)
    text = re.sub(r'\)\s*、\s*', ')、', text)
    text = re.sub(r'\s+\(\s*', '(', text)
    text = re.sub(r'\s+\)\s*', ') ', text)
    text = re.sub(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*円', r'\1,\2,\3円', text)
    text = re.sub(r'(\d+)\s+円', r'\1円', text)
    text = re.sub(r'(\d+)\s+株', r'\1株', text)
    text = re.sub(r'(\d+)\s+名', r'\1名', text)
    text = re.sub(r'各\s+位\s+各\s+位', '各位', text)
    text = re.sub(r'株\s*式\s*会\s*社\s+株\s*式\s*会\s*社', '株式会社', text)
    if len(text) != original_len:
        print(f"After formatting fixes: {original_len} -> {len(text)} chars")

    # 4) Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    print(f"Final text length: {len(text)}")
    print(f"Final text preview: {text[:200]}...")
    
    return text

if __name__ == "__main__":
    result = debug_clean_text(sample_text)
    print(f"\nFINAL RESULT LENGTH: {len(result)}")
    if len(result) < 100:
        print("❌ CRITICAL: Text was over-cleaned!")
        print(f"Remaining text: '{result}'")
    else:
        print("✅ Text cleaning preserved content")