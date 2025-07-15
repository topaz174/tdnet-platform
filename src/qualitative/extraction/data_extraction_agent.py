#!/usr/bin/env python3
"""
Financial Data Extraction Agent
=====================================

This agent extracts specific numeric financial data from PDFs identified by the retrieval system.
It uses LLM-powered parsing to extract structured financial information like:
- Earnings revisions (revenue, operating profit, net income)
- Dividend changes (amount, yield, payout ratio)
- Guidance updates (forward-looking projections)
- M&A values (deal size, premiums)

Architecture:
1. Retrieval Agent identifies relevant documents
2. Extraction Agent processes each PDF to extract numeric data
3. Structured results are returned with confidence scores
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import pdfplumber
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# OCR dependencies
import easyocr
import fitz  # PyMuPDF for better PDF to image conversion
import numpy as np
from PIL import Image
import io
import torch

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Types of financial data we can extract"""
    EARNINGS_REVISION = "earnings_revision"
    DIVIDEND_CHANGE = "dividend_change"
    GUIDANCE_UPDATE = "guidance_update"
    SHARE_BUYBACK = "share_buyback"
    M_A_TRANSACTION = "ma_transaction"
    SPLIT_MERGER = "split_merger"

@dataclass
class FinancialMetric:
    """Structured financial metric with before/after values"""
    metric_name: str  # e.g., "売上高", "営業利益", "配当金"
    previous_value: Optional[float] = None
    revised_value: Optional[float] = None
    change_amount: Optional[float] = None
    change_percentage: Optional[float] = None
    unit: Optional[str] = None  # e.g., "百万円", "円", "%"
    period: Optional[str] = None  # e.g., "2025年3月期", "第2四半期"
    confidence: float = 0.0  # 0-1 confidence score

@dataclass
class ExtractedData:
    """Complete extracted financial data from a document"""
    document_id: int
    company_code: str
    company_name: str
    document_title: str
    document_date: date
    data_type: DataType
    metrics: List[FinancialMetric]
    summary: str
    extraction_confidence: float
    raw_text_sample: str  # Sample of source text for verification

class FinancialDataExtractor:
    """LLM-powered financial data extraction from PDFs with OCR support"""
    
    def __init__(self, enable_ocr=True, use_gpu=True):
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Use latest model for best accuracy
            temperature=0.1,  # Low temperature for factual extraction
            max_tokens=2000
        )
        
        # OCR configuration
        self.enable_ocr = enable_ocr
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.ocr_reader = None
        
        if self.enable_ocr:
            self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader with optimal settings"""
        try:
            logger.info(f"Initializing EasyOCR with GPU: {self.use_gpu}")
            self.ocr_reader = easyocr.Reader(
                ['ja', 'en'],  # Japanese and English support
                gpu=self.use_gpu,
                verbose=False  # Reduce output noise
            )
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            self.enable_ocr = False
            self.ocr_reader = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """Extract text from PDF with OCR fallback for image-based PDFs"""
        if not os.path.exists(pdf_path):
            return "", False
        
        text = ""
        is_text_based = False
        
        # Step 1: Try pdfplumber first (fastest for text-based PDFs)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        is_text_based = True
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row and any(cell for cell in row if cell):
                                text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        
        # Step 2: Fallback to pypdf
        if not text.strip():
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        is_text_based = True
            except Exception as e:
                logger.warning(f"pypdf failed for {pdf_path}: {e}")
        
        # Step 3: OCR fallback for image-based PDFs
        if not text.strip() and self.enable_ocr and self.ocr_reader is not None:
            logger.info(f"No text found with standard methods, trying OCR for {pdf_path}")
            ocr_text = self._extract_text_with_ocr(pdf_path)
            if ocr_text:
                text = ocr_text
                is_text_based = False  # Mark as OCR-extracted
                logger.info(f"✅ OCR extracted {len(text)} characters from {pdf_path}")
        
        return text.strip(), is_text_based
    
    def _extract_text_with_ocr(self, pdf_path: str, max_pages: int = 3) -> str:
        """Extract text using OCR with GPU acceleration"""
        if not self.ocr_reader:
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            extracted_text = ""
            
            # Process up to max_pages for balance of speed vs coverage
            pages_to_process = min(len(doc), max_pages)
            logger.info(f"Processing {pages_to_process} pages with OCR...")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Convert PDF page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR accuracy
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Convert PIL Image to numpy array for EasyOCR
                img_array = np.array(image)
                
                # Run OCR
                results = self.ocr_reader.readtext(img_array)
                
                # Extract text with confidence filtering
                page_text = ""
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # Filter low-confidence results
                        page_text += text + " "
                
                if page_text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text.strip()}\n"
            
            doc.close()
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return ""
    
    def classify_document_type(self, title: str, text: str) -> DataType:
        """Classify the type of financial announcement"""
        title_lower = title.lower()
        text_sample = text[:1000].lower()
        
        # Pattern matching for document types
        if any(term in title_lower for term in ["業績予想", "業績見通し", "予想修正", "見通し修正"]):
            return DataType.EARNINGS_REVISION
        elif any(term in title_lower for term in ["配当", "増配", "減配", "復配", "無配"]):
            return DataType.DIVIDEND_CHANGE
        elif any(term in title_lower for term in ["自己株式", "株式買戻し", "自社株買い"]):
            return DataType.SHARE_BUYBACK
        elif any(term in title_lower for term in ["買収", "合併", "m&a", "統合", "子会社化"]):
            return DataType.M_A_TRANSACTION
        elif any(term in title_lower for term in ["株式分割", "株式併合", "分割"]):
            return DataType.SPLIT_MERGER
        else:
            return DataType.GUIDANCE_UPDATE
    
    def extract_financial_data(self, text: str, title: str, data_type: DataType) -> Tuple[List[FinancialMetric], str, float]:
        """Use LLM to extract structured financial data from text"""
        
        # Create specialized prompts based on data type
        if data_type == DataType.EARNINGS_REVISION:
            extraction_prompt = self._create_earnings_extraction_prompt(text, title)
        elif data_type == DataType.DIVIDEND_CHANGE:
            extraction_prompt = self._create_dividend_extraction_prompt(text, title)
        else:
            extraction_prompt = self._create_general_extraction_prompt(text, title, data_type)
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a financial data extraction expert specializing in Japanese corporate disclosures. Extract precise numeric data with high accuracy."),
                HumanMessage(content=extraction_prompt)
            ])
            
            # Parse the LLM response
            return self._parse_llm_response(response.content, data_type)
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return [], f"Error during extraction: {e}", 0.0
    
    def _create_earnings_extraction_prompt(self, text: str, title: str) -> str:
        """Create prompt for earnings revision extraction"""
        return f"""
Analyze this Japanese corporate earnings revision announcement and extract all numeric financial data.

Title: {title}

Document text:
{text[:3000]}

Please extract the following information in JSON format:

{{
    "metrics": [
        {{
            "metric_name": "売上高" | "営業利益" | "経常利益" | "純利益" | etc,
            "previous_value": <number or null>,
            "revised_value": <number or null>,
            "change_amount": <number or null>,
            "change_percentage": <number or null>,
            "unit": "百万円" | "億円" | "円" | "%" | etc,
            "period": "2025年3月期" | "第2四半期" | etc,
            "confidence": <0-1 score>
        }}
    ],
    "summary": "<brief summary of the revision>",
    "overall_confidence": <0-1 score>
}}

Focus on:
- Revenue (売上高)
- Operating profit (営業利益) 
- Ordinary profit (経常利益)
- Net income (純利益)
- Previous vs revised forecasts
- Percentage changes
- Specific time periods

Extract only data that is clearly stated with high confidence. Use null for missing values.
"""

    def _create_dividend_extraction_prompt(self, text: str, title: str) -> str:
        """Create prompt for dividend change extraction"""
        return f"""
Analyze this Japanese corporate dividend announcement and extract all numeric data.

Title: {title}

Document text:
{text[:3000]}

Please extract the following information in JSON format:

{{
    "metrics": [
        {{
            "metric_name": "配当金" | "中間配当" | "期末配当" | "配当性向" | etc,
            "previous_value": <number or null>,
            "revised_value": <number or null>,
            "change_amount": <number or null>,
            "change_percentage": <number or null>,
            "unit": "円" | "%" | etc,
            "period": "2025年3月期" | "中間" | "期末" | etc,
            "confidence": <0-1 score>
        }}
    ],
    "summary": "<brief summary of the dividend change>",
    "overall_confidence": <0-1 score>
}}

Focus on:
- Dividend per share (1株当たり配当金)
- Interim dividend (中間配当)
- Year-end dividend (期末配当)
- Dividend yield (配当利回り)
- Payout ratio (配当性向)
- Previous vs new amounts

Extract only clearly stated numeric values.
"""

    def _create_general_extraction_prompt(self, text: str, title: str, data_type: DataType) -> str:
        """Create general extraction prompt for other data types"""
        return f"""
Analyze this Japanese corporate announcement and extract relevant numeric financial data.

Title: {title}
Data Type: {data_type.value}

Document text:
{text[:3000]}

Please extract financial metrics in JSON format:

{{
    "metrics": [
        {{
            "metric_name": "<metric name>",
            "previous_value": <number or null>,
            "revised_value": <number or null>,
            "change_amount": <number or null>,
            "change_percentage": <number or null>,
            "unit": "<unit>",
            "period": "<time period>",
            "confidence": <0-1 score>
        }}
    ],
    "summary": "<brief summary>",
    "overall_confidence": <0-1 score>
}}

Extract any numeric financial data that appears relevant to the announcement type.
"""

    def _parse_llm_response(self, response: str, data_type: DataType) -> Tuple[List[FinancialMetric], str, float]:
        """Parse LLM JSON response into structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return [], "No JSON found in response", 0.0
            
            data = json.loads(json_match.group())
            
            metrics = []
            for metric_data in data.get("metrics", []):
                metric = FinancialMetric(
                    metric_name=metric_data.get("metric_name", ""),
                    previous_value=metric_data.get("previous_value"),
                    revised_value=metric_data.get("revised_value"),
                    change_amount=metric_data.get("change_amount"),
                    change_percentage=metric_data.get("change_percentage"),
                    unit=metric_data.get("unit"),
                    period=metric_data.get("period"),
                    confidence=metric_data.get("confidence", 0.5)
                )
                metrics.append(metric)
            
            summary = data.get("summary", "")
            confidence = data.get("overall_confidence", 0.5)
            
            return metrics, summary, confidence
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return [], f"JSON parsing error: {e}", 0.0
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return [], f"Response parsing error: {e}", 0.0

    def process_document(self, pdf_path: str, document_id: int, company_code: str, 
                        company_name: str, title: str, doc_date: date) -> Optional[ExtractedData]:
        """Process a single PDF document and extract financial data"""
        logger.info(f"Processing document {document_id}: {title}")
        
        # Extract text from PDF
        text, is_text_based = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path} even with OCR")
            return None
        
        if not is_text_based:
            logger.info(f"Text extracted via OCR from image-based PDF: {pdf_path}")
        else:
            logger.info(f"Text extracted directly from text-based PDF: {pdf_path}")
        
        # Classify document type
        data_type = self.classify_document_type(title, text)
        
        # Extract financial data
        metrics, summary, confidence = self.extract_financial_data(text, title, data_type)
        
        # Create result
        result = ExtractedData(
            document_id=document_id,
            company_code=company_code,
            company_name=company_name,
            document_title=title,
            document_date=doc_date,
            data_type=data_type,
            metrics=metrics,
            summary=summary,
            extraction_confidence=confidence,
            raw_text_sample=text[:500]
        )
        
        return result

def main():
    """Test the extraction system with the sample PDF"""
    extractor = FinancialDataExtractor()
    
    # Test with the provided PDF
    sample_pdf = "15-30_24850_業績予想修正に関するお知らせ.pdf"
    
    if os.path.exists(sample_pdf):
        result = extractor.process_document(
            pdf_path=sample_pdf,
            document_id=24850,
            company_code="24850",
            company_name="Sample Company",
            title="業績予想修正に関するお知らせ",
            doc_date=date.today()
        )
        
        if result:
            print("=== Extraction Result ===")
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
        else:
            print("No data extracted")
    else:
        print(f"Sample PDF {sample_pdf} not found")

if __name__ == "__main__":
    main()