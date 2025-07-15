#!/usr/bin/env python3
"""
Hedge Fund Level Financial Intelligence Agent
===========================================

This agent provides sophisticated financial analysis capabilities equivalent to
what professional hedge fund analysts use. It combines:

1. Enhanced document retrieval with semantic search
2. OCR-enabled financial data extraction 
3. Advanced financial analytics with 25+ metrics
4. Professional investment thesis generation
5. Risk assessment and opportunity identification

Capabilities:
- Answer any financial question a hedge fund analyst might ask
- Provide comprehensive company analysis with detailed metrics
- Generate professional investment theses with supporting data
- Perform comparative analysis across companies and sectors
- Extract and analyze data from image-based PDFs using OCR
- Calculate sophisticated ratios used by institutional investors
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, date
import json
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

# Import our systems
from financial_data_extraction_agent import (
    FinancialDataExtractor, 
    ExtractedData, 
    FinancialMetric,
    DataType
)
from advanced_financial_analytics import (
    AdvancedFinancialAnalytics,
    FinancialData,
    MetricResult,
    AnalysisResult,
    MetricCategory
)

# LLM for intelligent responses
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveAnalysis:
    """Complete hedge fund level analysis result"""
    company_code: str
    company_name: str
    query: str
    
    # Document analysis
    documents_analyzed: int
    extraction_confidence: float
    
    # Financial metrics
    financial_metrics: List[MetricResult]
    overall_score: float
    investment_thesis: str
    
    # Risk and opportunities
    key_risks: List[str]
    key_opportunities: List[str]
    
    # Professional response
    executive_summary: str
    detailed_analysis: str
    recommendation: str
    
    # Supporting data
    extracted_financial_data: List[ExtractedData]
    calculation_details: Dict[str, Any]

class HedgeFundFinancialAgent:
    """Professional-grade financial intelligence agent"""
    
    def __init__(self):
        """Initialize all components of the hedge fund agent"""
        logger.info("🏦 Initializing Hedge Fund Financial Intelligence Agent...")
        
        # Initialize data extraction with OCR support
        self.extractor = FinancialDataExtractor(enable_ocr=True, use_gpu=True)
        logger.info("✅ Financial data extractor initialized with OCR support")
        
        # Initialize advanced analytics engine
        self.analytics = AdvancedFinancialAnalytics()
        logger.info("✅ Advanced financial analytics engine initialized")
        
        # Initialize LLM for intelligent responses
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # Low temperature for factual analysis
            max_tokens=3000
        )
        logger.info("✅ GPT-4 LLM initialized for response generation")
        
        # Mock database connection (in real implementation, connect to PostgreSQL)
        self.mock_database = self._initialize_mock_database()
        logger.info("✅ Database interface initialized")
        
        logger.info("🚀 Hedge Fund Financial Agent ready for analysis!")
    
    def _initialize_mock_database(self) -> Dict[str, Any]:
        """Initialize mock database with sample company data"""
        return {
            "2485": {
                "company_name": "株式会社ティア",
                "sector": "technology",
                "pdf_path": "15-30_24850_業績予想修正に関するお知らせ.pdf",
                "market_data": {
                    "market_cap": 20000_000_000,  # 20B JPY
                    "enterprise_value": 22000_000_000,
                    "shares_outstanding": 100_000_000,
                    "stock_price": 200.0
                }
            }
        }
    
    def analyze_company(self, company_code: str, query: str = None) -> ComprehensiveAnalysis:
        """
        Perform comprehensive hedge fund level analysis of a company
        
        Args:
            company_code: Company identifier (e.g., "2485")
            query: Specific analytical question (optional)
            
        Returns:
            ComprehensiveAnalysis with full professional analysis
        """
        logger.info(f"🔍 Starting comprehensive analysis for company {company_code}")
        
        # Get company information
        company_info = self.mock_database.get(company_code)
        if not company_info:
            raise ValueError(f"Company {company_code} not found in database")
        
        company_name = company_info["company_name"]
        logger.info(f"📊 Analyzing: {company_name} ({company_code})")
        
        # Step 1: Extract financial data from documents
        extracted_data = self._extract_financial_data(company_code, company_info)
        
        # Step 2: Convert to analytics format and calculate metrics
        financial_data = self._convert_to_financial_data(extracted_data, company_info)
        analysis_result = self.analytics.analyze_company(financial_data, company_code, company_name)
        
        # Step 3: Generate professional response
        professional_response = self._generate_professional_response(
            analysis_result, extracted_data, query or "Provide comprehensive financial analysis"
        )
        
        # Step 4: Compile comprehensive analysis
        comprehensive_analysis = ComprehensiveAnalysis(
            company_code=company_code,
            company_name=company_name,
            query=query or "Comprehensive Analysis",
            documents_analyzed=len(extracted_data),
            extraction_confidence=extracted_data[0].extraction_confidence if extracted_data else 0.0,
            financial_metrics=analysis_result.metrics,
            overall_score=analysis_result.overall_score,
            investment_thesis=analysis_result.investment_thesis,
            key_risks=analysis_result.key_risks,
            key_opportunities=analysis_result.key_opportunities,
            executive_summary=professional_response["executive_summary"],
            detailed_analysis=professional_response["detailed_analysis"],
            recommendation=professional_response["recommendation"],
            extracted_financial_data=extracted_data,
            calculation_details={"analytics_version": "v1.0", "analysis_date": datetime.now().isoformat()}
        )
        
        logger.info(f"✅ Analysis complete for {company_name}")
        return comprehensive_analysis
    
    def _extract_financial_data(self, company_code: str, company_info: Dict) -> List[ExtractedData]:
        """Extract financial data from company documents using OCR"""
        logger.info(f"📄 Extracting financial data for {company_code}")
        
        pdf_path = company_info.get("pdf_path")
        if not pdf_path:
            logger.warning(f"No PDF available for {company_code}")
            return []
        
        # Extract data from the PDF
        result = self.extractor.process_document(
            pdf_path=pdf_path,
            document_id=int(company_code),
            company_code=company_code,
            company_name=company_info["company_name"],
            title="業績予想修正に関するお知らせ",
            doc_date=date.today()
        )
        
        if result:
            logger.info(f"✅ Extracted {len(result.metrics)} financial metrics")
            return [result]
        else:
            logger.warning(f"No financial data extracted for {company_code}")
            return []
    
    def _convert_to_financial_data(self, extracted_data: List[ExtractedData], company_info: Dict) -> FinancialData:
        """Convert extracted data to analytics format"""
        if not extracted_data:
            return FinancialData()
        
        # Get the most recent data
        latest_data = extracted_data[0]
        market_data = company_info.get("market_data", {})
        
        # Convert extracted metrics to financial data structure
        financial_data = FinancialData(
            market_cap=market_data.get("market_cap"),
            enterprise_value=market_data.get("enterprise_value"),
            shares_outstanding=market_data.get("shares_outstanding"),
            stock_price=market_data.get("stock_price"),
            period=latest_data.document_date.strftime("%Y年%m月期") if latest_data.document_date else None
        )
        
        # Extract values from metrics
        for metric in latest_data.metrics:
            metric_name = metric.metric_name.lower()
            
            # Map Japanese metric names to our data structure
            if "売上" in metric_name or "revenue" in metric_name:
                financial_data.revenue = metric.revised_value or metric.previous_value
            elif "営業利益" in metric_name or "operating" in metric_name:
                financial_data.operating_income = metric.revised_value or metric.previous_value
            elif "純利益" in metric_name or "net" in metric_name:
                financial_data.net_income = metric.revised_value or metric.previous_value
        
        # Estimate missing values based on available data
        if financial_data.revenue and financial_data.operating_income:
            financial_data.operating_margin = financial_data.operating_income / financial_data.revenue
        
        if financial_data.revenue and financial_data.net_income:
            financial_data.net_margin = financial_data.net_income / financial_data.revenue
        
        # Estimate balance sheet items (would come from actual data in production)
        if financial_data.revenue:
            # Conservative estimates based on typical Japanese company ratios
            financial_data.total_assets = financial_data.revenue * 1.5
            financial_data.shareholders_equity = financial_data.total_assets * 0.6
            financial_data.total_debt = financial_data.total_assets * 0.3
            financial_data.current_assets = financial_data.total_assets * 0.4
            financial_data.current_liabilities = financial_data.total_assets * 0.2
        
        return financial_data
    
    def _generate_professional_response(self, analysis: AnalysisResult, extracted_data: List[ExtractedData], query: str) -> Dict[str, str]:
        """Generate professional hedge fund level response"""
        logger.info("🎯 Generating professional investment analysis response")
        
        # Prepare context for LLM
        metrics_summary = self._format_metrics_for_llm(analysis.metrics)
        extracted_summary = self._format_extracted_data_for_llm(extracted_data)
        
        prompt = f"""
You are a senior equity research analyst at a top-tier hedge fund. Provide a professional investment analysis for {analysis.company_name} ({analysis.company_code}).

QUERY: {query}

FINANCIAL METRICS ANALYSIS:
{metrics_summary}

EXTRACTED FINANCIAL DATA:
{extracted_summary}

OVERALL INVESTMENT SCORE: {analysis.overall_score:.1f}/100

Provide a comprehensive response with:

1. EXECUTIVE SUMMARY (2-3 sentences max)
- Key investment highlights
- Overall recommendation direction

2. DETAILED ANALYSIS (comprehensive professional analysis)
- Financial performance assessment
- Key metric interpretations
- Competitive positioning insights
- Risk factors and mitigants

3. INVESTMENT RECOMMENDATION (clear recommendation with rationale)
- BUY/HOLD/SELL recommendation
- Target price rationale (if applicable)
- Key catalysts and risks
- Time horizon considerations

Use professional hedge fund language and focus on actionable investment insights.
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a senior equity research analyst at a prestigious hedge fund. Provide institutional-quality investment analysis."),
                HumanMessage(content=prompt)
            ])
            
            # Parse the response into sections
            content = response.content
            sections = self._parse_llm_response(content)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generating professional response: {e}")
            return {
                "executive_summary": f"Analysis completed for {analysis.company_name} with overall score {analysis.overall_score:.1f}/100",
                "detailed_analysis": "Detailed analysis temporarily unavailable",
                "recommendation": "Further analysis required"
            }
    
    def _format_metrics_for_llm(self, metrics: List[MetricResult]) -> str:
        """Format metrics for LLM consumption"""
        formatted = []
        
        # Group by category
        categories = {}
        for metric in metrics:
            cat = metric.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metric)
        
        for category, cat_metrics in categories.items():
            formatted.append(f"\n{category.upper()}:")
            for metric in cat_metrics:
                value_str = f"{metric.value:.3f}" if metric.value else "N/A"
                formatted.append(f"  {metric.name}: {value_str} - {metric.interpretation}")
        
        return "\n".join(formatted)
    
    def _format_extracted_data_for_llm(self, extracted_data: List[ExtractedData]) -> str:
        """Format extracted data for LLM consumption"""
        if not extracted_data:
            return "No financial data extracted"
        
        formatted = []
        for data in extracted_data:
            formatted.append(f"\nDocument: {data.document_title}")
            formatted.append(f"Date: {data.document_date}")
            formatted.append(f"Confidence: {data.extraction_confidence:.1%}")
            formatted.append("Metrics:")
            
            for metric in data.metrics[:8]:  # Top 8 metrics
                prev = f"{metric.previous_value:,.0f}" if metric.previous_value else "N/A"
                rev = f"{metric.revised_value:,.0f}" if metric.revised_value else "N/A"
                change = f"{metric.change_percentage:+.1f}%" if metric.change_percentage else "N/A"
                formatted.append(f"  {metric.metric_name}: {prev} → {rev} ({change}) {metric.unit}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, content: str) -> Dict[str, str]:
        """Parse LLM response into structured sections"""
        sections = {
            "executive_summary": "",
            "detailed_analysis": "",
            "recommendation": ""
        }
        
        # Simple parsing logic (could be enhanced with regex)
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "EXECUTIVE SUMMARY" in line.upper():
                current_section = "executive_summary"
            elif "DETAILED ANALYSIS" in line.upper():
                current_section = "detailed_analysis"
            elif "RECOMMENDATION" in line.upper() or "INVEST" in line.upper():
                current_section = "recommendation"
            elif current_section and not line.startswith(("1.", "2.", "3.", "#", "**")):
                sections[current_section] += line + " "
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def answer_analytical_question(self, query: str) -> str:
        """
        Answer sophisticated financial questions like a hedge fund analyst
        
        Examples:
        - "What is the investment thesis for company 2485?"
        - "Analyze the profitability trends for company 2485"
        - "What are the key risks for investing in company 2485?"
        """
        logger.info(f"❓ Answering analytical question: {query}")
        
        # Extract company code from query (simple implementation)
        company_code = self._extract_company_code(query)
        
        if company_code:
            # Perform full analysis
            analysis = self.analyze_company(company_code, query)
            
            # Return appropriate section based on query type
            if "thesis" in query.lower() or "investment" in query.lower():
                return f"**Investment Thesis for {analysis.company_name}:**\n\n{analysis.investment_thesis}\n\n**Executive Summary:**\n{analysis.executive_summary}\n\n**Overall Score:** {analysis.overall_score:.1f}/100"
            elif "risk" in query.lower():
                risks_text = "\n".join([f"• {risk}" for risk in analysis.key_risks])
                return f"**Key Investment Risks for {analysis.company_name}:**\n\n{risks_text}\n\n**Risk Assessment:**\n{analysis.detailed_analysis}"
            elif "opportunity" in query.lower() or "opportunities" in query.lower():
                opps_text = "\n".join([f"• {opp}" for opp in analysis.key_opportunities])
                return f"**Key Investment Opportunities for {analysis.company_name}:**\n\n{opps_text}\n\n**Opportunity Analysis:**\n{analysis.detailed_analysis}"
            else:
                return f"**Comprehensive Analysis for {analysis.company_name}:**\n\n{analysis.executive_summary}\n\n{analysis.detailed_analysis}\n\n**Recommendation:**\n{analysis.recommendation}"
        else:
            return "Please specify a company code (e.g., 2485) in your question for detailed analysis."
    
    def _extract_company_code(self, query: str) -> Optional[str]:
        """Extract company code from query text"""
        # Look for 4-digit codes
        import re
        codes = re.findall(r'\b\d{4}\b', query)
        
        if codes:
            # Check if code exists in our database
            for code in codes:
                if code in self.mock_database:
                    return code
        
        # If no code found, default to sample company
        if any(word in query.lower() for word in ["company", "ティア", "sample"]):
            return "2485"
        
        return None

# Example usage and testing
def main():
    """Demonstrate hedge fund level financial analysis capabilities"""
    print("🏦 HEDGE FUND FINANCIAL INTELLIGENCE AGENT")
    print("=" * 50)
    
    # Initialize agent
    agent = HedgeFundFinancialAgent()
    
    # Test comprehensive analysis
    print("\n📊 COMPREHENSIVE COMPANY ANALYSIS")
    print("-" * 40)
    
    analysis = agent.analyze_company("2485", "Provide investment recommendation")
    
    print(f"Company: {analysis.company_name} ({analysis.company_code})")
    print(f"Overall Score: {analysis.overall_score:.1f}/100")
    print(f"Documents Analyzed: {analysis.documents_analyzed}")
    print(f"Extraction Confidence: {analysis.extraction_confidence:.1%}")
    print()
    
    print("EXECUTIVE SUMMARY:")
    print(analysis.executive_summary)
    print()
    
    print("INVESTMENT RECOMMENDATION:")
    print(analysis.recommendation)
    print()
    
    print("KEY METRICS (Top 10):")
    for i, metric in enumerate(analysis.financial_metrics[:10], 1):
        value_str = f"{metric.value:.3f}" if metric.value else "N/A"
        print(f"{i:2d}. {metric.name}: {value_str}")
        print(f"     → {metric.interpretation}")
    print()
    
    # Test analytical questions
    print("\n🎯 ANALYTICAL Q&A DEMONSTRATION")
    print("-" * 40)
    
    questions = [
        "What is the investment thesis for company 2485?",
        "What are the key risks for investing in company 2485?",
        "Analyze the profitability metrics for company 2485"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("A:", agent.answer_analytical_question(question)[:500] + "..." if len(agent.answer_analytical_question(question)) > 500 else agent.answer_analytical_question(question))
        print()

if __name__ == "__main__":
    main()