# Production-Ready Hybrid Financial Intelligence Agent
# Institutional-grade agent for Japanese financial markets

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache
import hashlib
import time

# LangChain imports
from langchain.schema import BaseRetriever
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ENHANCED CORE COMPONENTS =====

class QueryType(Enum):
    STRUCTURED = "structured"      # Pure SQL queries
    UNSTRUCTURED = "unstructured"  # Pure RAG queries
    HYBRID = "hybrid"              # Requires both approaches
    CONTEXTUAL = "contextual"      # Structured data + narrative context

class QueryComplexity(Enum):
    SIMPLE = "simple"          # Single metric, single company
    MODERATE = "moderate"      # Multiple metrics or companies
    COMPLEX = "complex"        # Cross-company analysis, trends
    EXPERT = "expert"          # Multi-dimensional analysis with context

class DataSource(Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    MIXED = "mixed"
    EXTERNAL = "external"

@dataclass
class Company:
    ticker: str
    name: str
    sector: str
    market_cap: Optional[float] = None
    confidence: float = 1.0

@dataclass
class QueryAnalysis:
    query_type: QueryType
    structured_elements: List[str]
    unstructured_elements: List[str]
    confidence: float
    suggested_approach: str

@dataclass
class QueryIntent:
    companies: List[Company]
    metrics: List[str]
    time_periods: List[str]
    analysis_type: str
    complexity: QueryComplexity
    data_sources: List[DataSource]
    comparative: bool = False
    temporal: bool = False
    confidence: float = 0.0

@dataclass
class AnalysisResult:
    query: str
    intent: QueryIntent
    structured_data: Dict[str, Any] = field(default_factory=dict)
    narrative_analysis: Dict[str, Any] = field(default_factory=dict)
    synthesis: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    visualizations: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    adequacy_score: float = 0.0
    reasoning_notes: str = ""

# ===== QUERY CLASSIFIER =====

class QueryClassifier:
    """Intelligent query classification for routing to appropriate data sources"""
    
    def __init__(self, llm):
        self.llm = llm
        self.classification_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze this financial query and classify it:
            Query: {query}
            
            Classify as:
            1. STRUCTURED: Requires specific numerical data (revenue, profit, ratios)
            2. UNSTRUCTURED: Requires narrative analysis (strategy, risks, outlook)
            3. HYBRID: Requires both numbers and narrative context
            4. CONTEXTUAL: Numbers first, then context to explain them
            
            Return JSON with:
            - query_type: classification
            - structured_elements: list of numerical/factual elements needed
            - unstructured_elements: list of narrative/contextual elements needed
            - confidence: 0.0-1.0
            - suggested_approach: brief explanation
            """
        )
    
    def classify_query(self, query: str) -> QueryAnalysis:
        """Classify query to determine optimal processing approach"""
        try:
            result = self.llm(self.classification_prompt.format(query=query))
            parsed = json.loads(result)
            return QueryAnalysis(
                query_type=QueryType(parsed['query_type'].lower()),
                structured_elements=parsed['structured_elements'],
                unstructured_elements=parsed['unstructured_elements'],
                confidence=parsed['confidence'],
                suggested_approach=parsed['suggested_approach']
            )
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return QueryAnalysis(
                query_type=QueryType.HYBRID,
                structured_elements=[],
                unstructured_elements=[],
                confidence=0.5,
                suggested_approach="Default hybrid approach due to classification error"
            )

# ===== COMPANY IDENTIFICATION SERVICE =====

class CompanyIdentificationService:
    """Production-grade company identification and matching service"""
    
    def __init__(self, sql_db: SQLDatabase, llm):
        self.sql_db = sql_db
        self.llm = llm
        self.company_cache = {}
        self.fuzzy_match_threshold = 0.8
        
        # Load company database
        self._load_company_database()
    
    def _load_company_database(self):
        """Load comprehensive company database with multiple identifiers"""
        try:
            query = """
            SELECT DISTINCT 
                ticker,
                company_name,
                company_name_en,
                sector,
                industry,
                market_cap_jpy,
                aliases,
                keywords
            FROM company_master
            WHERE market = 'TSE' AND status = 'active'
            ORDER BY market_cap_jpy DESC NULLS LAST
            """
            
            companies = self.sql_db.run(query)
            
            # Build comprehensive lookup tables
            self.ticker_to_company = {}
            self.name_to_ticker = {}
            self.alias_to_ticker = {}
            self.sector_companies = {}
            
            for company in companies:
                ticker = company['ticker']
                name = company['company_name']
                name_en = company.get('company_name_en', '')
                sector = company.get('sector', 'Unknown')
                
                # Primary mappings
                self.ticker_to_company[ticker] = {
                    'ticker': ticker,
                    'name': name,
                    'name_en': name_en,
                    'sector': sector,
                    'market_cap': company.get('market_cap_jpy'),
                    'aliases': company.get('aliases', []),
                    'keywords': company.get('keywords', [])
                }
                
                # Name variations
                self.name_to_ticker[name.lower()] = ticker
                if name_en:
                    self.name_to_ticker[name_en.lower()] = ticker
                
                # Aliases
                for alias in company.get('aliases', []):
                    self.alias_to_ticker[alias.lower()] = ticker
                
                # Sector groupings
                if sector not in self.sector_companies:
                    self.sector_companies[sector] = []
                self.sector_companies[sector].append(ticker)
            
            logger.info(f"Loaded {len(companies)} companies into identification service")
            
        except Exception as e:
            logger.error(f"Error loading company database: {e}")
            # Fallback to basic company data
            self._load_fallback_companies()
    
    def _load_fallback_companies(self):
        """Load fallback company data if database query fails"""
        self.ticker_to_company = {
            '7203': {'ticker': '7203', 'name': 'トヨタ自動車', 'name_en': 'Toyota Motor', 'sector': 'Automotive'},
            '6758': {'ticker': '6758', 'name': 'ソニーグループ', 'name_en': 'Sony Group', 'sector': 'Technology'},
            '7267': {'ticker': '7267', 'name': 'ホンダ', 'name_en': 'Honda Motor', 'sector': 'Automotive'},
            '6501': {'ticker': '6501', 'name': '日立製作所', 'name_en': 'Hitachi', 'sector': 'Technology'},
            '8306': {'ticker': '8306', 'name': '三菱UFJフィナンシャル・グループ', 'name_en': 'Mitsubishi UFJ Financial Group', 'sector': 'Banking'}
        }
        
        self.name_to_ticker = {}
        self.alias_to_ticker = {}
        self.sector_companies = {'Automotive': ['7203', '7267'], 'Technology': ['6758', '6501'], 'Banking': ['8306']}
        
        for ticker, info in self.ticker_to_company.items():
            self.name_to_ticker[info['name'].lower()] = ticker
            if info.get('name_en'):
                self.name_to_ticker[info['name_en'].lower()] = ticker
    
    def identify_companies(self, query: str) -> List[Company]:
        """Identify companies mentioned in query with confidence scores"""
        companies = []
        
        # Extract explicit company mentions
        explicit_companies = self._extract_explicit_mentions(query)
        companies.extend(explicit_companies)
        
        # Handle sector/industry mentions
        sector_companies = self._extract_sector_companies(query)
        companies.extend(sector_companies)
        
        # Handle comparative queries
        comparative_companies = self._extract_comparative_companies(query)
        companies.extend(comparative_companies)
        
        # Use LLM for complex company identification
        if not companies:
            llm_companies = self._llm_company_identification(query)
            companies.extend(llm_companies)
        
        # Deduplicate and rank by confidence
        unique_companies = self._deduplicate_companies(companies)
        
        return sorted(unique_companies, key=lambda x: x.confidence, reverse=True)
    
    def _extract_explicit_mentions(self, query: str) -> List[Company]:
        """Extract explicitly mentioned companies by ticker or name"""
        companies = []
        query_lower = query.lower()
        
        # Check for ticker patterns (4-digit numbers)
        ticker_pattern = r'\b\d{4}\b'
        potential_tickers = re.findall(ticker_pattern, query)
        
        for ticker in potential_tickers:
            if ticker in self.ticker_to_company:
                company_info = self.ticker_to_company[ticker]
                companies.append(Company(
                    ticker=ticker,
                    name=company_info['name'],
                    sector=company_info['sector'],
                    market_cap=company_info.get('market_cap'),
                    confidence=0.95
                ))
        
        # Check for company names
        for name, ticker in self.name_to_ticker.items():
            if name in query_lower:
                company_info = self.ticker_to_company[ticker]
                companies.append(Company(
                    ticker=ticker,
                    name=company_info['name'],
                    sector=company_info['sector'],
                    market_cap=company_info.get('market_cap'),
                    confidence=0.9
                ))
        
        return companies
    
    def _extract_sector_companies(self, query: str) -> List[Company]:
        """Extract companies based on sector/industry mentions"""
        companies = []
        query_lower = query.lower()
        
        # Sector keywords mapping
        sector_keywords = {
            'automotive': ['automotive', 'auto', 'car', 'vehicle', 'toyota', 'honda', 'nissan'],
            'technology': ['tech', 'technology', 'semiconductor', 'software', 'ai', 'digital'],
            'banking': ['bank', 'banking', 'financial', 'finance', 'credit'],
            'retail': ['retail', 'shopping', 'consumer', 'store'],
            'manufacturing': ['manufacturing', 'industrial', 'machinery', 'factory'],
            'telecommunications': ['telecom', 'mobile', 'communication', 'network'],
            'energy': ['energy', 'oil', 'gas', 'power', 'utility'],
            'pharmaceuticals': ['pharma', 'pharmaceutical', 'drug', 'medicine', 'healthcare']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if sector in self.sector_companies:
                    # Return top companies in sector by market cap
                    sector_tickers = self.sector_companies[sector][:5]  # Top 5
                    for ticker in sector_tickers:
                        company_info = self.ticker_to_company[ticker]
                        companies.append(Company(
                            ticker=ticker,
                            name=company_info['name'],
                            sector=company_info['sector'],
                            market_cap=company_info.get('market_cap'),
                            confidence=0.7
                        ))
        
        return companies
    
    def _extract_comparative_companies(self, query: str) -> List[Company]:
        """Extract companies from comparative queries"""
        companies = []
        
        # Look for comparative patterns
        comparative_patterns = [
            r'compare\s+(\w+)\s+(?:with|to|and)\s+(\w+)',
            r'(\w+)\s+vs\s+(\w+)',
            r'(\w+)\s+versus\s+(\w+)',
            r'between\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in comparative_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                company1, company2 = match.groups()
                
                # Try to identify each company
                for company_name in [company1, company2]:
                    identified = self._identify_single_company(company_name)
                    if identified:
                        companies.append(identified)
        
        return companies
    
    def _identify_single_company(self, company_name: str) -> Optional[Company]:
        """Identify a single company from a name or ticker"""
        company_name_lower = company_name.lower()
        
        # Check direct name match
        if company_name_lower in self.name_to_ticker:
            ticker = self.name_to_ticker[company_name_lower]
            company_info = self.ticker_to_company[ticker]
            return Company(
                ticker=ticker,
                name=company_info['name'],
                sector=company_info['sector'],
                market_cap=company_info.get('market_cap'),
                confidence=0.9
            )
        
        # Check if it's a ticker
        if company_name.isdigit() and len(company_name) == 4:
            if company_name in self.ticker_to_company:
                company_info = self.ticker_to_company[company_name]
                return Company(
                    ticker=company_name,
                    name=company_info['name'],
                    sector=company_info['sector'],
                    market_cap=company_info.get('market_cap'),
                    confidence=0.95
                )
        
        # Fuzzy matching
        return self._fuzzy_match_company(company_name)
    
    def _fuzzy_match_company(self, company_name: str) -> Optional[Company]:
        """Perform fuzzy matching for company identification"""
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0
        
        for name, ticker in self.name_to_ticker.items():
            similarity = SequenceMatcher(None, company_name.lower(), name).ratio()
            if similarity > best_score and similarity > self.fuzzy_match_threshold:
                best_score = similarity
                best_match = ticker
        
        if best_match:
            company_info = self.ticker_to_company[best_match]
            return Company(
                ticker=best_match,
                name=company_info['name'],
                sector=company_info['sector'],
                market_cap=company_info.get('market_cap'),
                confidence=best_score
            )
        
        return None
    
    def _llm_company_identification(self, query: str) -> List[Company]:
        """Use LLM for complex company identification"""
        identification_prompt = PromptTemplate(
            input_variables=["query", "companies_sample"],
            template="""
            Identify Japanese companies mentioned in this query:
            Query: {query}
            
            Available companies (sample):
            {companies_sample}
            
            Return a JSON array of identified companies with:
            - ticker: company ticker code
            - name: company name
            - confidence: 0.0-1.0 confidence score
            
            Focus on Japanese listed companies. If no specific companies are mentioned, return empty array.
            """
        )
        
        # Sample of major companies for context
        sample_companies = list(self.ticker_to_company.items())[:20]
        sample_text = "\n".join([f"{ticker}: {info['name']}" for ticker, info in sample_companies])
        
        try:
            result = self.llm(identification_prompt.format(
                query=query,
                companies_sample=sample_text
            ))
            
            identified = json.loads(result)
            companies = []
            
            for item in identified:
                ticker = item.get('ticker')
                if ticker and ticker in self.ticker_to_company:
                    company_info = self.ticker_to_company[ticker]
                    companies.append(Company(
                        ticker=ticker,
                        name=company_info['name'],
                        sector=company_info['sector'],
                        market_cap=company_info.get('market_cap'),
                        confidence=item.get('confidence', 0.5)
                    ))
            
            return companies
            
        except Exception as e:
            logger.error(f"LLM company identification failed: {e}")
            return []
    
    def _deduplicate_companies(self, companies: List[Company]) -> List[Company]:
        """Remove duplicate companies and keep highest confidence"""
        ticker_to_company = {}
        
        for company in companies:
            if company.ticker not in ticker_to_company:
                ticker_to_company[company.ticker] = company
            else:
                # Keep the one with higher confidence
                if company.confidence > ticker_to_company[company.ticker].confidence:
                    ticker_to_company[company.ticker] = company
        
        return list(ticker_to_company.values())

# ===== ENHANCED QUERY ANALYSIS =====

class IntelligentQueryAnalyzer:
    """Advanced query analysis with intent recognition and complexity assessment"""
    
    def __init__(self, llm, company_service: CompanyIdentificationService):
        self.llm = llm
        self.company_service = company_service
        
        # Financial metrics taxonomy
        self.metrics_taxonomy = {
            'profitability': ['revenue', 'profit', 'net_income', 'operating_income', 'ebitda', 'margins'],
            'financial_health': ['debt', 'equity', 'assets', 'liabilities', 'cash', 'working_capital'],
            'efficiency': ['roa', 'roe', 'roic', 'asset_turnover', 'inventory_turnover'],
            'valuation': ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'market_cap', 'book_value'],
            'growth': ['revenue_growth', 'profit_growth', 'dividend_growth', 'cagr'],
            'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio', 'debt_to_equity'],
            'market': ['stock_price', 'volume', 'volatility', 'beta', 'dividend_yield']
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Comprehensive query analysis with intent recognition"""
        
        # Identify companies
        companies = self.company_service.identify_companies(query)
        
        # Extract metrics and time periods
        metrics = self._extract_metrics(query)
        time_periods = self._extract_time_periods(query)
        
        # Determine analysis type
        analysis_type = self._determine_analysis_type(query)
        
        # Assess complexity
        complexity = self._assess_complexity(query, companies, metrics)
        
        # Determine data sources needed
        data_sources = self._determine_data_sources(query, analysis_type)
        
        # Check for comparative and temporal analysis
        comparative = self._is_comparative_query(query)
        temporal = self._is_temporal_query(query)
        
        # Calculate overall confidence
        confidence = self._calculate_intent_confidence(companies, metrics, analysis_type)
        
        return QueryIntent(
            companies=companies,
            metrics=metrics,
            time_periods=time_periods,
            analysis_type=analysis_type,
            complexity=complexity,
            data_sources=data_sources,
            comparative=comparative,
            temporal=temporal,
            confidence=confidence
        )
    
    def _extract_metrics(self, query: str) -> List[str]:
        """Extract financial metrics from query"""
        metrics = []
        query_lower = query.lower()
        
        # Direct metric mentions
        all_metrics = []
        for category, metric_list in self.metrics_taxonomy.items():
            all_metrics.extend(metric_list)
        
        for metric in all_metrics:
            if metric.replace('_', ' ') in query_lower or metric in query_lower:
                metrics.append(metric)
        
        # Use LLM for complex metric extraction
        if not metrics:
            metrics = self._llm_extract_metrics(query)
        
        return metrics
    
    def _llm_extract_metrics(self, query: str) -> List[str]:
        """Use LLM to extract financial metrics from complex queries"""
        metrics_prompt = PromptTemplate(
            input_variables=["query", "metrics_taxonomy"],
            template="""
            Extract financial metrics from this query:
            Query: {query}
            
            Available metrics:
            {metrics_taxonomy}
            
            Return a JSON array of relevant metrics mentioned or implied in the query.
            If the query asks for "financial performance", include key profitability and health metrics.
            If no specific metrics are mentioned, return an empty array.
            """
        )
        
        try:
            result = self.llm(metrics_prompt.format(
                query=query,
                metrics_taxonomy=json.dumps(self.metrics_taxonomy, indent=2)
            ))
            return json.loads(result)
        except:
            return []
    
    def _extract_time_periods(self, query: str) -> List[str]:
        """Extract time periods from query"""
        periods = []
        
        # Year patterns
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        periods.extend(years)
        
        # Relative time patterns
        relative_patterns = {
            'last_year': r'last year|previous year',
            'this_year': r'this year|current year',
            'last_quarter': r'last quarter|previous quarter',
            'this_quarter': r'this quarter|current quarter',
            'ytd': r'year to date|ytd',
            'last_5_years': r'last 5 years|past 5 years',
            'last_3_years': r'last 3 years|past 3 years'
        }
        
        for period_name, pattern in relative_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                periods.append(period_name)
        
        return periods if periods else ['latest']
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis required"""
        query_lower = query.lower()
        
        analysis_types = {
            'comparison': ['compare', 'versus', 'vs', 'difference', 'better', 'worse'],
            'trend': ['trend', 'over time', 'growth', 'change', 'evolution'],
            'valuation': ['valuation', 'value', 'worth', 'price', 'expensive', 'cheap'],
            'performance': ['performance', 'how well', 'results', 'success'],
            'risk': ['risk', 'volatile', 'risky', 'safe', 'stable'],
            'forecast': ['forecast', 'predict', 'future', 'outlook', 'projection'],
            'screening': ['find', 'identify', 'screen', 'filter', 'best', 'top']
        }
        
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return analysis_type
        
        return 'descriptive'  # Default
    
    def _assess_complexity(self, query: str, companies: List[Company], metrics: List[str]) -> QueryComplexity:
        """Assess query complexity based on multiple factors"""
        complexity_score = 0
        
        # Factor 1: Number of companies
        if len(companies) == 0:
            complexity_score += 0  # No companies identified
        elif len(companies) == 1:
            complexity_score += 1  # Single company
        elif len(companies) <= 3:
            complexity_score += 2  # Few companies
        else:
            complexity_score += 3  # Many companies
        
        # Factor 2: Number of metrics
        if len(metrics) == 0:
            complexity_score += 0
        elif len(metrics) <= 2:
            complexity_score += 1
        elif len(metrics) <= 5:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # Factor 3: Query characteristics
        query_lower = query.lower()
        complex_keywords = ['analyze', 'comprehensive', 'detailed', 'deep dive', 'thorough']
        if any(keyword in query_lower for keyword in complex_keywords):
            complexity_score += 2
        
        # Factor 4: Temporal analysis
        temporal_keywords = ['trend', 'over time', 'historical', 'forecast']
        if any(keyword in query_lower for keyword in temporal_keywords):
            complexity_score += 1
        
        # Factor 5: Comparative analysis
        comparative_keywords = ['compare', 'versus', 'vs', 'against']
        if any(keyword in query_lower for keyword in comparative_keywords):
            complexity_score += 1
        
        # Map score to complexity enum
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 4:
            return QueryComplexity.MODERATE
        elif complexity_score <= 6:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
    
    def _determine_data_sources(self, query: str, analysis_type: str) -> List[DataSource]:
        """Determine which data sources are needed"""
        sources = []
        query_lower = query.lower()
        
        # Structured data indicators
        structured_keywords = ['revenue', 'profit', 'financial', 'ratio', 'debt', 'cash', 'price']
        if any(keyword in query_lower for keyword in structured_keywords):
            sources.append(DataSource.STRUCTURED)
        
        # Unstructured data indicators
        unstructured_keywords = ['strategy', 'management', 'outlook', 'risk', 'commentary', 'guidance']
        if any(keyword in query_lower for keyword in unstructured_keywords):
            sources.append(DataSource.UNSTRUCTURED)
        
        # Mixed analysis needs both
        if analysis_type in ['comparison', 'performance', 'forecast']:
            if DataSource.STRUCTURED not in sources:
                sources.append(DataSource.STRUCTURED)
            if DataSource.UNSTRUCTURED not in sources:
                sources.append(DataSource.UNSTRUCTURED)
        
        # Default to mixed if unclear
        if not sources:
            sources = [DataSource.MIXED]
        
        return sources
    
    def _is_comparative_query(self, query: str) -> bool:
        """Check if query requires comparative analysis"""
        comparative_keywords = ['compare', 'versus', 'vs', 'against', 'difference', 'better', 'worse']
        return any(keyword in query.lower() for keyword in comparative_keywords)
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query requires temporal analysis"""
        temporal_keywords = ['trend', 'over time', 'growth', 'change', 'historical', 'evolution']
        return any(keyword in query.lower() for keyword in temporal_keywords)
    
    def _calculate_intent_confidence(self, companies: List[Company], metrics: List[str], analysis_type: str) -> float:
        """Calculate confidence in intent understanding"""
        confidence = 0.0
        
        # Company identification confidence
        if companies:
            avg_company_confidence = sum(c.confidence for c in companies) / len(companies)
            confidence += avg_company_confidence * 0.4
        
        # Metrics clarity
        if metrics:
            confidence += 0.3
        
        # Analysis type clarity
        if analysis_type != 'descriptive':
            confidence += 0.3
        
        return min(confidence, 1.0)

# ===== PROCESSING ENGINES =====

class StructuredDataProcessor:
    """Processes structured financial data queries"""
    
    def __init__(self, sql_db: SQLDatabase, llm):
        self.sql_db = sql_db
        self.llm = llm
    
    async def process(self, intent: QueryIntent) -> Dict[str, Any]:
        """Process structured data query based on intent"""
        results = {}
        
        for company in intent.companies:
            company_data = await self._get_company_data(company, intent.metrics, intent.time_periods)
            results[company.ticker] = company_data
        
        return results
    
    async def _get_company_data(self, company: Company, metrics: List[str], time_periods: List[str]) -> Dict[str, Any]:
        """Get structured data for a specific company"""
        try:
            # Build dynamic query based on company and metrics
            query = """
            SELECT 
                fm.ticker,
                fm.period_end_date,
                fm.metric_name,
                fm.metric_value_jpy,
                d.company_name
            FROM financial_metrics fm
            JOIN documents d ON fm.document_id = d.id
            WHERE fm.ticker = %s
            """
            
            params = [company.ticker]
            
            if metrics:
                placeholders = ",".join(["%s"] * len(metrics))
                query += f" AND fm.metric_name IN ({placeholders})"
                params.extend(metrics)
            
            # Add time period filtering
            if time_periods and 'latest' not in time_periods:
                # Add time filtering logic here
                pass
            
            query += " ORDER BY fm.period_end_date DESC LIMIT 50"
            
            result = self.sql_db.run(query, params)
            
            return {
                'company': company.name,
                'ticker': company.ticker,
                'data': result,
                'data_points': len(result) if result else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting company data for {company.ticker}: {e}")
            return {'company': company.name, 'ticker': company.ticker, 'data': [], 'error': str(e)}

class NarrativeAnalysisProcessor:
    """Processes unstructured narrative analysis queries"""
    
    def __init__(self, vector_retriever: BaseRetriever, llm):
        self.vector_retriever = vector_retriever
        self.llm = llm
    
    async def process(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Process narrative analysis query"""
        try:
            # Enhanced query for better retrieval
            enhanced_query = self._enhance_query_for_retrieval(query, intent)
            
            # Retrieve relevant documents
            docs = self.vector_retriever.get_relevant_documents(enhanced_query)
            
            # Analyze and synthesize
            analysis = await self._analyze_narrative(query, docs, intent)
            
            return {
                'query': query,
                'enhanced_query': enhanced_query,
                'analysis': analysis,
                'sources': [doc.metadata for doc in docs],
                'confidence': self._calculate_confidence(docs),
                'document_count': len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error in narrative analysis: {e}")
            return {'error': str(e), 'analysis': '', 'sources': [], 'confidence': 0.0}
    
    def _enhance_query_for_retrieval(self, query: str, intent: QueryIntent) -> str:
        """Enhance query with company and context information"""
        enhanced_parts = [query]
        
        # Add company names for better retrieval
        for company in intent.companies:
            enhanced_parts.append(company.name)
            if hasattr(company, 'name_en') and company.name_en:
                enhanced_parts.append(company.name_en)
        
        # Add relevant financial terms
        if intent.metrics:
            enhanced_parts.extend(intent.metrics)
        
        return " ".join(enhanced_parts)
    
    async def _analyze_narrative(self, query: str, docs: List, intent: QueryIntent) -> str:
        """Analyze narrative content to answer query"""
        analysis_prompt = PromptTemplate(
            input_variables=["query", "documents", "companies"],
            template="""
            Analyze these corporate documents to answer the question:
            Question: {query}
            
            Focus on these companies: {companies}
            
            Documents:
            {documents}
            
            Provide a detailed analysis with specific examples from the documents.
            Focus on the financial and strategic aspects relevant to the question.
            """
        )
        
        doc_content = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                  for i, doc in enumerate(docs[:10])])  # Limit to top 10 docs
        
        company_names = ", ".join([c.name for c in intent.companies])
        
        return self.llm(analysis_prompt.format(
            query=query, 
            documents=doc_content,
            companies=company_names
        ))
    
    def _calculate_confidence(self, docs: List) -> float:
        """Calculate confidence score based on document relevance"""
        if not docs:
            return 0.0
        
        # Simple confidence calculation based on number and recency of documents
        confidence = min(len(docs) / 10.0, 1.0)  # More docs = higher confidence
        return confidence

class SynthesisEngine:
    """Combines structured and unstructured analysis into coherent responses"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def hybrid_synthesis(self, query: str, intent: QueryIntent, 
                             structured_data: Dict, narrative_data: Dict) -> str:
        """Synthesize hybrid analysis combining structured and narrative data"""
        
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "structured_data", "narrative_analysis", "companies"],
            template="""
            Provide a comprehensive analysis combining quantitative data and qualitative insights:
            
            Question: {query}
            Companies: {companies}
            
            Quantitative Data:
            {structured_data}
            
            Qualitative Analysis:
            {narrative_analysis}
            
            Synthesize both sources to provide a complete answer that:
            1. Uses specific numbers from the quantitative data
            2. Explains the context using qualitative insights
            3. Identifies key trends and patterns
            4. Provides actionable insights
            5. Notes any limitations or uncertainties
            """
        )
        
        company_names = ", ".join([c.name for c in intent.companies])
        
        return self.llm(synthesis_prompt.format(
            query=query,
            companies=company_names,
            structured_data=json.dumps(structured_data, indent=2, default=str),
            narrative_analysis=narrative_data.get('analysis', 'No narrative analysis available')
        ))
    
    async def expert_synthesis(self, query: str, intent: QueryIntent, 
                             combined_data: Dict) -> str:
        """Advanced synthesis for expert-level analysis"""
        
        expert_prompt = PromptTemplate(
            input_variables=["query", "analysis_data", "companies", "complexity"],
            template="""
            Provide an institutional-grade financial analysis:
            
            Question: {query}
            Companies: {companies}
            Analysis Complexity: {complexity}
            
            Analysis Components:
            {analysis_data}
            
            Deliver a comprehensive analysis that:
            1. Provides executive summary with key findings
            2. Detailed quantitative analysis with specific metrics
            3. Qualitative assessment of strategic positioning
            4. Risk assessment and key considerations
            5. Comparative analysis where relevant
            6. Future outlook and implications
            7. Investment recommendations or considerations
            8. Data quality and limitation notes
            
            Structure the response professionally for institutional use.
            """
        )
        
        company_names = ", ".join([c.name for c in intent.companies])
        
        return self.llm(expert_prompt.format(
            query=query,
            companies=company_names,
            complexity=intent.complexity.value,
            analysis_data=json.dumps(combined_data, indent=2, default=str)
        ))

class ValidationEngine:
    """Validates analysis results for accuracy and completeness"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
    
    async def validate(self, result: AnalysisResult) -> AnalysisResult:
        """Validate the analysis result"""
        
        # Extract and validate numerical claims
        numerical_validation = await self._validate_numerical_claims(result)
        
        # Check for logical consistency
        consistency_check = await self._check_logical_consistency(result)
        
        # Assess completeness
        completeness_score = self._assess_completeness(result)
        
        # Update result with validation information
        result.metadata['validation'] = {
            'numerical_validation': numerical_validation,
            'consistency_check': consistency_check,
            'completeness_score': completeness_score
        }
        
        # Adjust confidence based on validation results
        validation_confidence = (numerical_validation['score'] + 
                               consistency_check['score'] + 
                               completeness_score) / 3
        
        result.confidence = min(result.confidence * validation_confidence, 1.0)
        
        return result
    
    async def _validate_numerical_claims(self, result: AnalysisResult) -> Dict[str, Any]:
        """Validate numerical claims in the analysis"""
        
        validation_prompt = PromptTemplate(
            input_variables=["synthesis", "structured_data"],
            template="""
            Extract and validate numerical claims from this analysis:
            
            Analysis: {synthesis}
            
            Supporting Data: {structured_data}
            
            For each numerical claim:
            1. Identify the specific number or percentage
            2. Check if it's supported by the data
            3. Note any discrepancies
            
            Return validation results as JSON with score (0.0-1.0) and notes.
            """
        )
        
        try:
            validation_result = self.llm(validation_prompt.format(
                synthesis=result.synthesis,
                structured_data=json.dumps(result.structured_data, default=str)
            ))
            
            return json.loads(validation_result)
        except:
            return {'score': 0.5, 'notes': 'Validation failed'}
    
    async def _check_logical_consistency(self, result: AnalysisResult) -> Dict[str, Any]:
        """Check for logical consistency in the analysis"""
        
        consistency_prompt = PromptTemplate(
            input_variables=["synthesis"],
            template="""
            Check this financial analysis for logical consistency:
            
            Analysis: {synthesis}
            
            Look for:
            1. Contradictory statements
            2. Unsupported conclusions
            3. Logical flow issues
            4. Missing context
            
            Return consistency assessment as JSON with score (0.0-1.0) and notes.
            """
        )
        
        try:
            consistency_result = self.llm(consistency_prompt.format(
                synthesis=result.synthesis
            ))
            
            return json.loads(consistency_result)
        except:
            return {'score': 0.7, 'notes': 'Consistency check failed'}
    
    def _assess_completeness(self, result: AnalysisResult) -> float:
        """Assess completeness of the analysis"""
        completeness_score = 0.0
        
        # Check if synthesis is provided
        if result.synthesis and len(result.synthesis) > 100:
            completeness_score += 0.3
        
        # Check if structured data is provided
        if result.structured_data:
            completeness_score += 0.3
        
        # Check if narrative analysis is provided
        if result.narrative_analysis:
            completeness_score += 0.2
        
        # Check if sources are provided
        if result.sources:
            completeness_score += 0.2
        
        return completeness_score

class PerformanceMonitor:
    """Monitors and tracks agent performance"""
    
    def __init__(self):
        self.query_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'average_response_time': 0.0,
            'average_confidence': 0.0,
            'error_rate': 0.0
        }
    
    def record_query(self, query: str, execution_time: float, confidence: float, error: bool = False):
        """Record query performance metrics"""
        self.query_history.append({
            'query': query,
            'execution_time': execution_time,
            'confidence': confidence,
            'error': error,
            'timestamp': datetime.now()
        })
        
        # Update aggregated metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update aggregated performance metrics"""
        if not self.query_history:
            return
        
        self.performance_metrics['total_queries'] = len(self.query_history)
        
        execution_times = [q['execution_time'] for q in self.query_history]
        self.performance_metrics['average_response_time'] = sum(execution_times) / len(execution_times)
        
        confidences = [q['confidence'] for q in self.query_history]
        self.performance_metrics['average_confidence'] = sum(confidences) / len(confidences)
        
        errors = [q['error'] for q in self.query_history]
        self.performance_metrics['error_rate'] = sum(errors) / len(errors)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'metrics': self.performance_metrics,
            'recent_queries': self.query_history[-10:],  # Last 10 queries
            'trends': self._calculate_trends()
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.query_history) < 2:
            return {}
        
        # Simple trend calculation over last 10 queries
        recent_queries = self.query_history[-10:]
        older_queries = self.query_history[-20:-10] if len(self.query_history) >= 20 else []
        
        trends = {}
        if older_queries:
            recent_avg_time = sum(q['execution_time'] for q in recent_queries) / len(recent_queries)
            older_avg_time = sum(q['execution_time'] for q in older_queries) / len(older_queries)
            trends['response_time_trend'] = (recent_avg_time - older_avg_time) / older_avg_time
            
            recent_avg_confidence = sum(q['confidence'] for q in recent_queries) / len(recent_queries)
            older_avg_confidence = sum(q['confidence'] for q in older_queries) / len(older_queries)
            trends['confidence_trend'] = (recent_avg_confidence - older_avg_confidence) / older_avg_confidence
        
        return trends

# ===== REASONING AGENT FOR ANSWER ADEQUACY =====

class ReasoningAgent:
    """Agent that evaluates the adequacy and quality of analysis results"""
    
    def __init__(self, llm):
        self.llm = llm
        self.adequacy_criteria = {
            'completeness': 0.25,  # Does it answer the full question?
            'accuracy': 0.25,      # Are the facts and numbers correct?
            'relevance': 0.20,     # Is the information relevant to the query?
            'clarity': 0.15,       # Is the response clear and well-structured?
            'actionability': 0.15  # Does it provide actionable insights?
        }
    
    async def evaluate_answer_adequacy(self, result: AnalysisResult) -> AnalysisResult:
        """Evaluate the adequacy of the analysis result"""
        
        # Evaluate each criterion
        evaluation_scores = {}
        
        for criterion, weight in self.adequacy_criteria.items():
            score = await self._evaluate_criterion(result, criterion)
            evaluation_scores[criterion] = score
        
        # Calculate weighted adequacy score
        adequacy_score = sum(score * weight for score, weight in 
                           zip(evaluation_scores.values(), self.adequacy_criteria.values()))
        
        # Generate reasoning notes
        reasoning_notes = await self._generate_reasoning_notes(result, evaluation_scores)
        
        # Update result with adequacy assessment
        result.adequacy_score = adequacy_score
        result.reasoning_notes = reasoning_notes
        result.metadata['adequacy_evaluation'] = evaluation_scores
        
        # Suggest improvements if adequacy is low
        if adequacy_score < 0.7:
            improvements = await self._suggest_improvements(result, evaluation_scores)
            result.recommendations.extend(improvements)
        
        return result
    
    async def _evaluate_criterion(self, result: AnalysisResult, criterion: str) -> float:
        """Evaluate a specific adequacy criterion"""
        
        criterion_prompts = {
            'completeness': """
                Evaluate how completely this analysis answers the original question:
                
                Question: {query}
                Analysis: {synthesis}
                
                Score from 0.0 to 1.0 based on:
                - Are all parts of the question addressed?
                - Is any critical information missing?
                - Does it cover the requested scope?
                
                Return only the numerical score.
            """,
            'accuracy': """
                Evaluate the accuracy of this financial analysis:
                
                Analysis: {synthesis}
                Supporting Data: {structured_data}
                
                Score from 0.0 to 1.0 based on:
                - Are numerical claims supported by data?
                - Are financial concepts used correctly?
                - Are there any obvious errors?
                
                Return only the numerical score.
            """,
            'relevance': """
                Evaluate how relevant this analysis is to the question:
                
                Question: {query}
                Analysis: {synthesis}
                
                Score from 0.0 to 1.0 based on:
                - Does it stay focused on the question?
                - Is unnecessary information minimized?
                - Are the examples and data points relevant?
                
                Return only the numerical score.
            """,
            'clarity': """
                Evaluate the clarity and structure of this analysis:
                
                Analysis: {synthesis}
                
                Score from 0.0 to 1.0 based on:
                - Is it well-organized and logical?
                - Is the language clear and professional?
                - Are complex concepts explained well?
                
                Return only the numerical score.
            """,
            'actionability': """
                Evaluate how actionable this financial analysis is:
                
                Analysis: {synthesis}
                
                Score from 0.0 to 1.0 based on:
                - Does it provide specific insights?
                - Are there clear implications or recommendations?
                - Can a reader take action based on this information?
                
                Return only the numerical score.
            """
        }
        
        try:
            prompt = criterion_prompts[criterion].format(
                query=result.query,
                synthesis=result.synthesis,
                structured_data=json.dumps(result.structured_data, default=str)
            )
            
            score_text = self.llm(prompt)
            score = float(re.search(r'(\d+\.?\d*)', score_text).group(1))
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error evaluating {criterion}: {e}")
            return 0.5  # Default neutral score
    
    async def _generate_reasoning_notes(self, result: AnalysisResult, 
                                      evaluation_scores: Dict[str, float]) -> str:
        """Generate reasoning notes explaining the adequacy evaluation"""
        
        reasoning_prompt = PromptTemplate(
            input_variables=["query", "synthesis", "evaluation_scores"],
            template="""
            Generate reasoning notes for this analysis evaluation:
            
            Original Question: {query}
            Analysis: {synthesis}
            
            Evaluation Scores:
            {evaluation_scores}
            
            Provide concise reasoning notes that explain:
            1. What the analysis does well
            2. Where it could be improved
            3. Overall assessment of adequacy
            
            Keep it professional and constructive.
            """
        )
        
        try:
            reasoning = self.llm(reasoning_prompt.format(
                query=result.query,
                synthesis=result.synthesis,
                evaluation_scores=json.dumps(evaluation_scores, indent=2)
            ))
            return reasoning
        except Exception as e:
            logger.error(f"Error generating reasoning notes: {e}")
            return "Unable to generate detailed reasoning notes due to processing error."
    
    async def _suggest_improvements(self, result: AnalysisResult, 
                                  evaluation_scores: Dict[str, float]) -> List[str]:
        """Suggest specific improvements for low-scoring criteria"""
        
        improvements = []
        
        # Identify low-scoring criteria
        low_scoring_criteria = [criterion for criterion, score in evaluation_scores.items() 
                               if score < 0.6]
        
        improvement_suggestions = {
            'completeness': [
                "Consider addressing all parts of the original question",
                "Include additional relevant metrics or time periods",
                "Provide more comprehensive coverage of the topic"
            ],
            'accuracy': [
                "Verify numerical claims against source data",
                "Double-check financial calculations and ratios",
                "Ensure proper use of financial terminology"
            ],
            'relevance': [
                "Focus more directly on the specific question asked",
                "Remove tangential information",
                "Prioritize most relevant data points and insights"
            ],
            'clarity': [
                "Improve organization and logical flow",
                "Simplify complex explanations",
                "Add clear headings or structure"
            ],
            'actionability': [
                "Provide specific recommendations or implications",
                "Include clear next steps or considerations",
                "Highlight key takeaways for decision-making"
            ]
        }
        
        for criterion in low_scoring_criteria:
            if criterion in improvement_suggestions:
                improvements.extend(improvement_suggestions[criterion])
        
        return improvements[:3]  # Return top 3 suggestions

# ===== PRODUCTION-GRADE HYBRID AGENT =====

class ProductionHybridFinancialAgent:
    """Production-ready institutional-grade hybrid financial intelligence agent"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm, config: Dict[str, Any] = None):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.config = config or {}
        
        # Initialize core services
        self.company_service = CompanyIdentificationService(sql_db, llm)
        self.query_analyzer = IntelligentQueryAnalyzer(llm, self.company_service)
        self.query_classifier = QueryClassifier(llm)
        
        # Initialize specialized processors
        self.structured_processor = StructuredDataProcessor(sql_db, llm)
        self.narrative_processor = NarrativeAnalysisProcessor(vector_retriever, llm)
        self.synthesis_engine = SynthesisEngine(llm)
        self.validation_engine = ValidationEngine(sql_db, vector_retriever, llm)
        self.reasoning_agent = ReasoningAgent(llm)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 interactions
            return_messages=True
        )
        
        logger.info("Production Hybrid Financial Agent initialized successfully")
    
    async def process_query(self, query: str, user_context: Dict[str, Any] = None) -> AnalysisResult:
        """Main entry point for query processing"""
        start_time = time.time()
        
        try:
            # Step 1: Classify and analyze query
            logger.info(f"Processing query: {query}")
            query_classification = self.query_classifier.classify_query(query)
            intent = self.query_analyzer.analyze_query(query)
            
            # Step 2: Validate companies were identified if needed
            if not intent.companies and self._requires_company_context(intent):
                return await self._handle_no_companies_identified(query, intent)
            
            # Step 3: Route to appropriate processing pipeline
            if intent.complexity == QueryComplexity.EXPERT:
                result = await self._expert_analysis_pipeline(query, intent, query_classification)
            elif intent.complexity == QueryComplexity.COMPLEX:
                result = await self._complex_analysis_pipeline(query, intent, query_classification)
            else:
                result = await self._standard_analysis_pipeline(query, intent, query_classification)
            
            # Step 4: Validate result
            validated_result = await self.validation_engine.validate(result)
            
            # Step 5: Evaluate answer adequacy
            final_result = await self.reasoning_agent.evaluate_answer_adequacy(validated_result)
            
            # Step 6: Update performance metrics
            execution_time = time.time() - start_time
            final_result.execution_time = execution_time
            
            self.performance_monitor.record_query(query, execution_time, final_result.confidence)
            
            # Step 7: Update conversation memory
            self._update_memory(query, final_result)
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            execution_time = time.time() - start_time
            self.performance_monitor.record_query(query, execution_time, 0.0, error=True)
            return self._create_error_result(query, str(e))
    
    async def _expert_analysis_pipeline(self, query: str, intent: QueryIntent, 
                                      classification: QueryAnalysis) -> AnalysisResult:
        """Pipeline for expert-level analysis requiring comprehensive approach"""
        
        # Parallel execution of multiple analysis components
        tasks = []
        
        # Always include structured and narrative analysis for expert queries
        tasks.append(self.structured_processor.process(intent))
        tasks.append(self.narrative_processor.process(query, intent))
        
        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        structured_data = results[0] if not isinstance(results[0], Exception) else {}
        narrative_data = results[1] if not isinstance(results[1], Exception) else {}
        
        # Synthesize comprehensive response
        synthesis = await self.synthesis_engine.expert_synthesis(
            query, intent, {'structured': structured_data, 'narrative': narrative_data}
        )
        
        return AnalysisResult(
            query=query,
            intent=intent,
            structured_data=structured_data,
            narrative_analysis=narrative_data,
            synthesis=synthesis,
            confidence=self._calculate_result_confidence(structured_data, narrative_data),
            sources=narrative_data.get('sources', [])
        )
    
    async def _complex_analysis_pipeline(self, query: str, intent: QueryIntent, 
                                       classification: QueryAnalysis) -> AnalysisResult:
        """Pipeline for complex analysis requiring multiple data sources"""
        
        # Sequential processing based on classification
        if classification.query_type == QueryType.STRUCTURED:
            structured_data = await self.structured_processor.process(intent)
            narrative_data = {}
            synthesis = await self._synthesize_structured_only(query, intent, structured_data)
        elif classification.query_type == QueryType.UNSTRUCTURED:
            structured_data = {}
            narrative_data = await self.narrative_processor.process(query, intent)
            synthesis = narrative_data.get('analysis', 'Analysis not available')
        else:  # HYBRID or CONTEXTUAL
            structured_data = await self.structured_processor.process(intent)
            narrative_data = await self.narrative_processor.process(query, intent)
            synthesis = await self.synthesis_engine.hybrid_synthesis(
                query, intent, structured_data, narrative_data
            )
        
        return AnalysisResult(
            query=query,
            intent=intent,
            structured_data=structured_data,
            narrative_analysis=narrative_data,
            synthesis=synthesis,
            confidence=self._calculate_result_confidence(structured_data, narrative_data),
            sources=narrative_data.get('sources', [])
        )
    
    async def _standard_analysis_pipeline(self, query: str, intent: QueryIntent, 
                                        classification: QueryAnalysis) -> AnalysisResult:
        """Pipeline for standard analysis queries"""
        
        # Simple routing based on classification
        if classification.query_type == QueryType.STRUCTURED:
            structured_data = await self.structured_processor.process(intent)
            synthesis = await self._synthesize_structured_only(query, intent, structured_data)
            return AnalysisResult(
                query=query,
                intent=intent,
                structured_data=structured_data,
                synthesis=synthesis,
                confidence=self._calculate_result_confidence(structured_data, {}),
                sources=[]
            )
        else:
            narrative_data = await self.narrative_processor.process(query, intent)
            return AnalysisResult(
                query=query,
                intent=intent,
                narrative_analysis=narrative_data,
                synthesis=narrative_data.get('analysis', 'Analysis not available'),
                confidence=self._calculate_result_confidence({}, narrative_data),
                sources=narrative_data.get('sources', [])
            )
    
    async def _synthesize_structured_only(self, query: str, intent: QueryIntent, 
                                        structured_data: Dict) -> str:
        """Synthesize response for structured-only queries"""
        
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "data", "companies"],
            template="""
            Analyze this financial data to answer the question:
            
            Question: {query}
            Companies: {companies}
            
            Financial Data:
            {data}
            
            Provide a clear, concise analysis that:
            1. Directly answers the question using the data
            2. Highlights key findings and trends
            3. Provides context for the numbers
            4. Notes any limitations in the data
            """
        )
        
        company_names = ", ".join([c.name for c in intent.companies])
        
        return self.llm(synthesis_prompt.format(
            query=query,
            companies=company_names,
            data=json.dumps(structured_data, indent=2, default=str)
        ))
    
    def _requires_company_context(self, intent: QueryIntent) -> bool:
        """Check if the query requires specific company context"""
        # If query is about specific metrics or comparisons, we need companies
        return (intent.metrics or intent.comparative or 
                intent.analysis_type in ['comparison', 'performance', 'valuation'])
    
    async def _handle_no_companies_identified(self, query: str, intent: QueryIntent) -> AnalysisResult:
        """Handle case where no companies were identified but they're needed"""
        
        synthesis = f"""
        I was unable to identify specific companies from your query: "{query}"
        
        To provide a comprehensive financial analysis, I need you to specify which companies 
        you'd like me to analyze. You can provide:
        - Company names (e.g., "Toyota", "Sony")
        - Ticker symbols (e.g., "7203", "6758")
        - Industry sectors (e.g., "automotive companies", "tech companies")
        
        Please rephrase your question with specific company information, and I'll be happy 
        to provide a detailed analysis.
        """
        
        return AnalysisResult(
            query=query,
            intent=intent,
            synthesis=synthesis,
            confidence=0.8,  # High confidence in the clarification request
            recommendations=["Specify target companies for analysis"],
            metadata={'issue': 'no_companies_identified'}
        )
    
    def _calculate_result_confidence(self, structured_data: Dict, narrative_data: Dict) -> float:
        """Calculate overall confidence in the result"""
        confidence = 0.0
        
        # Structured data contribution
        if structured_data:
            data_quality = 0.8 if any(structured_data.values()) else 0.3
            confidence += data_quality * 0.5
        
        # Narrative data contribution
        if narrative_data:
            narrative_confidence = narrative_data.get('confidence', 0.5)
            confidence += narrative_confidence * 0.5
        
        # Ensure we have some minimum confidence
        return max(confidence, 0.3)
    
    def _update_memory(self, query: str, result: AnalysisResult):
        """Update conversation memory with query and result"""
        memory_entry = f"Q: {query}\nA: {result.synthesis[:200]}..."
        self.memory.save_context(
            {"input": query},
            {"output": memory_entry}
        )
    
    def _create_error_result(self, query: str, error_message: str) -> AnalysisResult:
        """Create error result for failed queries"""
        return AnalysisResult(
            query=query,
            intent=QueryIntent(
                companies=[],
                metrics=[],
                time_periods=[],
                analysis_type='error',
                complexity=QueryComplexity.SIMPLE,
                data_sources=[]
            ),
            synthesis=f"I encountered an error processing your query: {error_message}. Please try rephrasing your question or contact support if the issue persists.",
            confidence=0.0,
            metadata={'error': error_message}
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return self.performance_monitor.get_performance_report()

# ===== USAGE EXAMPLE =====

async def main():
    """Example usage of the Production Hybrid Financial Agent"""
    
    # Initialize components (you would replace these with your actual implementations)
    sql_db = SQLDatabase.from_uri("postgresql://user:pass@localhost/financial_db")
    vector_retriever = None  # Your vector retriever implementation
    llm = None  # Your LLM implementation
    
    # Initialize the agent
    agent = ProductionHybridFinancialAgent(sql_db, vector_retriever, llm)
    
    # Example queries
    test_queries = [
        "What was Toyota's revenue growth in 2023?",
        "Compare the profitability of Toyota and Honda over the last 3 years",
        "What are the main risk factors mentioned in Sony's latest earnings report?",
        "Analyze the automotive sector's performance and provide investment recommendations"
    ]
    
    # Process queries
    for query in test_queries:
        print(f"\nProcessing: {query}")
        result = await agent.process_query(query)
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Adequacy Score: {result.adequacy_score:.2f}")
        print(f"Synthesis: {result.synthesis[:200]}...")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print("-" * 80)
    
    # Get performance report
    performance = agent.get_performance_report()
    print(f"\nPerformance Report: {performance}")

if __name__ == "__main__":
    asyncio.run(main())