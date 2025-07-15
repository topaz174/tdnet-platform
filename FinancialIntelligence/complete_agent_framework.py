"""
LangGraph-based agent framework for financial analytics
Integrates with the Reason-ModernColBERT retrieval system
"""

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime, date, timedelta
import json
import re
from dataclasses import dataclass
import asyncio
import logging
import math
from statistics import mean, median, stdev

from langchain.schema import Document
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np

from retrieval_system import FinancialRetrievalSystem, RetrievalConfig, RetrievalResult

class AgentState(TypedDict):
    query: str
    reasoning_steps: List[str]
    search_results: List[RetrievalResult]
    financial_data: Dict[str, Any]
    calculations: Dict[str, float]
    response: str
    metadata: Dict[str, Any]

@dataclass
class FinancialQuery:
    original_query: str
    query_type: str  # earnings, dividends, management, m&a, metrics, comparison
    companies: List[str]
    time_period: Optional[tuple]
    financial_metrics: List[str]
    comparison_type: Optional[str]

class FinancialEntityExtractor:
    """Extract financial entities and parameters from queries"""
    
    def __init__(self):
        self.company_patterns = [
            r'([A-Z]{2,6})',  # Stock symbols
            r'(\d{4,5})',     # Japanese company codes
        ]
        
        self.time_patterns = {
            'last_quarter': timedelta(days=90),
            'last_3_months': timedelta(days=90),
            'last_6_months': timedelta(days=180),
            'last_year': timedelta(days=365),
            'ytd': None,  # Year to date
            'trailing_12': timedelta(days=365),
            'ttm': timedelta(days=365),  # Trailing twelve months
        }
        
        self.financial_metrics = [
            'revenue', 'sales', 'operating_income', 'net_income', 'profit',
            'margin', 'roe', 'roa', 'debt_ratio', 'current_ratio',
            'earnings', 'dividend', 'buyback', 'capex', 'free_cash_flow'
        ]
    
    def extract_entities(self, query: str) -> FinancialQuery:
        """Extract structured information from natural language query"""
        query_lower = query.lower()
        
        # Extract companies - be more selective to avoid false matches
        companies = []
        
        # Only look for Japanese company codes (4-5 digits) 
        japanese_codes = re.findall(r'\b(\d{4,5})\b', query)
        companies.extend(japanese_codes)
        
        # Only look for stock symbols if they appear to be intentional (e.g., in parentheses or after company names)
        # This avoids matching random English words
        stock_symbols = re.findall(r'\b([A-Z]{2,6})\b(?:\s|$)', query)
        # Only add if it looks like a real stock symbol (all caps, 2-5 chars, not common English words)
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'DOES', 'EACH', 'FROM', 'HAVE', 'LIKE', 'MAKE', 'MOST', 'MOVE', 'MUST', 'NAME', 'OVER', 'SAID', 'SOME', 'TIME', 'VERY', 'WHAT', 'WITH', 'WORD', 'WORK', 'WOULD', 'WRITE', 'YEAR', 'WHERE', 'WHICH', 'THEIR', 'THERE', 'THESE', 'THEY', 'THIS', 'THAT', 'THAN', 'THEN', 'THEM', 'WHEN', 'WILL', 'WITH'}
        for symbol in stock_symbols:
            if len(symbol) <= 5 and symbol.upper() not in common_words:
                companies.append(symbol)
        
        # Extract time period
        time_period = None
        for period_name, delta in self.time_patterns.items():
            if period_name.replace('_', ' ') in query_lower:
                if delta:
                    end_date = date.today()
                    start_date = end_date - delta
                    time_period = (start_date, end_date)
                break
        
        # Extract financial metrics
        metrics = [metric for metric in self.financial_metrics if metric in query_lower]
        
        # Classify query type
        query_type = self._classify_query_type(query_lower)
        
        # Detect comparison type
        comparison_type = None
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'against']):
            comparison_type = 'company_comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'historical']):
            comparison_type = 'time_series'
        elif any(word in query_lower for word in ['increase', 'decrease', 'change', 'revised']):
            comparison_type = 'change_analysis'
        
        return FinancialQuery(
            original_query=query,
            query_type=query_type,
            companies=companies,
            time_period=time_period,
            financial_metrics=metrics,
            comparison_type=comparison_type
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of financial query"""
        if any(word in query for word in ['earnings', '決算', 'quarterly', 'annual']):
            return 'earnings'
        elif any(word in query for word in ['dividend', '配当', 'payout']):
            return 'dividends'
        elif any(word in query for word in ['management', '経営', 'ceo', 'cfo', '取締役']):
            return 'management'
        elif any(word in query for word in ['merger', 'acquisition', 'm&a', '買収', '合併']):
            return 'ma'
        elif any(word in query for word in ['buyback', '自己株式', 'repurchase']):
            return 'buyback'
        elif any(word in query for word in ['margin', 'ratio', 'metric', 'performance']):
            return 'metrics'
        else:
            return 'general'

class DocumentSearchTool(BaseTool):
    """Tool for searching financial documents"""
    
    name: str = "document_search"
    description: str = "Search financial disclosure documents using semantic similarity"
    
    def __init__(self, retrieval_system: FinancialRetrievalSystem):
        super().__init__()
        object.__setattr__(self, '_retrieval_system', retrieval_system)
    
    async def _arun(self, query: str, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Async implementation of document search"""
        return await self._retrieval_system.search(query, filters=filters or {})
    
    def _run(self, query: str, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Sync wrapper for async search"""
        return asyncio.run(self._arun(query, filters))

class FinancialCalculatorTool(BaseTool):
    """Tool for financial calculations and metrics computation"""
    
    name: str = "financial_calculator"
    description: str = "Perform financial calculations and compute metrics"
    
    def _run(self, calculation_type: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Perform financial calculations"""
        try:
            if calculation_type == "growth_rate":
                return self._calculate_growth_rate(data)
            elif calculation_type == "financial_ratios":
                return self._calculate_ratios(data)
            elif calculation_type == "trend_analysis":
                return self._analyze_trends(data)
            elif calculation_type == "comparison_metrics":
                return self._compare_companies(data)
            elif calculation_type == "statistical_summary":
                return self._statistical_summary(data)
            else:
                return {"error": f"Unknown calculation type: {calculation_type}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_growth_rate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various growth rates"""
        results = {}
        
        if "current_value" in data and "previous_value" in data:
            current = data["current_value"]
            previous = data["previous_value"]
            if previous != 0:
                results["growth_rate"] = ((current - previous) / previous) * 100
                results["absolute_change"] = current - previous
        
        if "values" in data and len(data["values"]) >= 2:
            values = data["values"]
            # Compound Annual Growth Rate (CAGR)
            n_periods = len(values) - 1
            if values[0] != 0 and n_periods > 0:
                results["cagr"] = (((values[-1] / values[0]) ** (1/n_periods)) - 1) * 100
        
        return results
    
    def _calculate_ratios(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial ratios"""
        results = {}
        
        # Profitability ratios
        if all(k in data for k in ["net_income", "revenue"]):
            results["net_margin"] = (data["net_income"] / data["revenue"]) * 100
        
        if all(k in data for k in ["operating_income", "revenue"]):
            results["operating_margin"] = (data["operating_income"] / data["revenue"]) * 100
        
        if all(k in data for k in ["net_income", "total_assets"]):
            results["roa"] = (data["net_income"] / data["total_assets"]) * 100
        
        if all(k in data for k in ["net_income", "shareholders_equity"]):
            results["roe"] = (data["net_income"] / data["shareholders_equity"]) * 100
        
        # Liquidity ratios
        if all(k in data for k in ["current_assets", "current_liabilities"]):
            results["current_ratio"] = data["current_assets"] / data["current_liabilities"]
        
        # Leverage ratios
        if all(k in data for k in ["total_debt", "total_assets"]):
            results["debt_ratio"] = (data["total_debt"] / data["total_assets"]) * 100
        
        if all(k in data for k in ["total_debt", "shareholders_equity"]):
            results["debt_to_equity"] = data["total_debt"] / data["shareholders_equity"]
        
        return results
    
    def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze trends in time series data"""
        if "values" not in data or len(data["values"]) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        values = data["values"]
        n = len(values)
        
        # Linear regression for trend
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_mean = mean(values)
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "trend_slope": slope,
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "r_squared": r_squared,
            "volatility": stdev(values) if len(values) > 1 else 0,
            "average": mean(values),
            "median": median(values)
        }
    
    def _compare_companies(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compare metrics across companies"""
        if "companies" not in data:
            return {"error": "No company data provided"}
        
        companies = data["companies"]
        results = {}
        
        # Extract all metrics available
        all_metrics = set()
        for company_data in companies.values():
            all_metrics.update(company_data.keys())
        
        for metric in all_metrics:
            values = []
            company_names = []
            
            for company, metrics in companies.items():
                if metric in metrics:
                    values.append(metrics[metric])
                    company_names.append(company)
            
            if len(values) >= 2:
                results[f"{metric}_average"] = mean(values)
                results[f"{metric}_median"] = median(values)
                results[f"{metric}_std"] = stdev(values)
                results[f"{metric}_min"] = min(values)
                results[f"{metric}_max"] = max(values)
                
                # Find best and worst performers
                max_idx = values.index(max(values))
                min_idx = values.index(min(values))
                results[f"{metric}_best"] = company_names[max_idx]
                results[f"{metric}_worst"] = company_names[min_idx]
        
        return results
    
    def _statistical_summary(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Generate statistical summary of numerical data"""
        if "values" not in data:
            return {"error": "No values provided"}
        
        values = data["values"]
        if not values:
            return {"error": "Empty values list"}
        
        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "std": stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "percentile_25": np.percentile(values, 25),
            "percentile_75": np.percentile(values, 75)
        }

class DataExtractionTool(BaseTool):
    """Tool for extracting structured data from search results"""
    
    name: str = "data_extraction"
    description: str = "Extract structured financial data from document search results"
    
    def _run(self, search_results: List[RetrievalResult], data_type: str) -> Dict[str, Any]:
        """Extract specific types of financial data from search results"""
        extracted_data = {}
        
        for result in search_results:
            company = result.code
            if company not in extracted_data:
                extracted_data[company] = {
                    "company_name": result.name,
                    "documents": [],
                    "extracted_metrics": {}
                }
            
            # Add document info
            doc_info = {
                "title": result.title,
                "date": result.date.isoformat(),
                "similarity_score": result.score,
                "reasoning_context": result.ctx[:500] if result.ctx else ""
            }
            extracted_data[company]["documents"].append(doc_info)
            
            # Extract numerical data from reasoning context
            if data_type == "financial_numbers":
                numbers = self._extract_financial_numbers(result.ctx or "")
                extracted_data[company]["extracted_metrics"].update(numbers)
            elif data_type == "percentages":
                percentages = self._extract_percentages(result.ctx or "")
                extracted_data[company]["extracted_metrics"].update(percentages)
            elif data_type == "dates":
                dates = self._extract_dates(result.ctx or "")
                extracted_data[company]["extracted_metrics"].update(dates)
        
        return extracted_data
    
    def _extract_financial_numbers(self, text: str) -> Dict[str, float]:
        """Extract financial numbers from text"""
        numbers = {}
        
        # Japanese currency patterns
        yen_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*億円',  # Hundred millions
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*兆円',  # Trillions
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*円'     # Regular yen
        ]
        
        multipliers = {'億円': 100_000_000, '兆円': 1_000_000_000_000, '円': 1}
        
        for i, pattern in enumerate(yen_patterns):
            matches = re.findall(pattern, text)
            unit = list(multipliers.keys())[i]
            multiplier = multipliers[unit]
            
            for match in matches:
                value = float(match.replace(',', '')) * multiplier
                numbers[f"amount_{unit}_{match}"] = value
        
        return numbers
    
    def _extract_percentages(self, text: str) -> Dict[str, float]:
        """Extract percentage values from text"""
        percentages = {}
        
        # Percentage patterns
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pct_pattern, text)
        
        for i, match in enumerate(matches):
            percentages[f"percentage_{i}"] = float(match)
        
        return percentages
    
    def _extract_dates(self, text: str) -> Dict[str, str]:
        """Extract dates from text"""
        dates = {}
        
        # Date patterns (Japanese and Western formats)
        date_patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # Japanese format
            r'(\d{4})-(\d{1,2})-(\d{1,2})',      # ISO format
            r'(\d{1,2})/(\d{1,2})/(\d{4})'       # US format
        ]
        
        for i, pattern in enumerate(date_patterns):
            matches = re.findall(pattern, text)
            for j, match in enumerate(matches):
                if len(match) == 3:
                    if i == 0:  # Japanese format
                        date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                    elif i == 1:  # ISO format
                        date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                    else:  # US format
                        date_str = f"{match[2]}-{match[0].zfill(2)}-{match[1].zfill(2)}"
                    
                    dates[f"date_{i}_{j}"] = date_str
        
        return dates

class FinancialAgent:
    """Main agent class that orchestrates the financial analysis workflow"""
    
    def __init__(self, retrieval_system: FinancialRetrievalSystem, llm: ChatOpenAI):
        self.retrieval_system = retrieval_system
        self.llm = llm
        self.entity_extractor = FinancialEntityExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize tools
        self.tools = [
            DocumentSearchTool(retrieval_system),
            FinancialCalculatorTool(),
            DataExtractionTool()
        ]
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("extract_data", self._extract_data)
        workflow.add_node("perform_calculations", self._perform_calculations)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "search_documents")
        workflow.add_edge("search_documents", "extract_data")
        workflow.add_edge("extract_data", "perform_calculations")
        workflow.add_edge("perform_calculations", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def _parse_query(self, state: AgentState) -> AgentState:
        """Parse and understand the user query"""
        query = state["query"]
        
        # Extract entities and structure
        financial_query = self.entity_extractor.extract_entities(query)
        
        # Add reasoning step
        reasoning_step = f"Parsed query: Type={financial_query.query_type}, " \
                        f"Companies={financial_query.companies}, " \
                        f"Metrics={financial_query.financial_metrics}, " \
                        f"Comparison={financial_query.comparison_type}"
        
        state["reasoning_steps"] = [reasoning_step]
        state["metadata"] = {
            "financial_query": financial_query.__dict__,
            "processing_start": datetime.now().isoformat()
        }
        
        return state
    
    async def _search_documents(self, state: AgentState) -> AgentState:
        """Search for relevant documents"""
        query = state["query"]
        financial_query = state["metadata"]["financial_query"]
        
        # Build search filters
        filters = {}
        if financial_query["companies"] and len(financial_query["companies"]) > 0:
            # Only add company filter if we have actual company identifiers
            valid_companies = [c for c in financial_query["companies"] if c and len(c.strip()) > 0]
            if valid_companies:
                filters["company_codes"] = valid_companies
        
        if financial_query["time_period"]:
            filters["date_range"] = financial_query["time_period"]
        
        # Determine search strategy based on query complexity
        force_strategy = None
        if financial_query["comparison_type"] or len(financial_query["financial_metrics"]) > 2:
            force_strategy = "reasoning"
        
        # Perform search
        search_results = await self.retrieval_system.search(
            q=query,
            filters=filters,
            k=15,
            strategy=force_strategy
        )
        
        state["search_results"] = search_results
        state["reasoning_steps"].append(
            f"Found {len(search_results)} relevant documents using "
            f"{'reasoning' if force_strategy == 'reasoning' else 'semantic'} search"
        )
        
        return state
    
    async def _extract_data(self, state: AgentState) -> AgentState:
        """Extract structured data from search results"""
        search_results = state["search_results"]
        financial_query = state["metadata"]["financial_query"]
        
        # Determine what type of data to extract
        if financial_query["query_type"] in ["earnings", "metrics"]:
            data_type = "financial_numbers"
        elif "percentage" in state["query"].lower() or "%" in state["query"]:
            data_type = "percentages"
        else:
            data_type = "financial_numbers"  # Default
        
        # Extract data using the tool
        extractor = DataExtractionTool()
        extracted_data = extractor._run(search_results, data_type)
        
        state["financial_data"] = extracted_data
        state["reasoning_steps"].append(
            f"Extracted {data_type} from {len(extracted_data)} companies"
        )
        
        return state
    
    async def _perform_calculations(self, state: AgentState) -> AgentState:
        """Perform financial calculations and analysis"""
        financial_data = state["financial_data"]
        financial_query = state["metadata"]["financial_query"]
        
        calculator = FinancialCalculatorTool()
        calculations = {}
        
        # Determine calculation type based on query
        if financial_query["comparison_type"] == "company_comparison":
            calc_data = {"companies": {}}
            for company, data in financial_data.items():
                calc_data["companies"][company] = data["extracted_metrics"]
            
            calculations = calculator._run("comparison_metrics", calc_data)
        
        elif financial_query["comparison_type"] == "change_analysis":
            # Look for trend analysis opportunities
            for company, data in financial_data.items():
                metrics = data["extracted_metrics"]
                if len(metrics) >= 2:
                    values = list(metrics.values())
                    if all(isinstance(v, (int, float)) for v in values):
                        trend_calc = calculator._run("trend_analysis", {"values": values})
                        calculations[f"{company}_trend"] = trend_calc
        
        else:
            # General statistical analysis
            all_values = []
            for company_data in financial_data.values():
                for value in company_data["extracted_metrics"].values():
                    if isinstance(value, (int, float)):
                        all_values.append(value)
            
            if all_values:
                calculations = calculator._run("statistical_summary", {"values": all_values})
        
        state["calculations"] = calculations
        state["reasoning_steps"].append(
            f"Performed {len(calculations)} calculations and analyses"
        )
        
        return state
    
    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response"""
        query = state["query"]
        reasoning_steps = state["reasoning_steps"]
        search_results = state["search_results"]
        financial_data = state["financial_data"]
        calculations = state["calculations"]
        
        # Prepare context for LLM
        context = {
            "original_query": query,
            "reasoning_steps": reasoning_steps,
            "num_documents_found": len(search_results),
            "companies_analyzed": list(financial_data.keys()),
            "key_calculations": calculations,
            "detailed_results": [
                {
                    "company_code": r.code,
                    "company_name": r.name,
                    "title": r.title,
                    "date": r.date.isoformat(),
                    "score": r.score,
                    "context": r.ctx[:400] + "..." if len(r.ctx or "") > 400 else r.ctx or ""  # Increased context length
                }
                for r in search_results
            ],
            "company_summary": [
                {
                    "code": r.code,
                    "name": r.name,
                    "relevant_docs": len([x for x in search_results if x.code == r.code])
                }
                for r in search_results
            ]
        }
        
        # Generate response using LLM
        prompt = f"""
        Based on the comprehensive financial analysis performed, provide a detailed response to the user's query.
        
        User Query: {query}
        
        Analysis Context: {json.dumps(context, indent=2, default=str)}
        
        IMPORTANT INSTRUCTIONS:
        1. Use ONLY the company information provided in the analysis context
        2. List ALL companies found in the search results, not just a subset
        3. Use the exact company names and codes provided in the detailed_results
        4. Do not invent or assume company names not explicitly provided
        
        Please provide:
        1. A direct answer listing ALL companies that match the query criteria
        2. Key insights from the retrieved documents
        3. Relevant financial metrics and trends from the analysis
        4. Supporting evidence from the specific documents found
        5. Any limitations or caveats about the analysis
        
        Format the response professionally and ensure all {len(search_results)} companies found are properly represented in your answer.
        """
        
        response = await self.llm.ainvoke(prompt)
        
        state["response"] = response.content
        state["metadata"]["processing_end"] = datetime.now().isoformat()
        
        return state
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a financial query and return comprehensive results"""
        try:
            # Initialize state
            initial_state = AgentState(
                query=query,
                reasoning_steps=[],
                search_results=[],
                financial_data={},
                calculations={},
                response="",
                metadata={}
            )
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "response": final_state["response"],
                "reasoning_steps": final_state["reasoning_steps"],
                "documents_found": len(final_state["search_results"]),
                "companies_analyzed": list(final_state["financial_data"].keys()),
                "calculations": final_state["calculations"],
                "metadata": final_state["metadata"],
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "error": str(e),
                "success": False
            }

# Usage example and test framework
async def main():
    """Example usage of the financial agent"""
    
    # Initialize retrieval system
    # config = RetrievalConfig(
    #     postgres_url="postgresql://user:pass@localhost/financial_db",
    #     redis_url="redis://localhost:6379/0"
    # )
    
    import asyncio, sys, os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    
    cfg = RetrievalConfig(
        pg_dsn=os.getenv("PG_DSN", "postgresql://user:pass@localhost/financial_db"),
        redis_url=os.getenv("REDIS_URL","redis://localhost:6379/0")
    )
    
    retrieval_system = FinancialRetrievalSystem(cfg)
    await retrieval_system.init()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=2000
    )
    
    # Create agent
    agent = FinancialAgent(retrieval_system, llm)
    
    # Test queries
    test_queries = [
        "Which Japanese companies raised dividends last quarter?",
        "What companies revised their earnings guidance upward recently?",
        "株主還元方針の変更",  # Japanese query about shareholder return policy changes
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = await agent.process_query(query)
        
        if result["success"]:
            print(f"Response: {result['response']}")
            print(f"\nDocuments analyzed: {result['documents_found']}")
            print(f"Companies: {', '.join(result['companies_analyzed'])}")
            print(f"Reasoning steps: {len(result['reasoning_steps'])}")
        else:
            print(f"Error: {result['error']}")
    
    await retrieval_system.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
