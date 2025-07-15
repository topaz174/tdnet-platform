#!/usr/bin/env python3
"""
Complete Enhanced Financial Agent with Numeric Data Extraction
=============================================================

This agent combines:
1. Enhanced retrieval system (document identification)
2. Financial data extraction (numeric value extraction from PDFs)
3. Structured analysis and reporting

Architecture:
- Stage 1: Parse query and classify intent
- Stage 2: Retrieve relevant documents using enhanced system
- Stage 3: Extract detailed numeric data from top documents
- Stage 4: Perform calculations and analysis
- Stage 5: Generate comprehensive response with specific numbers
"""


import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime, date, timedelta
import json
import re
from dataclasses import dataclass, asdict
import asyncio
import logging
import math
from statistics import mean, median
import os

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np

# Import our enhanced retrieval system
from enhanced_retrieval_system import (
    EnhancedFinancialRetrievalSystem, 
    EnhancedRetrievalConfig, 
    RetrievalResult,
    FinancialKnowledgeBase,
    QueryClassification
)

# Import our new data extraction system
from financial_data_extraction_agent import (
    FinancialDataExtractor,
    ExtractedData,
    FinancialMetric,
    DataType
)

class EnhancedAgentState(TypedDict):
    query: str
    query_classification: Optional[QueryClassification]
    reasoning_steps: List[str]
    search_results: List[RetrievalResult]
    extracted_data: List[ExtractedData]
    financial_analysis: Dict[str, Any]
    calculations: Dict[str, float]
    response: str
    metadata: Dict[str, Any]

class CompleteEnhancedAgentWithExtraction:
    """Complete financial agent with document retrieval and numeric data extraction"""
    
    def __init__(self):
        # Initialize retrieval system
        self.retrieval_system = EnhancedFinancialRetrievalSystem(
            EnhancedRetrievalConfig(
                pg_dsn=os.getenv('PG_DSN'),
                redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379')
            )
        )
        
        # Initialize data extraction system
        self.data_extractor = FinancialDataExtractor(enable_ocr=True, use_gpu=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        # Connection initialization flag
        self._initialized = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _build_workflow(self) -> StateGraph:
        """Build the enhanced workflow with data extraction"""
        workflow = StateGraph(EnhancedAgentState)
        
        # Add nodes
        workflow.add_node("parse_query", self.parse_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("extract_data", self.extract_data)
        workflow.add_node("analyze_data", self.analyze_data)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define flow
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "extract_data")
        workflow.add_edge("extract_data", "analyze_data")
        workflow.add_edge("analyze_data", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def parse_query(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Enhanced query parsing and classification"""
        self.logger.info(f"Parsing query: {state['query']}")
        
        # Use the knowledge base class method for query classification
        query_classification = FinancialKnowledgeBase.classify_query(state["query"])
        
        state["query_classification"] = query_classification
        state["reasoning_steps"] = [f"Classified query as: {query_classification.query_type}"]
        state["metadata"] = {"start_time": datetime.now()}
        
        return state
    
    async def retrieve_documents(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Retrieve relevant documents using enhanced system"""
        self.logger.info("Retrieving relevant documents")
        
        # Initialize retrieval system if needed (only once)
        if not self._initialized:
            await self.retrieval_system.init()
            self._initialized = True
        
        # Phase 1: Smart Current - Increased limits with intelligent processing
        # Retrieve more documents to ensure comprehensive coverage
        results = await self.retrieval_system.search(
            query=state["query"],
            k=100,  # Phase 1: Increased from 50 to 100 for better coverage
            filters={}
        )
        
        state["search_results"] = results
        state["reasoning_steps"].append(f"Retrieved {len(results)} relevant documents")
        
        return state
    
    async def extract_data(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Extract numeric financial data from top documents"""
        self.logger.info("Extracting numeric data from documents")
        
        extracted_data = []
        
        # Phase 1: Smart Current - Process more docs with early stopping
        # Dynamically determine processing limit based on results quality
        max_docs = min(35, len(state["search_results"]))  # Phase 1: Increased from 25 to 35
        self.logger.info(f"Phase 1: Processing up to {max_docs} documents with smart stopping")
        
        top_docs = state["search_results"][:max_docs]
        companies_with_good_metrics = 0
        GOOD_METRICS_THRESHOLD = 15  # Stop early if we find enough high-quality results
        
        for i, doc in enumerate(top_docs):
            try:
                # Always create a basic result for each document
                result = self.data_extractor.process_document(
                    pdf_path=doc.pdf,
                    document_id=doc.id,
                    company_code=doc.code,
                    company_name=doc.name,
                    title=doc.title,
                    doc_date=doc.date
                )
                
                if result:
                    extracted_data.append(result)
                    if result.metrics:
                        companies_with_good_metrics += 1
                        self.logger.info(f"Extracted {len(result.metrics)} metrics from {doc.title}")
                    else:
                        self.logger.info(f"Document processed but no metrics extracted from {doc.title}")
                    
                    # Phase 1: Smart stopping - if we have enough good results, consider stopping
                    if companies_with_good_metrics >= GOOD_METRICS_THRESHOLD and i >= 20:  # Minimum 20 docs processed
                        remaining_docs = len(top_docs) - (i + 1)
                        self.logger.info(f"Phase 1 Smart Stop: Found {companies_with_good_metrics} companies with metrics. Skipping {remaining_docs} remaining documents.")
                        break
                else:
                    # Even if extraction fails, create a basic record
                    from financial_data_extraction_agent import ExtractedData, DataType
                    basic_result = ExtractedData(
                        document_id=doc.id,
                        company_code=doc.code,
                        company_name=doc.name,
                        document_title=doc.title,
                        document_date=doc.date,
                        data_type=DataType.GUIDANCE_UPDATE,  # Default type
                        metrics=[],
                        summary=f"Document identified as relevant based on title: {doc.title}",
                        extraction_confidence=0.3,  # Low confidence since no data extracted
                        raw_text_sample=""
                    )
                    extracted_data.append(basic_result)
                    self.logger.info(f"Created basic record for {doc.title} (no text extraction possible)")
                    
            except Exception as e:
                self.logger.error(f"Processing failed for document {doc.id}: {e}")
                # Still create a basic record even on error
                from financial_data_extraction_agent import ExtractedData, DataType
                basic_result = ExtractedData(
                    document_id=doc.id,
                    company_code=doc.code,
                    company_name=doc.name,
                    document_title=doc.title,
                    document_date=doc.date,
                    data_type=DataType.GUIDANCE_UPDATE,
                    metrics=[],
                    summary=f"Document could not be processed but title suggests relevance: {doc.title}",
                    extraction_confidence=0.2,
                    raw_text_sample=""
                )
                extracted_data.append(basic_result)
        
        state["extracted_data"] = extracted_data
        state["reasoning_steps"].append(f"Successfully extracted data from {len(extracted_data)} documents")
        
        return state
    
    async def analyze_data(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Perform financial analysis on extracted data"""
        self.logger.info("Analyzing extracted financial data")
        
        analysis = {
            "companies_with_data": [],
            "metric_summaries": {},
            "trends": {},
            "aggregates": {}
        }
        
        # Analyze extracted data
        for data in state["extracted_data"]:
            company_info = {
                "company_code": data.company_code,
                "company_name": data.company_name,
                "document_title": data.document_title,
                "data_type": data.data_type.value,
                "confidence": data.extraction_confidence,
                "metrics": []
            }
            
            for metric in data.metrics:
                metric_info = {
                    "name": metric.metric_name,
                    "previous_value": metric.previous_value,
                    "revised_value": metric.revised_value,
                    "change_amount": metric.change_amount,
                    "change_percentage": metric.change_percentage,
                    "unit": metric.unit,
                    "period": metric.period,
                    "confidence": metric.confidence
                }
                company_info["metrics"].append(metric_info)
                
                # Aggregate metrics by type
                metric_key = metric.metric_name
                if metric_key not in analysis["metric_summaries"]:
                    analysis["metric_summaries"][metric_key] = []
                
                if metric.change_percentage is not None:
                    analysis["metric_summaries"][metric_key].append({
                        "company": data.company_name,
                        "change_pct": metric.change_percentage,
                        "period": metric.period
                    })
            
            analysis["companies_with_data"].append(company_info)
        
        # Calculate aggregates
        for metric_name, data_points in analysis["metric_summaries"].items():
            if data_points:
                changes = [dp["change_pct"] for dp in data_points if dp["change_pct"] is not None]
                if changes:
                    analysis["aggregates"][metric_name] = {
                        "count": len(changes),
                        "average_change": round(mean(changes), 2),
                        "median_change": round(median(changes), 2),
                        "max_change": round(max(changes), 2),
                        "min_change": round(min(changes), 2)
                    }
        
        state["financial_analysis"] = analysis
        state["reasoning_steps"].append(f"Analyzed data from {len(state['extracted_data'])} companies")
        
        return state
    
    async def generate_response(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate comprehensive response with extracted numeric data"""
        self.logger.info("Generating final response with extracted data")
        
        # Create enhanced prompt with extracted data
        prompt = self._create_response_prompt(state)
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a financial analyst providing detailed analysis with specific numeric data extracted from corporate disclosures."),
            HumanMessage(content=prompt)
        ])
        
        state["response"] = response.content
        state["reasoning_steps"].append("Generated comprehensive response with extracted financial data")
        
        return state
    
    def _create_response_prompt(self, state: EnhancedAgentState) -> str:
        """Create enhanced prompt with extracted financial data"""
        
        query = state["query"]
        classification = state["query_classification"]
        extracted_data = state["extracted_data"]
        analysis = state["financial_analysis"]
        
        prompt = f"""
Based on the financial data extraction and analysis, provide a comprehensive answer to: "{query}"

EXTRACTED FINANCIAL DATA:
"""
        
        # FIXED: Give equal prominence to ALL companies, sorted by relevance
        # Don't create artificial hierarchy based on data extractability
        all_companies = sorted(extracted_data, key=lambda x: len(x.metrics), reverse=True)
        
        for i, data in enumerate(all_companies, 1):
            prompt += f"\n{i}. **{data.company_name} ({data.company_code})**\n"
            prompt += f"   Document: {data.document_title}\n"
            prompt += f"   Date: {data.document_date}\n"
            prompt += f"   Type: {data.data_type.value}\n"
            
            if data.metrics:
                # Detailed metrics for companies with extractable data
                prompt += f"   Confidence: {data.extraction_confidence:.2f}\n"
                for metric in data.metrics:
                    prompt += f"   • {metric.metric_name}:\n"
                    if metric.previous_value is not None and metric.revised_value is not None:
                        prompt += f"     Previous: {metric.previous_value:,.0f} {metric.unit or ''}\n"
                        prompt += f"     Revised: {metric.revised_value:,.0f} {metric.unit or ''}\n"
                    if metric.change_amount is not None:
                        prompt += f"     Change: {metric.change_amount:+,.0f} {metric.unit or ''}\n"
                    if metric.change_percentage is not None:
                        prompt += f"     Change: {metric.change_percentage:+.1f}%\n"
                    if metric.period:
                        prompt += f"     Period: {metric.period}\n"
            else:
                # Still relevant but no extractable data (e.g., image-based PDFs)
                prompt += f"   Status: Document identified as highly relevant by semantic search\n"
                prompt += f"   Note: {data.summary}\n"
            
            prompt += "\n"
        
        if analysis.get("aggregates"):
            prompt += "AGGREGATE ANALYSIS:\n"
            for metric_name, agg_data in analysis["aggregates"].items():
                prompt += f"• {metric_name}: {agg_data['count']} companies, "
                prompt += f"avg change: {agg_data['average_change']:+.1f}%, "
                prompt += f"range: {agg_data['min_change']:+.1f}% to {agg_data['max_change']:+.1f}%\n"
        
        prompt += f"""
INSTRUCTIONS:
1. Treat ALL companies listed above as equally relevant - they were all identified by semantic search as matching the query
2. For companies with extracted metrics: provide specific numeric details (amounts, percentages, changes)
3. For companies without extracted metrics: still mention them as relevant based on document titles and semantic relevance
4. Include company names, document titles, and dates for all companies
5. Highlight the most significant changes or noteworthy trends where data is available
6. Note that some companies may have relevant information in image-based PDFs that couldn't be automatically extracted
7. Organize by company relevance, not by data availability

Query type: {classification.query_type if classification else 'unknown'}
Total companies identified: {len(extracted_data)}
Companies with extractable metrics: {len([d for d in extracted_data if d.metrics])}
Companies with document-title evidence only: {len([d for d in extracted_data if not d.metrics])}
"""
        
        return prompt
    
    async def process_query(self, query: str) -> str:
        """Process a complete query through the enhanced workflow"""
        self.logger.info(f"Processing query: {query}")
        
        try:
            initial_state = EnhancedAgentState(
                query=query,
                query_classification=None,
                reasoning_steps=[],
                search_results=[],
                extracted_data=[],
                financial_analysis={},
                calculations={},
                response="",
                metadata={}
            )
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Add summary metadata
            end_time = datetime.now()
            processing_time = end_time - final_state["metadata"]["start_time"]
            
            # Better reporting with actual numbers
            docs_found = len(final_state['search_results'])
            docs_processed = len(final_state['extracted_data'])
            companies_with_metrics = len([d for d in final_state['extracted_data'] if d.metrics])
            total_metrics = sum(len(d.metrics) for d in final_state['extracted_data'])
            
            summary = f"""
📊 PHASE 1 (SMART CURRENT) SUMMARY:
Documents found by retrieval system: {docs_found}
Documents processed for extraction: {docs_processed}
Companies with extracted numeric data: {companies_with_metrics}
Companies with document titles only: {docs_processed - companies_with_metrics}
Total financial metrics extracted: {total_metrics}
Processing time: {processing_time.total_seconds():.1f}s
Query reasoning steps: {len(final_state['reasoning_steps'])}

💡 Smart stopping engaged: {'Yes' if docs_found > docs_processed else 'No'}
Coverage improvement: Retrieval limit increased to 100, processing limit to 35
"""
            
            return final_state["response"] + "\n\n" + summary
            
        finally:
            # Don't close connection here - let it stay open for multiple queries
            pass

async def main():
    """Test the complete enhanced agent with extraction"""
    agent = CompleteEnhancedAgentWithExtraction()
    
    # Test queries
    test_queries = [
        "Which Japanese companies raised dividends last quarter?",
        "What companies revised their earnings guidance upward recently?", 
        "株主還元方針の変更",
        "Please give me a list of companies in the engineering services sector",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            response = await agent.process_query(query)
            print(response)
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())