# Hybrid Agent Design Patterns for Financial Intelligence
# This module demonstrates advanced patterns for agents that handle both structured and unstructured data

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.schema import BaseRetriever
from langchain.sql_database import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

# ===== CORE DESIGN PATTERNS =====

class QueryType(Enum):
    STRUCTURED = "structured"      # Pure SQL queries
    UNSTRUCTURED = "unstructured"  # Pure RAG queries
    HYBRID = "hybrid"              # Requires both approaches
    CONTEXTUAL = "contextual"      # Structured data + narrative context

@dataclass
class QueryAnalysis:
    query_type: QueryType
    structured_elements: List[str]
    unstructured_elements: List[str]
    confidence: float
    suggested_approach: str

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
        result = self.llm(self.classification_prompt.format(query=query))
        parsed = json.loads(result)
        return QueryAnalysis(**parsed)

# ===== PATTERN 1: DUAL-SOURCE AGENT =====

class DualSourceFinancialAgent:
    """Agent that seamlessly switches between structured DB and RAG based on query needs"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.classifier = QueryClassifier(llm)
        
        # Define tools for each data source
        self.structured_tools = self._create_structured_tools()
        self.unstructured_tools = self._create_unstructured_tools()
        self.hybrid_tools = self._create_hybrid_tools()
    
    def _create_structured_tools(self) -> List[Tool]:
        """Tools for querying structured financial data"""
        return [
            Tool(
                name="get_financial_metrics",
                description="Get specific financial metrics for companies and time periods",
                func=self._get_financial_metrics
            ),
            Tool(
                name="calculate_financial_ratios",
                description="Calculate financial ratios like P/E, ROE, debt-to-equity",
                func=self._calculate_ratios
            ),
            Tool(
                name="compare_companies_metrics",
                description="Compare financial metrics between multiple companies",
                func=self._compare_companies
            ),
            Tool(
                name="time_series_analysis",
                description="Analyze trends in financial metrics over time",
                func=self._time_series_analysis
            )
        ]
    
    def _create_unstructured_tools(self) -> List[Tool]:
        """Tools for querying unstructured narrative content"""
        return [
            Tool(
                name="search_management_commentary",
                description="Search for management commentary and strategic insights",
                func=self._search_commentary
            ),
            Tool(
                name="analyze_risk_factors",
                description="Extract and analyze risk factors from disclosures",
                func=self._analyze_risks
            ),
            Tool(
                name="find_business_events",
                description="Find specific business events, announcements, or changes",
                func=self._find_events
            ),
            Tool(
                name="semantic_document_search",
                description="Perform semantic search across document content",
                func=self._semantic_search
            )
        ]
    
    def _create_hybrid_tools(self) -> List[Tool]:
        """Tools that combine both structured and unstructured data"""
        return [
            Tool(
                name="explain_financial_performance",
                description="Explain financial performance using both metrics and narrative",
                func=self._explain_performance
            ),
            Tool(
                name="contextualize_metrics",
                description="Provide context for financial metrics using management commentary",
                func=self._contextualize_metrics
            ),
            Tool(
                name="validate_narrative_claims",
                description="Validate narrative claims against actual financial data",
                func=self._validate_claims
            )
        ]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main entry point - routes query based on classification"""
        analysis = self.classifier.classify_query(query)
        
        if analysis.query_type == QueryType.STRUCTURED:
            return self._handle_structured_query(query, analysis)
        elif analysis.query_type == QueryType.UNSTRUCTURED:
            return self._handle_unstructured_query(query, analysis)
        elif analysis.query_type == QueryType.HYBRID:
            return self._handle_hybrid_query(query, analysis)
        else:  # CONTEXTUAL
            return self._handle_contextual_query(query, analysis)
    
    def _handle_structured_query(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle purely numerical/factual queries"""
        # Example: "What was Toyota's revenue in 2023?"
        metrics = self._get_financial_metrics(
            ticker="7203",  # Toyota's ticker
            metrics=analysis.structured_elements,
            year=2023
        )
        
        return {
            "type": "structured",
            "data": metrics,
            "visualization": self._suggest_visualization(metrics),
            "confidence": analysis.confidence
        }
    
    def _handle_unstructured_query(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle narrative/contextual queries"""
        # Example: "What are Toyota's main strategic priorities?"
        relevant_docs = self.vector_retriever.get_relevant_documents(query)
        
        # Use LLM to synthesize response from retrieved documents
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""
            Based on these corporate disclosure documents, answer the question:
            Question: {query}
            
            Documents:
            {documents}
            
            Provide a comprehensive answer based on the document content.
            """
        )
        
        response = self.llm(synthesis_prompt.format(
            query=query,
            documents="\n".join([doc.page_content for doc in relevant_docs])
        ))
        
        return {
            "type": "unstructured",
            "answer": response,
            "sources": [doc.metadata for doc in relevant_docs],
            "confidence": analysis.confidence
        }
    
    def _handle_hybrid_query(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle queries requiring both structured and unstructured data"""
        # Example: "How did Toyota's profitability change and what drove these changes?"
        
        # First, get structured data
        structured_data = self._get_financial_metrics(
            ticker="7203",
            metrics=analysis.structured_elements,
            periods=5  # Last 5 periods
        )
        
        # Then, get narrative context
        context_query = f"management commentary on {' '.join(analysis.unstructured_elements)}"
        relevant_docs = self.vector_retriever.get_relevant_documents(context_query)
        
        # Combine both sources
        combined_prompt = PromptTemplate(
            input_variables=["query", "structured_data", "narrative_context"],
            template="""
            Answer this question using both quantitative data and narrative context:
            Question: {query}
            
            Financial Data:
            {structured_data}
            
            Management Commentary:
            {narrative_context}
            
            Provide a comprehensive analysis that combines both data sources.
            """
        )
        
        response = self.llm(combined_prompt.format(
            query=query,
            structured_data=json.dumps(structured_data, indent=2),
            narrative_context="\n".join([doc.page_content for doc in relevant_docs])
        ))
        
        return {
            "type": "hybrid",
            "analysis": response,
            "structured_data": structured_data,
            "narrative_sources": [doc.metadata for doc in relevant_docs],
            "confidence": analysis.confidence
        }
    
    def _handle_contextual_query(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle queries that need structured data first, then context"""
        # Example: "Show me Sony's debt levels and explain the company's debt strategy"
        
        # Get the numbers first
        debt_metrics = self._get_financial_metrics(
            ticker="6758",  # Sony
            metrics=["total_debt", "debt_to_equity", "interest_coverage"],
            periods=3
        )
        
        # Then find context about debt strategy
        strategy_docs = self.vector_retriever.get_relevant_documents(
            "debt strategy financing capital structure"
        )
        
        return {
            "type": "contextual",
            "primary_data": debt_metrics,
            "context": strategy_docs,
            "visualization": self._suggest_visualization(debt_metrics),
            "confidence": analysis.confidence
        }
    
    # Tool implementations
    def _get_financial_metrics(self, ticker: str = None, metrics: List[str] = None, 
                             year: int = None, periods: int = 1) -> Dict[str, Any]:
        """Retrieve financial metrics from structured database"""
        query = """
        SELECT 
            fm.ticker,
            fm.period_end_date,
            fm.metric_name,
            fm.metric_value_jpy,
            d.company_name
        FROM financial_metrics fm
        JOIN documents d ON fm.document_id = d.id
        WHERE 1=1
        """
        
        params = []
        if ticker:
            query += " AND fm.ticker = %s"
            params.append(ticker)
        if metrics:
            placeholders = ",".join(["%s"] * len(metrics))
            query += f" AND fm.metric_name IN ({placeholders})"
            params.extend(metrics)
        if year:
            query += " AND EXTRACT(YEAR FROM fm.period_end_date) = %s"
            params.append(year)
        
        query += " ORDER BY fm.period_end_date DESC"
        if periods:
            query += f" LIMIT {periods * len(metrics or [1])}"
        
        # Execute query and return structured data
        return self.sql_db.run(query, params)
    
    def _search_commentary(self, query: str, company: str = None) -> List[Dict]:
        """Search for management commentary using vector similarity"""
        search_query = query
        if company:
            search_query = f"{company} {query}"
        
        docs = self.vector_retriever.get_relevant_documents(search_query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    def _explain_performance(self, ticker: str, metric: str, period: str) -> Dict[str, Any]:
        """Explain financial performance by combining metrics with narrative"""
        # Get the metrics
        metrics = self._get_financial_metrics(ticker=ticker, metrics=[metric])
        
        # Get explanatory context
        context_query = f"{ticker} {metric} performance explanation management commentary"
        context = self._search_commentary(context_query)
        
        # Synthesize explanation
        explanation_prompt = PromptTemplate(
            input_variables=["ticker", "metric", "data", "context"],
            template="""
            Explain the {metric} performance for company {ticker} using this data:
            
            Financial Data:
            {data}
            
            Management Commentary:
            {context}
            
            Provide a comprehensive explanation of the performance.
            """
        )
        
        explanation = self.llm(explanation_prompt.format(
            ticker=ticker,
            metric=metric,
            data=json.dumps(metrics, indent=2),
            context=json.dumps(context, indent=2)
        ))
        
        return {
            "explanation": explanation,
            "supporting_data": metrics,
            "context_sources": context
        }

# ===== PATTERN 2: COORDINATED MULTI-AGENT SYSTEM =====

class FinancialIntelligenceOrchestrator:
    """Orchestrates multiple specialized agents for complex financial analysis"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
        
        # Initialize specialized agents
        self.structured_agent = StructuredDataAgent(sql_db, llm)
        self.narrative_agent = NarrativeAnalysisAgent(vector_retriever, llm)
        self.synthesis_agent = SynthesisAgent(llm)
        self.validation_agent = ValidationAgent(sql_db, vector_retriever, llm)
    
    def complex_analysis(self, query: str) -> Dict[str, Any]:
        """Coordinate multiple agents for complex financial analysis"""
        
        # Step 1: Break down the query
        task_breakdown = self._decompose_query(query)
        
        # Step 2: Execute tasks in parallel/sequence
        results = {}
        for task in task_breakdown:
            if task["type"] == "structured":
                results[task["id"]] = self.structured_agent.execute(task["query"])
            elif task["type"] == "narrative":
                results[task["id"]] = self.narrative_agent.execute(task["query"])
        
        # Step 3: Synthesize results
        synthesis = self.synthesis_agent.combine_results(query, results)
        
        # Step 4: Validate and fact-check
        validated_result = self.validation_agent.validate(synthesis)
        
        return validated_result
    
    def _decompose_query(self, query: str) -> List[Dict]:
        """Break complex queries into subtasks for different agents"""
        decomposition_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Break down this complex financial query into subtasks:
            Query: {query}
            
            Create a list of subtasks, each with:
            - id: unique identifier
            - type: "structured" or "narrative"
            - query: specific question for that subtask
            - dependencies: list of other task IDs this depends on
            
            Return as JSON array.
            """
        )
        
        result = self.llm(decomposition_prompt.format(query=query))
        return json.loads(result)

class StructuredDataAgent:
    """Specialized agent for structured financial data queries"""
    
    def __init__(self, sql_db: SQLDatabase, llm):
        self.sql_db = sql_db
        self.llm = llm
        self.query_templates = self._load_query_templates()
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute structured data query"""
        # Convert natural language to SQL
        sql_query = self._generate_sql(query)
        
        # Execute query
        result = self.sql_db.run(sql_query)
        
        # Format result
        return {
            "type": "structured",
            "query": query,
            "sql": sql_query,
            "data": result,
            "summary": self._summarize_data(result)
        }
    
    def _generate_sql(self, query: str) -> str:
        """Convert natural language query to SQL"""
        sql_prompt = PromptTemplate(
            input_variables=["query", "schema"],
            template="""
            Convert this natural language query to SQL:
            Query: {query}
            
            Database Schema:
            {schema}
            
            Return only the SQL query.
            """
        )
        
        schema = self._get_schema_info()
        return self.llm(sql_prompt.format(query=query, schema=schema))
    
    def _get_schema_info(self) -> str:
        """Get relevant schema information"""
        return """
        financial_metrics (ticker, period_end_date, metric_name, metric_value_jpy)
        financial_statements (document_id, statement_type, period_end_date, data)
        documents (ticker, company_name, release_datetime, category)
        """
    
    def _summarize_data(self, data: Any) -> str:
        """Generate human-readable summary of data"""
        if isinstance(data, pd.DataFrame):
            return f"Retrieved {len(data)} records with {len(data.columns)} metrics"
        return f"Retrieved {len(data) if hasattr(data, '__len__') else 1} data points"

class NarrativeAnalysisAgent:
    """Specialized agent for narrative/unstructured content analysis"""
    
    def __init__(self, vector_retriever: BaseRetriever, llm):
        self.vector_retriever = vector_retriever
        self.llm = llm
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute narrative analysis query"""
        # Retrieve relevant documents
        docs = self.vector_retriever.get_relevant_documents(query)
        
        # Analyze and synthesize
        analysis = self._analyze_narrative(query, docs)
        
        return {
            "type": "narrative",
            "query": query,
            "analysis": analysis,
            "sources": [doc.metadata for doc in docs],
            "confidence": self._calculate_confidence(docs)
        }
    
    def _analyze_narrative(self, query: str, docs: List) -> str:
        """Analyze narrative content to answer query"""
        analysis_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""
            Analyze these corporate documents to answer the question:
            Question: {query}
            
            Documents:
            {documents}
            
            Provide a detailed analysis with specific examples from the documents.
            """
        )
        
        doc_content = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                  for i, doc in enumerate(docs)])
        
        return self.llm(analysis_prompt.format(query=query, documents=doc_content))
    
    def _calculate_confidence(self, docs: List) -> float:
        """Calculate confidence score based on document relevance"""
        if not docs:
            return 0.0
        
        # Simple confidence calculation based on number and recency of documents
        confidence = min(len(docs) / 10.0, 1.0)  # More docs = higher confidence
        return confidence

class SynthesisAgent:
    """Agent responsible for combining results from multiple sources"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def combine_results(self, original_query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple agents into coherent response"""
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "results"],
            template="""
            Synthesize these analysis results to answer the original question:
            Original Question: {query}
            
            Analysis Results:
            {results}
            
            Provide a comprehensive, coherent answer that integrates all the information.
            Highlight any contradictions or uncertainties.
            """
        )
        
        formatted_results = json.dumps(results, indent=2, default=str)
        synthesis = self.llm(synthesis_prompt.format(
            query=original_query,
            results=formatted_results
        ))
        
        return {
            "original_query": original_query,
            "synthesis": synthesis,
            "component_results": results,
            "methodology": "multi-agent_coordination"
        }

class ValidationAgent:
    """Agent that validates and fact-checks synthesized results"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
    
    def validate(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the synthesized result for accuracy and consistency"""
        
        # Extract claims that can be fact-checked
        claims = self._extract_verifiable_claims(synthesis_result["synthesis"])
        
        # Validate each claim
        validation_results = []
        for claim in claims:
            validation = self._validate_claim(claim)
            validation_results.append(validation)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(validation_results)
        
        return {
            **synthesis_result,
            "validation": {
                "claims_checked": validation_results,
                "overall_confidence": overall_confidence,
                "validation_notes": self._generate_validation_notes(validation_results)
            }
        }
    
    def _extract_verifiable_claims(self, synthesis: str) -> List[str]:
        """Extract specific claims that can be fact-checked"""
        extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract specific, verifiable claims from this text:
            Text: {text}
            
            Return a JSON list of claims that can be fact-checked against data.
            Focus on numerical claims, dates, and specific business facts.
            """
        )
        
        result = self.llm(extraction_prompt.format(text=synthesis))
        return json.loads(result)
    
    def _validate_claim(self, claim: str) -> Dict[str, Any]:
        """Validate a specific claim against available data"""
        # This would implement specific validation logic
        # For now, return a placeholder structure
        return {
            "claim": claim,
            "validation_status": "verified",  # or "disputed", "uncertain"
            "confidence": 0.85,
            "evidence": "Supporting evidence from data sources"
        }
    
    def _calculate_overall_confidence(self, validation_results: List[Dict]) -> float:
        """Calculate overall confidence based on individual claim validations"""
        if not validation_results:
            return 0.5
        
        confidences = [result["confidence"] for result in validation_results]
        return sum(confidences) / len(confidences)
    
    def _generate_validation_notes(self, validation_results: List[Dict]) -> str:
        """Generate human-readable validation notes"""
        verified_count = sum(1 for r in validation_results if r["validation_status"] == "verified")
        total_count = len(validation_results)
        
        return f"Validated {verified_count}/{total_count} verifiable claims in the analysis."

# ===== PATTERN 3: ADAPTIVE CONTEXT AGENT =====

class AdaptiveContextAgent:
    """Agent that dynamically adjusts its approach based on available data and context"""
    
    def __init__(self, sql_db: SQLDatabase, vector_retriever: BaseRetriever, llm):
        self.sql_db = sql_db
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.context_memory = {}  # Store conversation context
    
    def process_with_context(self, query: str, session_id: str) -> Dict[str, Any]:
        """Process query with adaptive context awareness"""
        
        # Get or create session context
        if session_id not in self.context_memory:
            self.context_memory[session_id] = {
                "previous_queries": [],
                "data_preferences": {},
                "focus_companies": set(),
                "analysis_depth": "standard"
            }
        
        context = self.context_memory[session_id]
        
        # Analyze query in context
        context_analysis = self._analyze_query_context(query, context)
        
        # Adapt approach based on context
        if context_analysis["requires_deep_dive"]:
            result = self._deep_analysis_approach(query, context)
        elif context_analysis["is_follow_up"]:
            result = self._follow_up_approach(query, context)
        else:
            result = self._standard_approach(query, context)
        
        # Update context memory
        self._update_context_memory(query, result, context)
        
        return result
    
    def _analyze_query_context(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze query in the context of conversation history"""
        context_prompt = PromptTemplate(
            input_variables=["query", "history"],
            template="""
            Analyze this query in the context of conversation history:
            Current Query: {query}
            
            Previous Queries: {history}
            
            Determine:
            - is_follow_up: Is this a follow-up to previous queries?
            - requires_deep_dive: Does this need comprehensive analysis?
            - focus_shift: Has the user shifted focus to new topics/companies?
            - complexity_level: simple/moderate/complex
            
            Return as JSON.
            """
        )
        
        history = [q["query"] for q in context["previous_queries"][-5:]]  # Last 5 queries
        result = self.llm(context_prompt.format(query=query, history=history))
        return json.loads(result)
    
    def _deep_analysis_approach(self, query: str, context: Dict) -> Dict[str, Any]:
        """Comprehensive analysis approach for complex queries"""
        return {
            "approach": "deep_analysis",
            "components": [
                "structured_data_analysis",
                "narrative_analysis", 
                "comparative_analysis",
                "trend_analysis",
                "risk_assessment"
            ],
            "result": "Comprehensive analysis result would go here"
        }
    
    def _follow_up_approach(self, query: str, context: Dict) -> Dict[str, Any]:
        """Efficient approach for follow-up queries building on previous context"""
        return {
            "approach": "follow_up",
            "building_on": context["previous_queries"][-1]["query"],
            "result": "Follow-up analysis result would go here"
        }
    
    def _standard_approach(self, query: str, context: Dict) -> Dict[str, Any]:
        """Standard approach for new queries"""
        return {
            "approach": "standard",
            "result": "Standard analysis result would go here"
        }
    
    def _update_context_memory(self, query: str, result: Dict, context: Dict):
        """Update conversation context memory"""
        context["previous_queries"].append({
            "query": query,
            "result_summary": result.get("summary", ""),
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        })
        
        # Keep only last 10 queries
        if len(context["previous_queries"]) > 10:
            context["previous_queries"] = context["previous_queries"][-10:]

# ===== USAGE EXAMPLES =====

def example_usage():
    """Example of how to use these hybrid agent patterns"""
    
    # Initialize components (pseudo-code)
    sql_db = SQLDatabase.from_uri("postgresql://...")
    vector_retriever = None  # Initialize your vector retriever
    llm = None  # Initialize your LLM
    
    # Pattern 1: Dual-Source Agent
    dual_agent = DualSourceFinancialAgent(sql_db, vector_retriever, llm)
    
    # Example queries
    queries = [
        "What was Toyota's revenue in 2023?",  # Structured
        "What are Toyota's main strategic priorities?",  # Unstructured  
        "How did Toyota's profitability change and what drove these changes?",  # Hybrid
        "Show me Sony's debt levels and explain their debt strategy"  # Contextual
    ]
    
    for query in queries:
        result = dual_agent.process_query(query)
        print(f"Query: {query}")
        print(f"Type: {result['type']}")
        print(f"Result: {result}")
        print("-" * 50)
    
    # Pattern 2: Multi-Agent Orchestration
    orchestrator = FinancialIntelligenceOrchestrator(sql_db, vector_retriever, llm)
    
    complex_query = "Analyze the competitive positioning of Japanese automakers in the EV market, including financial performance, strategic initiatives, and market outlook"
    complex_result = orchestrator.complex_analysis(complex_query)
    print(f"Complex Analysis Result: {complex_result}")
    
    # Pattern 3: Adaptive Context
    adaptive_agent = AdaptiveContextAgent(sql_db, vector_retriever, llm)
    
    session_id = "user_123"
    conversation_queries = [
        "Tell me about Toyota's financial performance",
        "How does this compare to Honda?", 
        "What about their EV strategies?",
        "Show me the actual investment numbers"
    ]
    
    for query in conversation_queries:
        result = adaptive_agent.process_with_context(query, session_id)
        print(f"Adaptive Query: {query}")
        print(f"Approach: {result['approach']}")
        print("-" * 30)

if __name__ == "__main__":
    example_usage()