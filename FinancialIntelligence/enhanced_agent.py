#!/usr/bin/env python3
"""
Enhanced Financial Intelligence Agent
Integrates with the enhanced retrieval system for sophisticated financial analysis
"""

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import os, logging, json

from enhanced_retrieval_system import (
    EnhancedFinancialRetrievalSystem, 
    EnhancedRetrievalConfig, 
    RetrievalResult,
    QueryClassification
)

# ------------------------------------------------------------------
# AGENT STATE & TOOLS
# ------------------------------------------------------------------
class EnhancedAgentState(TypedDict):
    query: str
    query_classification: Optional[QueryClassification]
    search_results: List[RetrievalResult]
    filtered_results: List[RetrievalResult]
    reasoning_steps: List[str]
    financial_context: Dict[str, Any]
    final_answer: str

class EnhancedDocSearchTool(BaseTool):
    name: str = "enhanced_doc_search"
    description: str = """
    Advanced financial document search with smart filtering and query enhancement.
    Supports filters: company_codes, date_range, classifications, subcategory.
    Example: search("dividend increases Toyota", filters={"company_codes": ["7203"], "date_range": [date(2024,1,1), date(2024,12,31)]})
    """
    
    def __init__(self, retrieval_system: EnhancedFinancialRetrievalSystem):
        super().__init__()
        object.__setattr__(self, '_system', retrieval_system)
    
    async def _arun(self, query: str, filters: Dict[str, Any] = None, k: int = 25) -> List[RetrievalResult]:
        """Enhanced async search with intelligent filtering"""
        filters = filters or {}
        return await self._system.search(query, filters=filters, k=k)
    
    def _run(self, query: str, filters: Dict[str, Any] = None, k: int = 25) -> List[RetrievalResult]:
        raise NotImplementedError("Use async version _arun instead")

class CompanyStatsTool(BaseTool):
    name: str = "company_stats"
    description: str = "Get document statistics and activity overview for a specific company by code"
    
    def __init__(self, retrieval_system: EnhancedFinancialRetrievalSystem):
        super().__init__()
        object.__setattr__(self, '_system', retrieval_system)
    
    async def _arun(self, company_code: str, days: int = 365) -> Dict[str, Any]:
        return await self._system.get_company_stats(company_code, days)
    
    def _run(self, company_code: str, days: int = 365) -> Dict[str, Any]:
        raise NotImplementedError("Use async version _arun instead")

# ------------------------------------------------------------------
# ENHANCED AGENT
# ------------------------------------------------------------------
class EnhancedFinancialAgent:
    """Advanced financial agent with multi-stage reasoning and smart retrieval"""
    
    def __init__(self, retrieval_system: EnhancedFinancialRetrievalSystem, llm: ChatOpenAI):
        self.retrieval_system = retrieval_system
        self.llm = llm
        self.search_tool = EnhancedDocSearchTool(retrieval_system)
        self.stats_tool = CompanyStatsTool(retrieval_system)
        self.flow = self._build_workflow()
        self.log = logging.getLogger(__name__)

    def _build_workflow(self):
        """Build LangGraph workflow with enhanced reasoning stages"""
        graph = StateGraph(EnhancedAgentState)
        
        # Add nodes for multi-stage processing
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("initial_search", self._initial_search_node)
        graph.add_node("refine_search", self._refine_search_node)
        graph.add_node("analyze_results", self._analyze_results_node)
        graph.add_node("compose_answer", self._compose_answer_node)
        
        # Define workflow edges
        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "initial_search")
        graph.add_edge("initial_search", "refine_search")
        graph.add_edge("refine_search", "analyze_results")
        graph.add_edge("analyze_results", "compose_answer")
        graph.add_edge("compose_answer", END)
        
        return graph.compile()

    async def _analyze_query_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Stage 1: Analyze and classify the user query"""
        query = state["query"]
        
        # Use the knowledge base for query classification
        query_class = self.retrieval_system.kb.classify_query(query)
        
        state["query_classification"] = query_class
        state["reasoning_steps"] = [
            f"🔍 Query Analysis: Type={query_class.query_type}",
            f"📊 Financial Terms: {query_class.financial_terms}",
            f"🏢 Companies: {query_class.companies}",
            f"⏰ Time Indicators: {query_class.time_indicators}",
            f"🔄 Expanded Query: {query_class.expanded_query[:100]}..."
        ]
        
        return state

    async def _initial_search_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Stage 2: Perform initial broad search"""
        query = state["query"]
        query_class = state["query_classification"]
        
        # Build smart filters based on query analysis
        filters = {}
        
        # Add time-based filters for temporal queries
        if query_class.query_type == "temporal" and any(term in query_class.time_indicators for term in ["最近", "直近", "最新"]):
            recent_date = date.today() - timedelta(days=90)
            filters["date_range"] = [recent_date, date.today()]
        
        # Perform enhanced search
        results = await self.search_tool._arun(query, filters=filters, k=30)
        
        state["search_results"] = results
        state["reasoning_steps"].append(f"🔎 Initial Search: Found {len(results)} documents")
        
        return state

    async def _refine_search_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Stage 3: Refine search if needed based on initial results"""
        initial_results = state["search_results"]
        query_class = state["query_classification"]
        
        # Check if we need additional searches
        refined_results = initial_results.copy()
        
        # For comparative queries, ensure we have diverse company representation
        if query_class.query_type == "comparative" and len(initial_results) > 0:
            companies_found = set(r.code for r in initial_results)
            
            if len(companies_found) < 2:
                # Search for more companies in the same category
                if initial_results:
                    classification = initial_results[0].classification_l1
                    additional_filters = {"classifications": [classification]}
                    additional_results = await self.search_tool._arun(
                        query_class.expanded_query, 
                        filters=additional_filters, 
                        k=20
                    )
                    
                    # Merge results, avoiding duplicates
                    existing_ids = set(r.id for r in refined_results)
                    new_results = [r for r in additional_results if r.id not in existing_ids]
                    refined_results.extend(new_results[:10])
        
        # For temporal queries, ensure chronological diversity
        elif query_class.query_type == "temporal" and len(initial_results) > 5:
            # Sort by date and take representative samples across time periods
            sorted_by_date = sorted(initial_results, key=lambda x: x.date, reverse=True)
            refined_results = sorted_by_date[:15]  # Focus on most recent
        
        state["filtered_results"] = refined_results
        state["reasoning_steps"].append(f"⚡ Refined Search: Selected {len(refined_results)} relevant documents")
        
        return state

    async def _analyze_results_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Stage 4: Analyze search results for financial insights"""
        results = state["filtered_results"]
        query_class = state["query_classification"]
        
        if not results:
            state["financial_context"] = {"analysis": "No relevant documents found"}
            return state
        
        # Analyze document patterns
        analysis = {
            "total_documents": len(results),
            "date_range": {
                "earliest": min(r.date for r in results),
                "latest": max(r.date for r in results)
            },
            "companies": list(set(r.code for r in results)),
            "company_count": len(set(r.code for r in results)),
            "classifications": list(set(r.classification_l1 for r in results if r.classification_l1)),
            "top_companies": {}
        }
        
        # Company frequency analysis
        company_counts = {}
        for result in results:
            company_counts[result.name] = company_counts.get(result.name, 0) + 1
        
        analysis["top_companies"] = dict(sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Time-based analysis for temporal queries
        if query_class.query_type == "temporal":
            # Group by month for trend analysis
            monthly_counts = {}
            for result in results:
                month_key = result.date.strftime("%Y-%m")
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            
            analysis["temporal_distribution"] = monthly_counts
        
        # Get detailed stats for top companies
        if len(analysis["companies"]) <= 3:
            company_details = {}
            for company_code in analysis["companies"][:3]:
                try:
                    stats = await self.stats_tool._arun(company_code, days=365)
                    company_details[company_code] = stats
                except Exception as e:
                    self.log.debug(f"Could not get stats for {company_code}: {e}")
            
            analysis["detailed_company_stats"] = company_details
        
        state["financial_context"] = analysis
        state["reasoning_steps"].append(f"📈 Analysis: {analysis['company_count']} companies, {len(analysis['classifications'])} document types")
        
        return state

    async def _compose_answer_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Stage 5: Compose comprehensive financial analysis answer"""
        query = state["query"]
        results = state["filtered_results"]
        context = state["financial_context"]
        query_class = state["query_classification"]
        
        if not results:
            state["final_answer"] = "I couldn't find any relevant financial documents matching your query. Please try rephrasing or check if the company codes/terms are correct."
            return state
        
        # Build context for LLM
        top_documents = results[:8]  # Focus on top results
        doc_summaries = []
        
        for i, doc in enumerate(top_documents, 1):
            summary = f"{i}. **{doc.name}** ({doc.code}) - {doc.title}\n"
            summary += f"   📅 Date: {doc.date} | Score: {doc.score:.3f}"
            if doc.classification_l1:
                summary += f" | Type: {doc.classification_l1}"
            if doc.ctx:
                summary += f"\n   💬 Context: {doc.ctx[:150]}..."
            doc_summaries.append(summary)
        
        documents_text = "\n\n".join(doc_summaries)
        
        # Build analysis summary
        analysis_text = f"""
**Query Analysis:**
- Type: {query_class.query_type.title()} Query
- Companies Found: {context['company_count']} ({', '.join(context['companies'][:5])})
- Date Range: {context['date_range']['earliest']} to {context['date_range']['latest']}
- Document Types: {', '.join(context['classifications'])}

**Key Findings:**
- Total Relevant Documents: {context['total_documents']}
- Most Active Companies: {', '.join([f"{k} ({v} docs)" for k, v in list(context['top_companies'].items())[:3]])}
"""
        
        # Add temporal insights for temporal queries
        if query_class.query_type == "temporal" and "temporal_distribution" in context:
            temporal_data = context["temporal_distribution"]
            recent_months = sorted(temporal_data.items())[-3:]
            analysis_text += f"\n- Recent Activity: {', '.join([f'{k}: {v} docs' for k, v in recent_months])}"
        
        # Compose final prompt
        prompt = f"""You are an expert financial analyst specializing in Japanese corporate disclosures. 

**User Question:** "{query}"

{analysis_text}

**Top Relevant Documents:**
{documents_text}

**Instructions:**
1. Provide a comprehensive analysis addressing the user's specific question
2. Cite specific companies, dates, and document types from the evidence
3. If this is a comparative query, highlight differences between companies
4. If this is a temporal query, explain trends and timing patterns
5. Be specific about what the documents reveal and any limitations
6. Use bullet points for clarity and include relevant financial metrics when mentioned
7. Write in English but include Japanese company names when appropriate

**Answer:**"""

        # Generate response
        response = await self.llm.ainvoke(prompt)
        
        # Add reasoning trail for transparency
        reasoning_summary = "\n".join([f"• {step}" for step in state["reasoning_steps"]])
        
        final_answer = f"{response.content}\n\n---\n**Search Process:**\n{reasoning_summary}"
        
        state["final_answer"] = final_answer
        
        return state

    # ----------
    # External API
    async def run(self, query: str, **kwargs) -> str:
        """Run the enhanced financial analysis workflow"""
        initial_state = EnhancedAgentState(
            query=query,
            query_classification=None,
            search_results=[],
            filtered_results=[],
            reasoning_steps=[],
            financial_context={},
            final_answer=""
        )
        
        try:
            final_state = await self.flow.ainvoke(initial_state)
            return final_state["final_answer"]
        except Exception as e:
            self.log.error(f"Error in agent workflow: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question."

    async def search_documents(self, query: str, **filters) -> List[RetrievalResult]:
        """Direct document search interface"""
        return await self.search_tool._arun(query, filters=filters)

    async def get_company_overview(self, company_code: str) -> Dict[str, Any]:
        """Get comprehensive company overview"""
        return await self.stats_tool._arun(company_code)

# ------------------------------------------------------------------
# DEMO USAGE
# ------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    
    async def demo():
        # Initialize system
        config = EnhancedRetrievalConfig(
            pg_dsn=os.getenv("PG_DSN", "postgresql://user:pass@localhost/tdnet"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        
        retrieval_system = EnhancedFinancialRetrievalSystem(config)
        await retrieval_system.init()
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        agent = EnhancedFinancialAgent(retrieval_system, llm)
        
        # Test queries
        test_queries = [
            "Which companies announced dividend increases in the last quarter?",
            "Show me recent earnings guidance revisions for Toyota and Honda",
            "What companies have had significant M&A activity recently?",
            "Compare the recent performance announcements of major tech companies",
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            answer = await agent.run(query)
            print(answer)
        
        await retrieval_system.close()
    
    asyncio.run(demo())