#!/usr/bin/env python3
"""
Simple Agent Integration Example

This script shows how to quickly integrate the Advanced Hybrid Financial Intelligence Agent
with your existing PostgreSQL database and document structure.

This is a minimal integration that works with your current 'disclosures' table
while the full extraction pipeline is being implemented.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

# LangChain imports
from langchain.schema import BaseRetriever, Document
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Import the agent
from advanced_hybrid_agent import ProductionHybridFinancialAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExistingDataVectorRetriever(BaseRetriever):
    """
    Vector retriever that works with your existing 'disclosures' table
    Uses the pre-computed embeddings you already have
    """
    
    def __init__(self, connection_string: str, embedding_model=None):
        self.connection_string = connection_string
        self.embedding_model = embedding_model or OpenAIEmbeddings()
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents using existing embeddings"""
        
        # Generate query embedding
        try:
            if hasattr(self.embedding_model, 'embed_query'):
                query_embedding = self.embedding_model.embed_query(query)
            else:
                # Fallback for different embedding model interfaces
                query_embedding = self.embedding_model.encode(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use your existing disclosures table with embeddings
                cur.execute("""
                    SELECT 
                        d.id,
                        d.title,
                        d.company_name,
                        d.company_code,
                        d.disclosure_date,
                        d.category,
                        d.subcategory,
                        d.pdf_path,
                        1 - (d.embedding <=> %s::vector) as similarity
                    FROM disclosures d
                    WHERE d.embedding IS NOT NULL
                    AND 1 - (d.embedding <=> %s::vector) > 0.6
                    ORDER BY similarity DESC
                    LIMIT 15
                """, (query_embedding, query_embedding))
                
                results = cur.fetchall()
                logger.info(f"Retrieved {len(results)} documents for query: {query}")
                
                documents = []
                for row in results:
                    # Create document with metadata
                    doc = Document(
                        page_content=f"""
Document: {row['title']}
Company: {row['company_name']} ({row['company_code']})
Date: {row['disclosure_date']}
Category: {row['category']} - {row['subcategory']}

[This is a placeholder for the actual document content. 
 In the full implementation, this would contain the extracted PDF text.]
                        """.strip(),
                        metadata={
                            "id": row['id'],
                            "title": row['title'],
                            "company_name": row['company_name'],
                            "company_code": row['company_code'],
                            "disclosure_date": str(row['disclosure_date']),
                            "category": row['category'],
                            "subcategory": row['subcategory'],
                            "similarity": float(row['similarity']),
                            "source": row['pdf_path']
                        }
                    )
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
        finally:
            conn.close()

class SimpleFinancialDataProcessor:
    """
    Simple financial data processor that works with your existing structure
    Can be enhanced once the extraction pipeline is implemented
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def get_company_basic_info(self, company_code: str) -> Dict[str, Any]:
        """Get basic company information from existing disclosures"""
        
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        company_code,
                        company_name,
                        COUNT(*) as disclosure_count,
                        MIN(disclosure_date) as first_disclosure,
                        MAX(disclosure_date) as latest_disclosure,
                        array_agg(DISTINCT category) as categories
                    FROM disclosures
                    WHERE company_code = %s
                    GROUP BY company_code, company_name
                """, (company_code,))
                
                result = cur.fetchone()
                return dict(result) if result else {}
                
        except Exception as e:
            logger.error(f"Error getting company info: {e}")
            return {}
        finally:
            conn.close()
    
    def get_recent_disclosures(self, company_code: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent disclosures for analysis"""
        
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if company_code:
                    cur.execute("""
                        SELECT company_code, company_name, disclosure_date, 
                               title, category, subcategory
                        FROM disclosures
                        WHERE company_code = %s
                        ORDER BY disclosure_date DESC
                        LIMIT %s
                    """, (company_code, limit))
                else:
                    cur.execute("""
                        SELECT company_code, company_name, disclosure_date, 
                               title, category, subcategory
                        FROM disclosures
                        ORDER BY disclosure_date DESC
                        LIMIT %s
                    """, (limit,))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting recent disclosures: {e}")
            return []
        finally:
            conn.close()

class QuickStartAgent:
    """
    Quick start wrapper for the Advanced Hybrid Financial Intelligence Agent
    Works with your existing data structure
    """
    
    def __init__(self, connection_string: str, openai_api_key: str = None):
        # Set up OpenAI API key
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # Initialize components
        self.sql_db = SQLDatabase.from_uri(connection_string)
        self.vector_retriever = ExistingDataVectorRetriever(connection_string)
        self.llm = OpenAI(temperature=0, max_tokens=1000)
        
        # Simple data processor for current structure
        self.data_processor = SimpleFinancialDataProcessor(connection_string)
        
        # Initialize the main agent
        self.agent = ProductionHybridFinancialAgent(
            sql_db=self.sql_db,
            vector_retriever=self.vector_retriever,
            llm=self.llm
        )
        
        logger.info("Quick Start Agent initialized successfully")
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Process a financial query"""
        
        logger.info(f"Processing query: {question}")
        
        try:
            # Process with the hybrid agent
            result = await self.agent.process_query(question)
            
            # Format response for easy consumption
            response = {
                "question": question,
                "answer": result.synthesis,
                "confidence": result.confidence,
                "adequacy_score": result.adequacy_score,
                "execution_time": result.execution_time,
                "companies_mentioned": [c.name for c in result.intent.companies],
                "query_complexity": result.intent.complexity.value,
                "sources_count": len(result.sources),
                "reasoning_notes": result.reasoning_notes
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your query: {str(e)}",
                "confidence": 0.0,
                "error": True
            }
    
    def get_company_overview(self, company_code: str) -> Dict[str, Any]:
        """Get a quick company overview using available data"""
        
        company_info = self.data_processor.get_company_basic_info(company_code)
        recent_disclosures = self.data_processor.get_recent_disclosures(company_code, 5)
        
        return {
            "company_info": company_info,
            "recent_activity": recent_disclosures
        }
    
    def get_market_overview(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent market activity overview"""
        
        recent_disclosures = self.data_processor.get_recent_disclosures(limit=limit)
        
        # Group by category for quick insights
        categories = {}
        for disclosure in recent_disclosures:
            category = disclosure['category']
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return {
            "recent_disclosures": recent_disclosures,
            "category_breakdown": categories,
            "total_disclosures": len(recent_disclosures)
        }

async def demo():
    """
    Demo function showing how to use the agent with your existing data
    """
    
    # Configuration - UPDATE THESE VALUES
    CONNECTION_STRING = "postgresql://username:password@localhost/your_database"
    OPENAI_API_KEY = "your-openai-api-key"  # Or set OPENAI_API_KEY environment variable
    
    print("🚀 Starting Financial Intelligence Agent Demo")
    print("=" * 60)
    
    try:
        # Initialize the quick start agent
        agent = QuickStartAgent(CONNECTION_STRING, OPENAI_API_KEY)
        
        # Demo queries
        demo_queries = [
            "What companies have disclosed earnings recently?",
            "Show me recent automotive company disclosures",
            "What are the latest developments in the technology sector?",
            "Find recent risk factor disclosures",
            "Compare recent disclosure activity across different categories"
        ]
        
        print("\n📋 Sample Queries:")
        for i, query in enumerate(demo_queries, 1):
            print(f"{i}. {query}")
        
        print("\n" + "=" * 60)
        
        # Process each demo query
        for query in demo_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            
            result = await agent.query(query)
            
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"📈 Adequacy: {result['adequacy_score']:.2f}")
            print(f"⏱️  Time: {result['execution_time']:.2f}s")
            print(f"🏢 Companies: {', '.join(result['companies_mentioned'][:3])}")
            print(f"\n💬 Answer: {result['answer'][:300]}...")
            
            if result.get('error'):
                print(f"❌ Error occurred: {result['answer']}")
            
            print("\n" + "=" * 60)
        
        # Show data overview
        print("\n📈 Market Overview:")
        market_overview = agent.get_market_overview()
        print(f"Recent disclosures: {market_overview['total_disclosures']}")
        print("Category breakdown:")
        for category, count in market_overview['category_breakdown'].items():
            print(f"  {category}: {count}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nPlease check your connection string and API key configuration.")

def interactive_mode():
    """
    Interactive mode for testing queries
    """
    
    # Configuration - UPDATE THESE VALUES
    CONNECTION_STRING = "postgresql://username:password@localhost/your_database"
    OPENAI_API_KEY = "your-openai-api-key"
    
    print("🤖 Financial Intelligence Agent - Interactive Mode")
    print("Type 'exit' to quit, 'help' for sample queries")
    print("=" * 60)
    
    try:
        agent = QuickStartAgent(CONNECTION_STRING, OPENAI_API_KEY)
        
        while True:
            query = input("\n💬 Your question: ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("\n📋 Sample queries:")
                print("- What companies disclosed earnings recently?")
                print("- Show me Toyota's recent disclosures")
                print("- What are the main risk factors in tech companies?")
                print("- Compare Honda and Nissan recent activity")
                continue
            elif not query:
                continue
            
            print("\n🔍 Processing...")
            
            async def process_query():
                return await agent.query(query)
            
            result = asyncio.run(process_query())
            
            print(f"\n📊 Confidence: {result['confidence']:.2f} | "
                  f"Adequacy: {result['adequacy_score']:.2f} | "
                  f"Time: {result['execution_time']:.2f}s")
            print(f"\n💬 {result['answer']}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        asyncio.run(demo())