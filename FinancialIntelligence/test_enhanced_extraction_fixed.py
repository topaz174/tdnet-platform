#!/usr/bin/env python3
"""
Test script for enhanced extraction with proper connection management
"""

import asyncio
import logging
from complete_enhanced_agent_with_extraction import CompleteEnhancedAgentWithExtraction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_extraction():
    """Test the enhanced extraction system with proper connection management"""
    
    # Create agent once and reuse
    agent = CompleteEnhancedAgentWithExtraction()
    
    test_queries = [
        "Which Japanese companies raised dividends last quarter?",
        "What companies revised their earnings guidance upward recently?",
        "株主還元方針の変更"
    ]
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/3: {query}")
            print('='*80)
            
            try:
                response = await agent.process_query(query)
                print(response)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                
            print(f"\n{'='*80}")
            print(f"TEST {i} COMPLETED")
            print('='*80)
    
    finally:
        # Clean up resources at the very end
        try:
            if hasattr(agent.retrieval_system, 'pg') and agent.retrieval_system.pg:
                await agent.retrieval_system.close()
                print("✓ Database connection closed")
        except Exception as e:
            print(f"Warning: Error closing connection: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_extraction())