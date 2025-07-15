#!/usr/bin/env python3
"""
Simple test of the enhanced retrieval system
"""

import asyncio
import os
from enhanced_retrieval_system import EnhancedFinancialRetrievalSystem, EnhancedRetrievalConfig

async def test_system():
    # Initialize system without Redis to avoid connection issues
    config = EnhancedRetrievalConfig(
        pg_dsn=os.getenv("PG_DSN"),
        redis_url=""  # Empty string to disable Redis
    )
    
    print("🧪 Testing Enhanced Financial Retrieval System...")
    
    try:
        system = EnhancedFinancialRetrievalSystem(config)
        await system.init()
        
        print("✅ System initialized successfully")
        
        # Test basic search
        test_queries = [
            "earnings",
            "配当",  # dividend in Japanese
            "recent announcements"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = await system.search(query, k=3)
            print(f"   Found {len(results)} documents")
            
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. {result.name} ({result.code}) - {result.title[:50]}...")
                print(f"      Date: {result.date} | Score: {result.score:.3f}")
        
        await system.close()
        print("\n✅ Enhanced system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_system())