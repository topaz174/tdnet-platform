#!/usr/bin/env python3
"""
Simple test of enhanced retrieval with asyncpg
"""

import asyncio
import os
from enhanced_retrieval_system import EnhancedFinancialRetrievalSystem, EnhancedRetrievalConfig

async def main():
    config = EnhancedRetrievalConfig(
        pg_dsn=os.getenv("PG_DSN"),
        redis_url=""  # Disable Redis to avoid issues
    )
    
    print("🧪 Testing Enhanced System with asyncpg...")
    
    system = EnhancedFinancialRetrievalSystem(config)
    await system.init()
    
    # Test query
    query = "dividend increases"
    print(f"\n🔍 Testing query: '{query}'")
    
    results = await system.search(query, k=5)
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.name} ({result.code}) - Score: {result.score:.4f}")
        print(f"   {result.title[:80]}...")
        print(f"   Date: {result.date}")
    
    await system.close()
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(main())