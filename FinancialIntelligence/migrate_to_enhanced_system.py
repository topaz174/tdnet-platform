#!/usr/bin/env python3
"""
Migration script to switch from old retrieval system to enhanced system
Ensures database compatibility and validates existing embeddings
"""

import os, asyncio, logging
from pathlib import Path
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def check_database_schema():
    """Check current database schema and compatibility"""
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        print("❌ PG_DSN environment variable not set")
        return False
    
    try:
        conn = await asyncpg.connect(pg_dsn)
        
        # Check if disclosures table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'disclosures'
            )
        """)
        
        if not table_exists:
            print("❌ 'disclosures' table not found")
            await conn.close()
            return False
        
        print("✅ 'disclosures' table found")
        
        # Check existing columns
        columns = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'disclosures'
            ORDER BY column_name
        """)
        
        column_names = {row['column_name'] for row in columns}
        required_columns = {
            'id', 'company_code', 'company_name', 'title', 
            'disclosure_date', 'pdf_path', 'embedding'
        }
        
        missing_columns = required_columns - column_names
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            await conn.close()
            return False
        
        print("✅ All required columns present")
        
        # Check for classification columns
        classification_columns = {'category', 'subcategory'} & column_names
        if classification_columns:
            print(f"✅ Classification columns found: {classification_columns}")
        else:
            print("⚠️  No classification columns found - will still work but with reduced filtering capability")
        
        # Check embedding dimensions
        embedding_info = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(embedding) as rows_with_embeddings
            FROM disclosures
        """)
        
        # Get embedding dimensions using vector-specific approach
        embedding_dim = None
        try:
            sample_embedding = await conn.fetchval("""
                SELECT embedding FROM disclosures WHERE embedding IS NOT NULL LIMIT 1
            """)
            if sample_embedding:
                # Convert vector to list and get length
                embedding_dim = len(sample_embedding)
        except Exception as e:
            print(f"⚠️  Could not determine embedding dimensions: {e}")
            embedding_dim = "unknown"
        
        print(f"📊 Database Stats:")
        print(f"   Total documents: {embedding_info['total_rows']}")
        print(f"   Documents with embeddings: {embedding_info['rows_with_embeddings']}")
        print(f"   Embedding dimensions: {embedding_dim}")
        
        if embedding_dim and embedding_dim not in [1536, 1024, 1000, 768]:
            print(f"ℹ️  Custom embedding dimension detected: {embedding_dim} (system will adapt automatically)")
        
        # Check for existing indexes
        indexes = await conn.fetch("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'disclosures' AND indexname LIKE '%embedding%'
        """)
        
        if indexes:
            print(f"✅ Vector indexes found: {[idx['indexname'] for idx in indexes]}")
        else:
            print("⚠️  No vector indexes found - will create during initialization")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

async def create_additional_indexes():
    """Create additional indexes for enhanced performance"""
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        return False
    
    try:
        conn = await asyncpg.connect(pg_dsn)
        
        print("🔧 Creating additional indexes for enhanced performance...")
        
        # Indexes for enhanced filtering (removed CONCURRENTLY to avoid transaction issues)
        index_queries = [
            "CREATE INDEX IF NOT EXISTS disclosures_company_code_idx ON disclosures (company_code);",
            "CREATE INDEX IF NOT EXISTS disclosures_disclosure_date_idx ON disclosures (disclosure_date DESC);",
            "CREATE INDEX IF NOT EXISTS disclosures_company_date_idx ON disclosures (company_code, disclosure_date DESC);",
        ]
        
        # Only create classification indexes if columns exist
        columns = await conn.fetch("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'disclosures' AND (column_name IN ('category', 'subcategory', 'classification_l1', 'classification_l2'))
        """)
        
        column_names = {col['column_name'] for col in columns}
        
        if 'category' in column_names:
            index_queries.append("CREATE INDEX IF NOT EXISTS disclosures_category_idx ON disclosures (category);")
        elif 'classification_l1' in column_names:
            index_queries.append("CREATE INDEX IF NOT EXISTS disclosures_classification_l1_idx ON disclosures (classification_l1);")
        
        if 'subcategory' in column_names:
            index_queries.append("CREATE INDEX IF NOT EXISTS disclosures_subcategory_idx ON disclosures (subcategory);")
        elif 'classification_l2' in column_names:
            index_queries.append("CREATE INDEX IF NOT EXISTS disclosures_classification_l2_idx ON disclosures (classification_l2);")
        
        for query in index_queries:
            try:
                await conn.execute(query)
                # Extract index name more reliably
                index_name = query.split()[4]  # Should be the index name after "CREATE INDEX IF NOT EXISTS"
                print(f"✅ Created/verified index: {index_name}")
            except Exception as e:
                print(f"⚠️  Index creation warning: {e}")
        
        await conn.close()
        print("✅ Additional indexes created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        return False

def validate_environment():
    """Validate environment variables and dependencies"""
    print("🔍 Validating environment...")
    
    required_env_vars = ["PG_DSN", "OPENAI_API_KEY"]
    optional_env_vars = ["REDIS_URL"]
    
    for var in required_env_vars:
        if not os.getenv(var):
            print(f"❌ Missing required environment variable: {var}")
            return False
        print(f"✅ {var} is set")
    
    for var in optional_env_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} not set - caching will be disabled")
    
    # Check if files exist
    required_files = [
        "enhanced_retrieval_system.py",
        "enhanced_agent.py"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ Missing file: {file}")
            return False
    
    return True

async def test_enhanced_system():
    """Test the enhanced system with a simple query"""
    print("🧪 Testing enhanced system...")
    
    try:
        from enhanced_retrieval_system import EnhancedFinancialRetrievalSystem, EnhancedRetrievalConfig
        from enhanced_agent import EnhancedFinancialAgent
        from langchain_openai import ChatOpenAI
        
        config = EnhancedRetrievalConfig(
            pg_dsn=os.getenv("PG_DSN"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        
        retrieval_system = EnhancedFinancialRetrievalSystem(config)
        await retrieval_system.init()
        
        # Test basic search
        results = await retrieval_system.search("earnings", k=5)
        print(f"✅ Search test: Found {len(results)} documents")
        
        # Test agent if OpenAI key is available
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            agent = EnhancedFinancialAgent(retrieval_system, llm)
            
            # Simple test query
            answer = await agent.run("What are recent earnings reports?")
            print(f"✅ Agent test: Generated {len(answer)} character response")
        else:
            print("⚠️  Skipping agent test - OPENAI_API_KEY not set")
        
        await retrieval_system.close()
        print("✅ Enhanced system test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system test failed: {e}")
        return False

async def main():
    """Main migration workflow"""
    print("🚀 Enhanced Financial Intelligence System Migration")
    print("=" * 60)
    
    # Step 1: Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix the issues above.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 2: Check database schema
    if not await check_database_schema():
        print("\n❌ Database schema validation failed. Please check your database setup.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 3: Create additional indexes
    if not await create_additional_indexes():
        print("\n⚠️  Index creation had issues, but system may still work.")
    
    print("\n" + "=" * 60)
    
    # Step 4: Test enhanced system
    if not await test_enhanced_system():
        print("\n❌ Enhanced system test failed. Please check the error messages above.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
    print("\nNext steps:")
    print("1. Update your code to import from enhanced_retrieval_system")
    print("2. Replace FinancialRetrievalSystem with EnhancedFinancialRetrievalSystem")
    print("3. Use enhanced_agent.py for advanced capabilities")
    print("4. Test with your specific queries and company codes")
    print("\nExample usage:")
    print("  from enhanced_agent import EnhancedFinancialAgent")
    print("  agent = EnhancedFinancialAgent(enhanced_system, llm)")
    print("  result = await agent.run('Your financial query here')")

if __name__ == "__main__":
    asyncio.run(main())