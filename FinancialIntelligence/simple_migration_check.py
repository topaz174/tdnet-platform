#!/usr/bin/env python3
"""
Simple migration checker using psycopg2 (synchronous)
Validates database schema for enhanced system compatibility
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def check_database_schema():
    """Check current database schema and compatibility"""
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        print("❌ PG_DSN environment variable not set")
        return False
    
    try:
        conn = psycopg2.connect(pg_dsn)
        cur = conn.cursor()
        
        # Check if disclosures table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'disclosures'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("❌ 'disclosures' table not found")
            return False
        
        print("✅ 'disclosures' table found")
        
        # Check existing columns
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'disclosures'
            ORDER BY column_name
        """)
        columns = cur.fetchall()
        column_names = {row[0] for row in columns}
        
        required_columns = {
            'id', 'company_code', 'company_name', 'title', 
            'disclosure_date', 'pdf_path', 'embedding'
        }
        
        missing_columns = required_columns - column_names
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        
        # Check for classification columns
        classification_columns = {'category', 'subcategory', 'classification_l1', 'classification_l2'} & column_names
        if classification_columns:
            print(f"✅ Classification columns found: {classification_columns}")
        else:
            print("⚠️  No classification columns found - will still work but with reduced filtering capability")
        
        # Check embedding stats (without using array_length on vector)
        cur.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(embedding) as rows_with_embeddings
            FROM disclosures
        """)
        total_rows, rows_with_embeddings = cur.fetchone()
        
        # Try to get a sample embedding to check dimensions
        embedding_dim = "unknown"
        try:
            cur.execute("SELECT embedding FROM disclosures WHERE embedding IS NOT NULL LIMIT 1")
            sample_embedding = cur.fetchone()
            if sample_embedding and sample_embedding[0]:
                # For pgvector, the embedding should be a list when fetched
                embedding_dim = len(sample_embedding[0])
        except Exception as e:
            print(f"⚠️  Could not determine embedding dimensions: {e}")
        
        print(f"📊 Database Stats:")
        print(f"   Total documents: {total_rows}")
        print(f"   Documents with embeddings: {rows_with_embeddings}")
        print(f"   Embedding dimensions: {embedding_dim}")
        
        if embedding_dim != "unknown" and embedding_dim not in [1536, 1024]:
            print(f"⚠️  Unusual embedding dimension: {embedding_dim}")
        
        # Check for existing indexes
        cur.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'disclosures' AND indexname LIKE '%embedding%'
        """)
        indexes = [row[0] for row in cur.fetchall()]
        
        if indexes:
            print(f"✅ Vector indexes found: {indexes}")
        else:
            print("⚠️  No vector indexes found - will create during initialization")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def create_additional_indexes():
    """Create additional indexes for enhanced performance"""
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        return False
    
    try:
        conn = psycopg2.connect(pg_dsn)
        cur = conn.cursor()
        
        print("🔧 Creating additional indexes for enhanced performance...")
        
        # Check which classification columns exist
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'disclosures' AND column_name IN ('category', 'subcategory', 'classification_l1', 'classification_l2')
        """)
        existing_columns = {row[0] for row in cur.fetchall()}
        
        # Base indexes
        index_queries = [
            ("disclosures_company_code_idx", "CREATE INDEX IF NOT EXISTS disclosures_company_code_idx ON disclosures (company_code)"),
            ("disclosures_disclosure_date_idx", "CREATE INDEX IF NOT EXISTS disclosures_disclosure_date_idx ON disclosures (disclosure_date DESC)"),
            ("disclosures_company_date_idx", "CREATE INDEX IF NOT EXISTS disclosures_company_date_idx ON disclosures (company_code, disclosure_date DESC)"),
        ]
        
        # Add classification indexes based on what exists
        if 'category' in existing_columns:
            index_queries.append(("disclosures_category_idx", "CREATE INDEX IF NOT EXISTS disclosures_category_idx ON disclosures (category)"))
        if 'subcategory' in existing_columns:
            index_queries.append(("disclosures_subcategory_idx", "CREATE INDEX IF NOT EXISTS disclosures_subcategory_idx ON disclosures (subcategory)"))
        if 'classification_l1' in existing_columns:
            index_queries.append(("disclosures_classification_l1_idx", "CREATE INDEX IF NOT EXISTS disclosures_classification_l1_idx ON disclosures (classification_l1)"))
        if 'classification_l2' in existing_columns:
            index_queries.append(("disclosures_classification_l2_idx", "CREATE INDEX IF NOT EXISTS disclosures_classification_l2_idx ON disclosures (classification_l2)"))
        
        for index_name, query in index_queries:
            try:
                cur.execute(query)
                conn.commit()
                print(f"✅ Created/verified index: {index_name}")
            except Exception as e:
                print(f"⚠️  Index creation warning for {index_name}: {e}")
                conn.rollback()
        
        conn.close()
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
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ Missing file: {file}")
            return False
    
    return True

def main():
    """Main migration workflow"""
    print("🚀 Enhanced Financial Intelligence System Migration Check")
    print("=" * 60)
    
    # Step 1: Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix the issues above.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 2: Check database schema
    if not check_database_schema():
        print("\n❌ Database schema validation failed. Please check your database setup.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 3: Create additional indexes
    if not create_additional_indexes():
        print("\n⚠️  Index creation had issues, but system may still work.")
    
    print("\n" + "=" * 60)
    print("🎉 MIGRATION CHECK COMPLETED SUCCESSFULLY!")
    print("\nYour database is compatible with the enhanced system!")
    print("\nNext steps:")
    print("1. Test the enhanced system:")
    print("   python enhanced_agent.py")
    print("2. Update your code to use enhanced_retrieval_system.py")
    print("3. Use the enhanced agent for advanced capabilities")

if __name__ == "__main__":
    main()