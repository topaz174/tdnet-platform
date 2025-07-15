"""
Migration script to add classification columns to existing database.
Run this once to add the new category and subcategory columns.
"""

from sqlalchemy import create_engine, text
from config import DB_URL

def add_classification_columns():
    """Add category and subcategory columns to the disclosures table."""
    engine = create_engine(DB_URL)
    
    try:
        with engine.connect() as conn:
            # Check if columns already exist
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'disclosures' 
                AND column_name IN ('category', 'subcategory')
            """))
            
            existing_columns = [row[0] for row in result]
            
            if 'category' not in existing_columns:
                print("Adding 'category' column...")
                conn.execute(text("ALTER TABLE disclosures ADD COLUMN category TEXT"))
                conn.commit()
                print("✓ Added 'category' column")
            else:
                print("✓ 'category' column already exists")
            
            if 'subcategory' not in existing_columns:
                print("Adding 'subcategory' column...")
                conn.execute(text("ALTER TABLE disclosures ADD COLUMN subcategory TEXT"))
                conn.commit()
                print("✓ Added 'subcategory' column")
            else:
                print("✓ 'subcategory' column already exists")
                
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        raise

if __name__ == "__main__":
    print("Adding classification columns to database...")
    add_classification_columns() 