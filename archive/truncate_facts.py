#!/usr/bin/env python3
"""
Truncate financial facts related tables and reset ID sequences.
This script clears all data from financial_facts, concepts, concept_tags, and context_dims
while preserving xbrl_filings and filing_sections data.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.config import DB_URL


def truncate_facts_tables():
    """Truncate financial facts related tables and reset sequences."""
    engine = create_engine(DB_URL)
    session = sessionmaker(bind=engine)()
    
    try:
        print("Starting truncation of financial facts related tables...")
        
        # Truncate tables in correct order (due to foreign key dependencies)
        tables_to_truncate = [
            'financial_facts',  # Has foreign keys to concepts and context_dims
            'concept_tags',     # Has foreign key to concepts
            'concepts',         # Referenced by concept_tags and financial_facts
            'context_dims'      # Referenced by financial_facts
        ]
        
        for table in tables_to_truncate:
            print(f"Truncating {table}...")
            
            # Truncate table and restart identity (reset auto-increment sequences)
            truncate_query = text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            session.execute(truncate_query)
            
            print(f"✓ {table} truncated and ID sequence reset")
        
        session.commit()
        print("\n✅ All financial facts tables truncated successfully!")
        print("Tables affected:")
        print("- financial_facts")
        print("- concepts") 
        print("- concept_tags")
        print("- context_dims")
        print("\nTables preserved:")
        print("- xbrl_filings")
        print("- filing_sections")
        print("- companies")
        print("- disclosures")
        
    except Exception as e:
        print(f"❌ Error during truncation: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    confirm = input("This will delete ALL data from financial_facts, concepts, concept_tags, and context_dims tables. Are you sure? (y/n): ")
    
    if confirm.lower() == 'y':
        truncate_facts_tables()
    else:
        print("Operation cancelled.") 