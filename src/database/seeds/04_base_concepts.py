#!/usr/bin/env python3
"""
Load base concepts from xsd_elements_organized.json into the concepts table.

This module handles seeding of base taxonomy concepts only.
Extension taxonomy processing is handled by src/quantitative/etl/extension_concept_processor.py
"""

import json
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# JSON file path (modify this as needed)
JSON_FILE_PATH = "/home/alex/dev/tdnet-platform/config/data/xsd_elements_universal.json"

# Import unified config
from config.config import DB_URL


def load_concepts():
    """Load base concepts from JSON file into the database."""
    
    if not Path(JSON_FILE_PATH).exists():
        print(f"Error: JSON file not found at {JSON_FILE_PATH}")
        return False
    
    # Load the JSON data
    print(f"Loading concepts from {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    
    # Prepare data for insertion
    concepts_to_insert = []
    for local_name, concept_info in concepts_data.items():
        concept_record = (
            concept_info.get('taxonomy'),      # taxonomy_prefix
            local_name,                        # local_name
            concept_info.get('label_en'),      # std_label_en
            concept_info.get('label_ja'),      # std_label_ja
            concept_info.get('item_type'),     # item_type
            concept_info.get('latest_version') # taxonomy_version
        )
        concepts_to_insert.append(concept_record)
    
    # Connect to database and insert data
    session = None
    try:
        engine = create_engine(DB_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print(f"Inserting {len(concepts_to_insert)} concepts...")
        
        # Clear existing data
        session.execute(text("DELETE FROM concepts"))
        print("Cleared existing concepts")
        
        # Insert new concepts in batches
        inserted_count = 0
        for concept_record in concepts_to_insert:
            taxonomy_prefix, local_name, std_label_en, std_label_ja, item_type, taxonomy_version = concept_record
            
            session.execute(
                text("""
                    INSERT INTO concepts (taxonomy_prefix, local_name, std_label_en, std_label_ja, item_type, taxonomy_version)
                    VALUES (:taxonomy_prefix, :local_name, :std_label_en, :std_label_ja, :item_type, :taxonomy_version)
                    ON CONFLICT (taxonomy_prefix, local_name) DO UPDATE SET
                        std_label_en = EXCLUDED.std_label_en,
                        std_label_ja = EXCLUDED.std_label_ja,
                        item_type = EXCLUDED.item_type,
                        taxonomy_version = EXCLUDED.taxonomy_version
                """),
                {
                    'taxonomy_prefix': taxonomy_prefix,
                    'local_name': local_name,
                    'std_label_en': std_label_en,
                    'std_label_ja': std_label_ja,
                    'item_type': item_type,
                    'taxonomy_version': taxonomy_version
                }
            )
            inserted_count += 1
            if inserted_count % 1000 == 0:
                print(f"Inserted {inserted_count} concepts...")
        
        # Commit changes
        session.commit()
        
        # Get count of total records
        count_result = session.execute(text("SELECT COUNT(*) FROM concepts"))
        count = count_result.scalar()
        
        print(f"Successfully loaded {count} concepts into the database")
        
        # Show some statistics
        standards_result = session.execute(text("""
            SELECT taxonomy_prefix, COUNT(*) 
            FROM concepts 
            GROUP BY taxonomy_prefix 
            ORDER BY taxonomy_prefix
        """))
        taxonomies = standards_result.fetchall()
        
        print("\nConcepts by taxonomy prefix:")
        for taxonomy_prefix, count in taxonomies:
            print(f"  {taxonomy_prefix}: {count}")
        
        item_types_result = session.execute(text("""
            SELECT item_type, COUNT(*) 
            FROM concepts 
            WHERE item_type IS NOT NULL
            GROUP BY item_type 
            ORDER BY COUNT(*) DESC
        """))
        item_types = item_types_result.fetchall()
        
        print("\nConcepts by item type:")
        for item_type, count in item_types:
            print(f"  {item_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error loading concepts: {e}")
        if session is not None:
            session.rollback()
        return False
    
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    success = load_concepts()
    sys.exit(0 if success else 1) 