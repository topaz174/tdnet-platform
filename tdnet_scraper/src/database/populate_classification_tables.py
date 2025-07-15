#!/usr/bin/env python3
"""
Populate classification tables with category and subcategory names.
Regex patterns remain in classification_rules.py - only names are stored in database.
"""
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import sessionmaker
from src.database.init_db import (
    engine, DisclosureCategory, DisclosureSubcategory, Base
)
from src.classifier.classification_rules import (
    CATEGORIES, CATEGORY_TAXONOMY, SUBCATEGORY_RULES, SUBCATEGORY_TO_PARENT,
    CATEGORY_TRANSLATIONS, SUBCATEGORY_TRANSLATIONS
)


def extract_english_and_japanese(name):
    """
    Extract English and Japanese parts from subcategory names.
    Format: "English Name [Japanese Text]"
    """
    match = re.match(r'^(.+?)\s*\[(.+?)\]$', name)
    if match:
        english = match.group(1).strip()
        japanese = match.group(2).strip()
        return english, japanese
    else:
        # No Japanese translation found, return name as English
        return name.strip(), None


def populate_classification_tables():
    """
    Populate the classification tables with category and subcategory names.
    Includes both English and Japanese translations.
    """
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Clear existing data to avoid conflicts
        print("Clearing existing classification data...")
        session.query(DisclosureSubcategory).delete()
        session.query(DisclosureCategory).delete()
        session.commit()
        
        print("Populating disclosure_categories...")
        # Insert categories with translations
        category_map = {}
        for category_name in CATEGORIES:
            category_jp = CATEGORY_TRANSLATIONS.get(category_name)
            category = DisclosureCategory(
                name=category_name,
                name_jp=category_jp
            )
            session.add(category)
            session.flush()  # Get the ID
            category_map[category_name] = category.id
            print(f"  Added category: {category_name} (JP: {category_jp}) (ID: {category.id})")
        
        session.commit()
        
        print("\nPopulating disclosure_subcategories...")
        # Insert subcategories using SUBCATEGORY_RULES to get ALL subcategories
        subcategory_count = 0
        
        # Extract subcategory names from SUBCATEGORY_RULES (list of tuples: (pattern, subcategory))
        for _, subcategory_name in SUBCATEGORY_RULES:
            # Find the parent category for this subcategory
            if subcategory_name in SUBCATEGORY_TO_PARENT:
                parent_category = SUBCATEGORY_TO_PARENT[subcategory_name]
                if parent_category in category_map:
                    category_id = category_map[parent_category]
                    
                    # Get Japanese translation from hardcoded mapping
                    japanese_name = SUBCATEGORY_TRANSLATIONS.get(subcategory_name)
                    
                    subcategory = DisclosureSubcategory(
                        category_id=category_id,
                        name=subcategory_name,
                        name_jp=japanese_name
                    )
                    session.add(subcategory)
                    subcategory_count += 1
                    print(f"  Added subcategory: {subcategory_name} (JP: {japanese_name}) -> {parent_category}")
                else:
                    print(f"  WARNING: Parent category '{parent_category}' not found for subcategory '{subcategory_name}'")
            else:
                print(f"  WARNING: No parent category mapping found for subcategory '{subcategory_name}'")
        
        session.commit()
        
        print(f"\n{'='*80}")
        print("Classification tables populated successfully!")
        print(f"{'='*80}")
        print(f"Categories inserted: {len(CATEGORIES)}")
        print(f"Subcategories inserted: {subcategory_count}")
        print("Regex patterns remain in classification_rules.py")
        print(f"{'='*80}")
        
        # Verify the data
        print("\nVerification:")
        categories_count = session.query(DisclosureCategory).count()
        subcategories_count = session.query(DisclosureSubcategory).count()
        
        print(f"Categories in DB: {categories_count}")
        print(f"Subcategories in DB: {subcategories_count}")
        
        print(f"\nExpected counts:")
        print(f"Categories: {len(CATEGORIES)}")
        print(f"Total subcategories from SUBCATEGORY_RULES: {len(SUBCATEGORY_RULES)}")
        total_subcats_taxonomy = sum(len(subcats) for subcats in CATEGORY_TAXONOMY.values())
        print(f"Total subcategories from CATEGORY_TAXONOMY: {total_subcats_taxonomy}")
        
        if len(SUBCATEGORY_RULES) != total_subcats_taxonomy:
            print(f"WARNING: Mismatch between SUBCATEGORY_RULES ({len(SUBCATEGORY_RULES)}) and CATEGORY_TAXONOMY ({total_subcats_taxonomy})")
            print("This may indicate missing subcategories in one of the structures.")
        
    except Exception as e:
        session.rollback()
        print(f"Error populating classification tables: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    populate_classification_tables() 