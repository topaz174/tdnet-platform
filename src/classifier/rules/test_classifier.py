#!/usr/bin/env python3
"""
Test script to classify the latest n disclosures and store results in disclosure_labels table.
This demonstrates the hybrid approach: hardcoded patterns + database storage.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import sessionmaker
from src.database.utils.init_db import Disclosure, DisclosureLabel, engine
from src.classifier.rules.classifier import classify_and_store_labels, classify_disclosure_title

# ============================================================================
# CONFIGURATION
# ============================================================================
N_LATEST_DISCLOSURES = 100  # Adjust this number as needed


def test_classify_latest_disclosures():
    """
    Classify the latest n disclosures and store results in disclosure_labels table.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print(f"Classifying Latest {N_LATEST_DISCLOSURES} Disclosures")
        print("=" * 60)
        
        # Get the latest n disclosures
        latest_disclosures = session.query(Disclosure).order_by(
            Disclosure.id.desc()
        ).limit(N_LATEST_DISCLOSURES).all()
        
        if not latest_disclosures:
            print("No disclosures found in the database!")
            return
        
        print(f"Found {len(latest_disclosures)} disclosures to classify")
        print(f"ID range: {latest_disclosures[-1].id} to {latest_disclosures[0].id}")
        print()
        
        successful_classifications = 0
        failed_classifications = 0
        
        # Process each disclosure
        for i, disclosure in enumerate(latest_disclosures, 1):
            try:
                print(f"{i:2d}. Processing ID {disclosure.id}")
                print(f"    Company: {disclosure.company_code} - {disclosure.company_name}")
                print(f"    Date: {disclosure.disclosure_date}")
                print(f"    Title: {disclosure.title[:80]}{'...' if len(disclosure.title) > 80 else ''}")
                
                # Classify and store in disclosure_labels table
                category, subcategory = classify_and_store_labels(
                    disclosure.id, 
                    disclosure.title
                )
                
                print(f"    ✅ Category: {category}")
                print(f"    ✅ Subcategory: {subcategory}")
                print(f"    ✅ Stored in disclosure_labels table")
                
                successful_classifications += 1
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                failed_classifications += 1
            
            print()  # Empty line for readability
        
        # Summary
        print("=" * 60)
        print("CLASSIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total disclosures processed: {len(latest_disclosures)}")
        print(f"Successful classifications: {successful_classifications}")
        print(f"Failed classifications: {failed_classifications}")
        print(f"Success rate: {successful_classifications/len(latest_disclosures)*100:.1f}%")
        
        # Check disclosure_labels table
        labels_count = session.query(DisclosureLabel).count()
        print(f"\nTotal entries in disclosure_labels table: {labels_count}")
        
        # Show latest entries
        latest_labels = session.query(DisclosureLabel).order_by(
            DisclosureLabel.labeled_at.desc()
        ).limit(5).all()
        
        if latest_labels:
            print(f"\nLatest 5 entries in disclosure_labels:")
            for label in latest_labels:
                print(f"  - Disclosure ID {label.disclosure_id}: "
                      f"Category {label.category_id}, Subcategory {label.subcat_id}, "
                      f"Labeled at {label.labeled_at}")
        
    except Exception as e:
        print(f"Error during classification test: {e}")
        raise
    finally:
        session.close()


def show_classification_examples():
    """
    Show some classification examples without storing in database.
    """
    print(f"\nClassification Examples (hardcoded patterns only):")
    print("-" * 50)
    
    examples = [
        "決算短信",
        "配当のお知らせ", 
        "役員異動について",
        "M&A実施について",
        "株式分割について",
        "未知のタイトル例"
    ]
    
    for i, title in enumerate(examples, 1):
        category, subcategory = classify_disclosure_title(title)
        print(f"{i}. '{title}'")
        print(f"   → {category}")
        if subcategory:
            print(f"   → {subcategory}")
        else:
            print(f"   → (No subcategory)")
        print()


def main():
    """
    Main function with options.
    """
    global N_LATEST_DISCLOSURES
    print("TDnet Classification Test")
    print("=" * 40)
    print(f"Configuration: N_LATEST_DISCLOSURES = {N_LATEST_DISCLOSURES}")
    print()
    
    while True:
        print("Options:")
        print("1. Classify latest disclosures and store in disclosure_labels")
        print("2. Show classification examples (no database storage)")
        print("3. Change N_LATEST_DISCLOSURES setting")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-3): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            test_classify_latest_disclosures()
        elif choice == "2":
            show_classification_examples()
        elif choice == "3":
            try:
                new_n = int(input(f"Enter new value for N_LATEST_DISCLOSURES (current: {N_LATEST_DISCLOSURES}): "))
                if new_n > 0:
                    N_LATEST_DISCLOSURES = new_n
                    print(f"Updated N_LATEST_DISCLOSURES to {N_LATEST_DISCLOSURES}")
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        else:
            print("Invalid choice. Please try again.")
        
        print()  # Empty line for readability


if __name__ == "__main__":
    main() 