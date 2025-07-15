import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import re
import csv
import sys
import os
from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.init_db import (
    Disclosure, DisclosureCategory, DisclosureSubcategory, 
    DisclosureLabel, engine
)
from src.classifier.classification_rules import SUBCATEGORY_RULES, PARENT_CATEGORY_RULES, SUBCATEGORY_TO_PARENT


# ============================================================================
# CORE CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_disclosure_title(title):
    """
    Classify a single disclosure title into parent and subcategories.
    Uses hardcoded rules from classification_rules.py for performance.
    
    Args:
        title (str): The disclosure title to classify
        
    Returns:
        tuple: (parent_categories_str, subcategories_str)
            - parent_categories_str: Comma-separated string of parent categories
            - subcategories_str: Comma-separated string of subcategories (empty if "Other")
    """
    # First, find subcategories using hardcoded rules
    matched_subcategories = []
    for pattern, subcat in SUBCATEGORY_RULES:
        if re.search(pattern, title):
            matched_subcategories.append(subcat)
    
    # If no subcategories found, this will be classified as "Other"
    is_other = len(matched_subcategories) == 0
    
    # Get parent categories from subcategories
    parent_categories = []
    for subcat in matched_subcategories:
        if subcat in SUBCATEGORY_TO_PARENT:
            parent_cat = SUBCATEGORY_TO_PARENT[subcat]
            if parent_cat not in parent_categories:
                parent_categories.append(parent_cat)
    
    # Apply parent category rules directly (for broader matching)
    for pattern, parent_cat in PARENT_CATEGORY_RULES:
        if re.search(pattern, title) and parent_cat not in parent_categories:
            parent_categories.append(parent_cat)
    
    # If still no parent categories, mark as OTHER
    if not parent_categories:
        parent_categories = ["OTHER"]
    
    # For "Other" items, don't include subcategories
    if is_other:
        subcategories_str = ""
    else:
        subcategories_str = ", ".join(matched_subcategories)
    
    parent_categories_str = ", ".join(parent_categories)
    
    return parent_categories_str, subcategories_str


def classify_and_store_labels(disclosure_id, title):
    """
    Classify a disclosure and store ALL results in the disclosure_labels table.
    Supports multiple categories and subcategories per disclosure.
    
    Args:
        disclosure_id (int): The disclosure ID
        title (str): The disclosure title  
        
    Returns:
        tuple: (category_str, subcategory_str) - the classification results
    """
    import re
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Classify the title
        category_str, subcategory_str = classify_disclosure_title(title)
        
        # Parse categories and subcategories
        # Categories: safe to split by ', ' (category names don't include commas)
        categories = [cat.strip() for cat in category_str.split(', ') if cat.strip()]

        # Subcategories: names themselves may contain commas, so avoid naive splitting.
        # Instead, scan for known subcategory names within the returned string.
        subcategories = []
        if subcategory_str:
            for subcat_name in SUBCATEGORY_TO_PARENT.keys():
                if subcat_name in subcategory_str:
                    subcategories.append(subcat_name)
        # Ensure deterministic ordering and uniqueness
        subcategories = list(dict.fromkeys(subcategories))
        
        # Look up category and subcategory IDs
        category_id_map = {}
        for cat_name in categories:
            cat = session.query(DisclosureCategory).filter_by(name=cat_name).first()
            if cat:
                category_id_map[cat_name] = cat.id
        
        subcat_id_map = {}
        for subcat_name in subcategories:
            subcat = session.query(DisclosureSubcategory).filter_by(name=subcat_name).first()
            if subcat:
                subcat_id_map[subcat_name] = subcat.id
        
        # Clear existing labels for this disclosure
        session.query(DisclosureLabel).filter_by(disclosure_id=disclosure_id).delete()
        
        # Create new labels for all category/subcategory combinations
        labels_to_add = []
        
        if not subcategories:
            # No subcategories - create one label per category
            for cat_name in categories:
                if cat_name in category_id_map:
                    labels_to_add.append(DisclosureLabel(
                        disclosure_id=disclosure_id,
                        category_id=category_id_map[cat_name],
                        subcat_id=None
                    ))
        else:
            # Have subcategories - create labels for each subcategory 
            # (categories are derived from subcategories)
            for subcat_name in subcategories:
                if subcat_name in subcat_id_map:
                    # Find the parent category for this subcategory
                    subcat_obj = session.query(DisclosureSubcategory).filter_by(name=subcat_name).first()
                    if subcat_obj:
                        labels_to_add.append(DisclosureLabel(
                            disclosure_id=disclosure_id,
                            category_id=subcat_obj.category_id,
                            subcat_id=subcat_obj.id
                        ))
        
        # Add all labels
        for label in labels_to_add:
            session.add(label)
        
        session.commit()
        return category_str, subcategory_str
        
    except Exception as e:
        session.rollback()
        print(f"Error storing classification for disclosure {disclosure_id}: {e}")
        # Still return classification results even if storage fails
        return classify_disclosure_title(title)
    finally:
        session.close()


def classify_titles_batch(titles):
    """
    Classify multiple titles and return statistics.
    
    Args:
        titles (list): List of titles to classify
        
    Returns:
        tuple: (classifications, stats)
            - classifications: List of dicts with title, parent_categories, subcategories
            - stats: Dictionary with classification statistics
    """
    classified = []
    other_count = 0
    total_count = len(titles)

    for title in titles:
        parent_categories_str, subcategories_str = classify_disclosure_title(title)
        
        # Count "Other" classifications
        if parent_categories_str == "OTHER":
            other_count += 1
        
        classified.append({
            "title": title, 
            "parent_categories": parent_categories_str,
            "subcategories": subcategories_str
        })

    # Calculate statistics
    properly_classified = total_count - other_count
    percent_other = (other_count / total_count * 100) if total_count > 0 else 0
    
    stats = {
        "total": total_count,
        "properly_classified": properly_classified,
        "other_count": other_count,
        "percent_other": percent_other,
        "percent_classified": 100 - percent_other if total_count > 0 else 0
    }
    
    return classified, stats


# ============================================================================
# DATABASE CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_all_existing_disclosures():
    """
    Classify all existing disclosures in the database and store results in disclosure_labels table.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all disclosure IDs that don't have entries in disclosure_labels
        classified_ids = session.query(DisclosureLabel.disclosure_id).distinct().subquery()
        unclassified = session.query(Disclosure).filter(
            ~Disclosure.id.in_(classified_ids)  # type: ignore
        ).order_by(Disclosure.disclosure_date.desc()).all()
        
        total_count = len(unclassified)
        print(f"Found {total_count} unclassified disclosures to process...")
        print(f"Processing order: disclosure_date DESC (newest first)")
        
        if total_count == 0:
            print("All disclosures are already classified!")
            return
        
        classified_count = 0
        batch_size = 10000
        
        for i, disclosure in enumerate(unclassified, 1):
            try:
                classify_and_store_labels(disclosure.id, disclosure.title)
                classified_count += 1
                
                # Print progress less frequently for speed
                if i % 5000 == 0 or i == total_count:
                    print(f"Progress: {i}/{total_count} ({i/total_count*100:.1f}%)")
                
                # Commit in larger batches for speed
                if i % batch_size == 0:
                    session.commit()
            
            except Exception as e:
                print(f"Error classifying disclosure {disclosure.id}: {e}")
                continue
        
        # Final commit
        session.commit()
        
        print(f"\n{'='*80}")
        print(f"Classification Complete!")
        print(f"{'='*80}")
        print(f"Total disclosures processed: {total_count}")
        print(f"Successfully classified: {classified_count}")
        print(f"Errors: {total_count - classified_count}")
        print(f"{'='*80}")
        
        # Show classification summary
        show_classification_summary(session)
        
    except Exception as e:
        session.rollback()
        print(f"Error during classification: {e}")
        raise
    finally:
        session.close()


def classify_by_ids(disclosure_ids):
    """
    Classify specific disclosures by their IDs.
    
    Args:
        disclosure_ids (list): List of disclosure IDs to classify
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        disclosures = session.query(Disclosure).filter(
            Disclosure.id.in_(disclosure_ids)
        ).order_by(Disclosure.disclosure_date.desc()).all()
        
        total_count = len(disclosures)
        print(f"Found {total_count} disclosures to classify...")
        print(f"Processing order: disclosure_date DESC (newest first)")
        
        if total_count == 0:
            print("No disclosures found with the given IDs!")
            return
        
        classified_count = 0
        
        for i, disclosure in enumerate(disclosures, 1):
            try:
                category, subcategory = classify_and_store_labels(disclosure.id, disclosure.title)
                classified_count += 1
                
                print(f"Progress: {i}/{total_count} - "
                      f"ID: {disclosure.id}, Company: {disclosure.company_code}, "
                      f"Category: {category}")
            
            except Exception as e:
                print(f"Error classifying disclosure {disclosure.id}: {e}")
                continue
        
        # Commit all changes
        session.commit()
        
        print(f"\n{'='*50}")
        print(f"Classification Complete!")
        print(f"{'='*50}")
        print(f"Total disclosures processed: {total_count}")
        print(f"Successfully classified: {classified_count}")
        print(f"Errors: {total_count - classified_count}")
        
    except Exception as e:
        session.rollback()
        print(f"Error during classification: {e}")
        raise
    finally:
        session.close()


def classify_specific_date_range(start_date, end_date):
    """
    Classify disclosures within a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get disclosures in the date range
        disclosures = session.query(Disclosure).filter(
            Disclosure.disclosure_date >= start_date,
            Disclosure.disclosure_date <= end_date
        ).order_by(Disclosure.disclosure_date.desc()).all()
        
        total_count = len(disclosures)
        print(f"Found {total_count} disclosures between {start_date} and {end_date}...")
        print(f"Processing order: disclosure_date DESC (newest first)")
        
        if total_count == 0:
            print("No disclosures found in the given date range!")
            return
        
        classified_count = 0
        batch_size = 10000
        
        for i, disclosure in enumerate(disclosures, 1):
            try:
                classify_and_store_labels(disclosure.id, disclosure.title)
                classified_count += 1
                
                # Print progress less frequently for speed
                if i % 5000 == 0 or i == total_count:
                    print(f"Progress: {i}/{total_count} ({i/total_count*100:.1f}%)")
                
                # Commit in larger batches for speed
                if i % batch_size == 0:
                    session.commit()
            
            except Exception as e:
                print(f"Error classifying disclosure {disclosure.id}: {e}")
                continue
        
        # Final commit
        session.commit()
        
        print(f"\n{'='*80}")
        print(f"Classification Complete!")
        print(f"{'='*80}")
        print(f"Total disclosures processed: {total_count}")
        print(f"Successfully classified: {classified_count}")
        print(f"Errors: {total_count - classified_count}")
        
    except Exception as e:
        session.rollback()
        print(f"Error during classification: {e}")
        raise
    finally:
        session.close()


def show_classification_summary(session):
    """
    Show a summary of classification statistics.
    """
    # Get all disclosures and labeled disclosures
    total_disclosures = session.query(Disclosure).count()
    labeled_disclosure_ids = session.query(DisclosureLabel.disclosure_id).distinct().subquery()
    classified_disclosures = session.query(Disclosure).filter(
        Disclosure.id.in_(labeled_disclosure_ids)
    ).count()
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total disclosures: {total_disclosures:,}")
    print(f"Classified disclosures: {classified_disclosures:,}")
    print(f"Unclassified disclosures: {total_disclosures - classified_disclosures:,}")
    print(f"Classification rate: {classified_disclosures/total_disclosures*100:.1f}%")
    
    # Category breakdown from disclosure_labels
    print(f"\n{'='*40}")
    print(f"CATEGORY BREAKDOWN")
    print(f"{'='*40}")
    
    # Query for category distribution
    category_counts = session.query(
        DisclosureCategory.name, func.count(DisclosureLabel.id)
    ).join(DisclosureLabel).group_by(
        DisclosureCategory.name
    ).order_by(func.count(DisclosureLabel.id).desc()).all()
    
    total_labels = sum(count for _, count in category_counts)
    
    for category, count in category_counts:
        percentage = (count / total_labels * 100) if total_labels > 0 else 0
        print(f"{category}: {count:,} ({percentage:.1f}%)")
    
    # Show total labels
        print(f"\n{'='*40}")
        print(f"DISCLOSURE LABELS")
        print(f"{'='*40}")
        print(f"Total labels: {total_labels:,}")


# ============================================================================
# LEGACY WEB SCRAPING FUNCTIONS (UNCHANGED)
# ============================================================================

def scrape_tdnet_titles(date_str, start_hour=None, end_hour=None):
    """
    Scrape TDnet disclosure titles for a specific date.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format
        start_hour (int, optional): Start hour (0-23) for filtering
        end_hour (int, optional): End hour (0-23) for filtering
        
    Returns:
        list: List of disclosure titles
    """
    # Convert date format for URL
    date_formatted = date_str.replace('-', '')
    
    # TDnet search URL
    url = f"https://www.release.tdnet.info/inbs/I_list_{date_formatted}_1.html"
    
    print(f"Scraping TDnet for date: {date_str}")
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all disclosure rows
        titles = []
        rows = soup.find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')  # type: ignore
            if len(cells) >= 5:  # Ensure we have enough columns
                try:
                    # Extract information
                    time_str = cells[0].get_text(strip=True) if cells[0] else ""
                    company_code = cells[1].get_text(strip=True) if cells[1] else ""
                    company_name = cells[2].get_text(strip=True) if cells[2] else ""
                    title = cells[3].get_text(strip=True) if cells[3] else ""
                    
                    # Filter by time if specified
                    if start_hour is not None or end_hour is not None:
                        if time_str and ':' in time_str:
                            try:
                                hour = int(time_str.split(':')[0])
                                if start_hour is not None and hour < start_hour:
                                    continue
                                if end_hour is not None and hour > end_hour:
                                    continue
                            except (ValueError, IndexError):
                                continue  # Skip if we can't parse the time
                    
                    # Add title if it's not empty and has meaningful content
                    if title and len(title.strip()) > 0:
                        titles.append(title)
                        
                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
        
        print(f"Found {len(titles)} titles for {date_str}")
        return titles
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []


def classify_titles(titles):
    """
    Classify a list of titles and return results.
    
    Args:
        titles (list): List of disclosure titles
        
    Returns:
        tuple: (DataFrame with results, statistics dict)
    """
    print(f"Classifying {len(titles)} titles...")
    
    classified, stats = classify_titles_batch(titles)
    
    # Create DataFrame
    df = pd.DataFrame(classified)
    
    print(f"\nClassification Results:")
    print(f"Total titles: {stats['total']}")
    print(f"Properly classified: {stats['properly_classified']} ({stats['percent_classified']:.1f}%)")
    print(f"Classified as 'Other': {stats['other_count']} ({stats['percent_other']:.1f}%)")
    
    return df, stats


def save_results(df, stats, titles_file="input.md", results_file="output.xlsx"):
    """
    Save classification results to files.
    
    Args:
        df (DataFrame): Classification results
        stats (dict): Classification statistics
        titles_file (str): Input titles file name
        results_file (str): Output results file name
    """
    # Save input titles
    print(f"\nSaving input titles to {titles_file}...")
    with open(titles_file, 'w', encoding='utf-8') as f:
        f.write("# Disclosure Titles for Classification\n\n")
        for i, title in enumerate(df['title'], 1):
            f.write(f"{i}. {title}\n")
    
    # Save results to Excel
    print(f"Saving results to {results_file}...")
    
    # Create summary sheet
    summary_data = {
        'Metric': ['Total Titles', 'Properly Classified', 'Classified as Other', 'Classification Rate'],
        'Value': [stats['total'], stats['properly_classified'], stats['other_count'], f"{stats['percent_classified']:.1f}%"]
    }
    summary_df = pd.DataFrame(summary_data)
    
    with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
        # Write summary
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write detailed results
        df.to_excel(writer, sheet_name='Classifications', index=False)
        
        # Write category breakdown
        category_counts = df['parent_categories'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        category_counts['Percentage'] = (category_counts['Count'] / len(df) * 100).round(1)
        category_counts.to_excel(writer, sheet_name='Category Breakdown', index=False)
    
    print(f"Results saved successfully!")


def save_ml_training_data(df, rules, output_file="ml_training_data.json"):
    """
    Save classification results in a format suitable for ML training.
    
    Args:
        df (DataFrame): Classification results
        rules (dict): Classification rules used
        output_file (str): Output JSON file name
    """
    print(f"Saving ML training data to {output_file}...")
    
    # Prepare training data
    training_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(df),
            "classification_method": "rule_based",
            "rules_version": "1.0"
        },
        "samples": []
    }
    
    for _, row in df.iterrows():
        sample = {
            "text": row['title'],
            "labels": {
                "parent_categories": row['parent_categories'].split(', ') if row['parent_categories'] else [],
                "subcategories": row['subcategories'].split(', ') if row['subcategories'] else []
            }
        }
        training_data["samples"].append(sample)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"ML training data saved successfully!")
    print(f"Total samples: {len(training_data['samples'])}")


def main():
    """
    Main function for interactive classification.
    """
    print("TDnet Disclosure Classifier")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Scrape and classify titles for a specific date")
        print("2. Classify all unclassified disclosures in database")
        print("3. Classify disclosures by specific IDs")
        print("4. Classify disclosures in date range")
        print("5. Show classification summary")
        print("6. Test classification on custom text")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            start_hour = input("Enter start hour (0-23, or press Enter for all): ").strip()
            end_hour = input("Enter end hour (0-23, or press Enter for all): ").strip()
            
            # Convert hour inputs
            start_hour = int(start_hour) if start_hour else None
            end_hour = int(end_hour) if end_hour else None
            
            # Scrape titles
            titles = scrape_tdnet_titles(date_str, start_hour, end_hour)
            
            if titles:
                # Classify titles
                df, stats = classify_titles(titles)
                
                # Save results
                titles_file = f"titles_{date_str.replace('-', '')}.md"
                results_file = f"classification_results_{date_str.replace('-', '')}.xlsx"
                save_results(df, stats, titles_file, results_file)
                
                # Save ML training data
                ml_file = f"ml_training_data_{date_str.replace('-', '')}.json"
                save_ml_training_data(df, {}, ml_file)
            else:
                print("No titles found for the specified date and time range.")
                
        elif choice == "2":
            classify_all_existing_disclosures()
        elif choice == "3":
            ids_input = input("Enter disclosure IDs (comma-separated): ").strip()
            try:
                ids = [int(id.strip()) for id in ids_input.split(',')]
                classify_by_ids(ids)
            except ValueError:
                print("Invalid input. Please enter numeric IDs separated by commas.")
        elif choice == "4":
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            classify_specific_date_range(start_date, end_date)
        elif choice == "5":
            Session = sessionmaker(bind=engine)
            session = Session()
            try:
                show_classification_summary(session)
            finally:
                session.close()
        elif choice == "6":
            while True:
                print("\n" + "="*60)
                print("TEST CLASSIFICATION ON CUSTOM TEXT")
                print("="*60)
                test_text = input("\nEnter text to classify (or 'back' to return to main menu): ").strip()
                
                if test_text.lower() == 'back':
                    break
                
                if not test_text:
                    print("Please enter some text to classify.")
                    continue
                
                # Classify the text
                parent_categories, subcategories = classify_disclosure_title(test_text)
                
                # Display results
                print(f"\nInput text: {test_text}")
                print("-" * 60)
                print(f"Parent Categories: {parent_categories}")
                print(f"Subcategories: {subcategories if subcategories else 'None'}")
                
                # Show which rules matched
                print(f"\nMatching Analysis:")
                print("-" * 30)
                
                # Check subcategory matches
                matched_subcats = []
                for pattern, subcat in SUBCATEGORY_RULES:
                    if re.search(pattern, test_text):
                        matched_subcats.append((subcat, pattern))
                
                if matched_subcats:
                    print("Subcategory matches:")
                    for subcat, pattern in matched_subcats:
                        print(f"  • {subcat}")
                        print(f"    Pattern: {pattern[:100]}{'...' if len(pattern) > 100 else ''}")
                else:
                    print("No subcategory patterns matched")
                
                # Check parent category matches
                matched_parents = []
                for pattern, parent in PARENT_CATEGORY_RULES:
                    if re.search(pattern, test_text):
                        matched_parents.append((parent, pattern))
                
                if matched_parents:
                    print("\nParent category matches:")
                    for parent, pattern in matched_parents:
                        print(f"  • {parent}")
                        print(f"    Pattern: {pattern}")
                else:
                    print("\nNo parent category patterns matched")
                
                print("\n" + "="*60)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()