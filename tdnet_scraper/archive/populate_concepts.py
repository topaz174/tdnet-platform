#!/usr/bin/env python3
"""
Populate concepts and concept_tags tables from existing concepts.json file.
This is a one-time migration script to move from JSON-based to database-based concept lookup.
"""

import json
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


class ConceptsPopulator:
    """Populates concepts and concept_tags tables from concepts.json."""
    
    def __init__(self):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.xsd_types = self._load_xsd_types()
    
    def _load_xsd_types(self) -> dict:
        """Load XSD element types mapping."""
        xsd_file = Path(__file__).parent / 'xsd_elements_organized.json'
        try:
            with open(xsd_file, 'r', encoding='utf-8') as f:
                xsd_data = json.load(f)
                # Create reverse mapping: element -> type
                element_to_type = {}
                for type_name, elements in xsd_data.items():
                    for element in elements:
                        element_to_type[element] = type_name
                return element_to_type
        except FileNotFoundError:
            print(f"Warning: {xsd_file} not found")
            return {}
    
    def _determine_concept_type(self, taxonomy_elements: list) -> str:
        """Determine the concept type based on its taxonomy elements."""
        # Count occurrences of each type among the taxonomy elements
        type_counts = {}
        for element in taxonomy_elements:
            element_type = self.xsd_types.get(element)
            if element_type:
                # Remove "ItemType" suffix to get clean type name
                clean_type = element_type.replace('ItemType', '') if element_type.endswith('ItemType') else element_type
                type_counts[clean_type] = type_counts.get(clean_type, 0) + 1
        
        if not type_counts:
            return None  # No known types found
        
        # Return the most common type
        return max(type_counts, key=type_counts.get)
    
    def populate_from_json(self) -> None:
        """Populate concepts and concept_tags from concepts.json."""
        concepts_file = Path(__file__).parent / 'concepts.json'
        
        if not concepts_file.exists():
            print(f"Error: {concepts_file} not found")
            return
        
        with open(concepts_file, 'r', encoding='utf-8') as f:
            concepts_data = json.load(f)
        
        session = self.Session()
        concepts_inserted = 0
        tags_inserted = 0
        
        try:
            for canonical_name_ja, concept_info in concepts_data.items():
                # Extract English name and taxonomy elements from the new structure
                canonical_name_en = concept_info['name_en']
                taxonomy_elements = concept_info['taxonomy_elements']
                
                concept_type = self._determine_concept_type(taxonomy_elements)
                
                # Check if concept already exists
                check_concept = text("""
                    SELECT id FROM concepts WHERE name_ja = :name_ja
                """)
                result = session.execute(check_concept, {'name_ja': canonical_name_ja})
                existing_concept = result.fetchone()
                
                if existing_concept:
                    concept_id = existing_concept.id
                    # Update type if it's missing
                    if concept_type:
                        update_type = text("""
                            UPDATE concepts SET type = :type WHERE id = :id AND type IS NULL
                        """)
                        session.execute(update_type, {'type': concept_type, 'id': concept_id})
                    print(f"Concept already exists: {canonical_name_ja} (ID: {concept_id}, Type: {concept_type})")
                else:
                    # Insert new concept
                    insert_concept = text("""
                        INSERT INTO concepts (name_ja, name_en, type)
                        VALUES (:name_ja, :name_en, :type)
                        RETURNING id
                    """)
                    result = session.execute(insert_concept, {
                        'name_ja': canonical_name_ja,
                        'name_en': canonical_name_en,
                        'type': concept_type
                    })
                    concept_id = result.fetchone().id
                    concepts_inserted += 1
                    print(f"Inserted concept: {canonical_name_ja} -> {canonical_name_en} (ID: {concept_id}, Type: {concept_type})")
                
                # Insert concept tags
                for taxonomy_element in taxonomy_elements:
                    try:
                        insert_tag = text("""
                            INSERT INTO concept_tags (raw_tag, concept_id)
                            VALUES (:raw_tag, :concept_id)
                            ON CONFLICT (raw_tag) DO NOTHING
                        """)
                        session.execute(insert_tag, {
                            'raw_tag': taxonomy_element,
                            'concept_id': concept_id,
                        })
                        tags_inserted += 1
                    except Exception as e:
                        print(f"Warning: Could not insert tag {taxonomy_element}: {e}")
                        continue
            
            session.commit()
            print(f"\nPopulation complete:")
            print(f"- Concepts inserted: {concepts_inserted}")
            print(f"- Tags inserted: {tags_inserted}")
            
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
        finally:
            session.close()


def main():
    """Main function."""
    print("Starting concepts population from concepts.json...")
    
    populator = ConceptsPopulator()
    populator.populate_from_json()
    
    print("\nConcepts population completed.")


if __name__ == "__main__":
    main() 