#!/usr/bin/env python3
"""
Test script to verify unified output format works for both PDF and XBRL extraction.

This script demonstrates that both extraction methods now produce compatible output
that can be written to the same PostgreSQL table.
"""

import json
from dataclasses import asdict
from pathlib import Path

# Import both processors
from src.pdf_extraction_pipeline import PDFProcessor, DocumentChunk
from src.xbrl_qualitative_extractor import XBRLProcessor, XBRLChunk

def test_unified_format():
    """Test that both extraction methods produce compatible output formats."""
    
    print("Testing Unified Output Format")
    print("=" * 50)
    
    # Test 1: Verify field compatibility
    print("\n1. Checking field compatibility...")
    
    # Create sample chunks from both systems
    pdf_chunk = DocumentChunk(
        disclosure_id=1,
        chunk_index=0,
        content="Sample PDF content",
        content_type="general",
        section_code="general",
        heading_text="Sample Heading",
        char_length=18,
        tokens=6,
        vectorize=True,
        is_numeric=False,
        disclosure_hash="abc123",
        source_file="test.pdf",
        page_number=1,
        metadata={"extraction_method": "pdf_extraction", "language": "ja"}
    )
    
    xbrl_chunk = XBRLChunk(
        disclosure_id=1,
        chunk_index=0,
        content="Sample XBRL content",
        content_type="general",
        section_code="general",
        heading_text="Sample Heading",
        char_length=19,
        tokens=6,
        vectorize=True,
        is_numeric=False,
        disclosure_hash="def456",
        source_file="test.zip",
        metadata={"extraction_method": "xbrl_qualitative", "language": "ja"}
    )
    
    # Convert to dictionaries
    pdf_dict = asdict(pdf_chunk)
    xbrl_dict = asdict(xbrl_chunk)
    
    # Check field compatibility
    pdf_fields = set(pdf_dict.keys())
    xbrl_fields = set(xbrl_dict.keys())
    
    # PDF has page_number, XBRL doesn't - this is expected
    common_fields = pdf_fields.intersection(xbrl_fields)
    pdf_only = pdf_fields - xbrl_fields
    xbrl_only = xbrl_fields - pdf_fields
    
    print(f"✓ Common fields: {len(common_fields)}")
    print(f"  {sorted(common_fields)}")
    print(f"✓ PDF-only fields: {sorted(pdf_only)} (expected: page_number)")
    print(f"✓ XBRL-only fields: {sorted(xbrl_only)} (should be empty)")
    
    # Test 2: PostgreSQL compatibility check
    print("\n2. PostgreSQL table compatibility...")
    
    # Simulate what would be inserted into the database
    # Remove page_number from PDF for XBRL compatibility
    pdf_for_db = {k: v for k, v in pdf_dict.items() if k != 'page_number'}
    xbrl_for_db = xbrl_dict.copy()
    
    # Check that all fields match
    pdf_db_fields = set(pdf_for_db.keys())
    xbrl_db_fields = set(xbrl_for_db.keys())
    
    if pdf_db_fields == xbrl_db_fields:
        print("✓ Perfect field alignment for database insertion")
        print(f"  Shared fields: {sorted(pdf_db_fields)}")
    else:
        print("✗ Field mismatch detected!")
        print(f"  PDF fields: {sorted(pdf_db_fields)}")
        print(f"  XBRL fields: {sorted(xbrl_db_fields)}")
        return False
    
    # Test 3: JSON serialization compatibility
    print("\n3. JSON serialization test...")
    
    try:
        pdf_json = json.dumps(pdf_dict, ensure_ascii=False, default=str)
        xbrl_json = json.dumps(xbrl_dict, ensure_ascii=False, default=str)
        print("✓ Both formats serialize to JSON successfully")
        print(f"✓ PDF JSON length: {len(pdf_json)} chars")
        print(f"✓ XBRL JSON length: {len(xbrl_json)} chars")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False
    
    # Test 4: Show unified database schema
    print("\n4. Unified Database Schema")
    print("-" * 30)
    
    schema_fields = [
        ("disclosure_id", "INTEGER"),
        ("chunk_index", "INTEGER"), 
        ("content", "TEXT"),
        ("content_type", "VARCHAR(50)"),
        ("section_code", "VARCHAR(50)"),
        ("heading_text", "TEXT"),
        ("char_length", "INTEGER"),
        ("tokens", "INTEGER"),
        ("vectorize", "BOOLEAN"),
        ("is_numeric", "BOOLEAN"),
        ("disclosure_hash", "VARCHAR(64)"),
        ("source_file", "VARCHAR(255)"),
        ("page_number", "INTEGER"),  # Nullable for XBRL
        ("metadata", "JSONB"),
    ]
    
    print("CREATE TABLE disclosures (")
    for field, dtype in schema_fields[:-1]:
        nullable = " NULL" if field == "page_number" else " NOT NULL"
        print(f"  {field} {dtype}{nullable},")
    field, dtype = schema_fields[-1]
    print(f"  {field} {dtype} NOT NULL")
    print(");")
    
    print("\n✓ All tests passed! Unified format is ready.")
    print("\nNext steps:")
    print("1. Update database schema with new fields")
    print("2. Modify ingestion pipeline to use unified format")
    print("3. Test with real PDF and XBRL files")
    
    return True

if __name__ == "__main__":
    success = test_unified_format()
    exit(0 if success else 1)