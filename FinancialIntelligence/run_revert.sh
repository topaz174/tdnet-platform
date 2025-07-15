#!/bin/bash

# Script to safely revert the new embedding system changes

set -e  # Exit on any error

echo "============================================"
echo "REVERT NEW EMBEDDING SYSTEM CHANGES"
echo "============================================"
echo
echo "This will remove the following from 'disclosures' table:"
echo "  - dense_embedding column (vector data)"
echo "  - colbert_doc_embeddings column (jsonb data)"
echo "  - reasoning_context column (text data)"
echo "  - financial_entities column (jsonb data)"
echo "  - content_hash column (varchar data)"
echo "  - processing_metadata column (jsonb data)"
echo "  - Associated indexes"
echo
echo "WARNING: This will permanently delete all embedding data!"
echo

# Check if .env file exists for database connection
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please ensure your database connection is configured."
    exit 1
fi

# Ask for confirmation
read -p "Are you sure you want to proceed? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    echo "Revert cancelled."
    exit 0
fi

echo
echo "Starting revert process..."

# Load environment variables if needed
if command -v python3 &> /dev/null; then
    PG_DSN=$(python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.environ.get('PG_DSN', ''))
")
else
    echo "Python3 not found. Please ensure PG_DSN environment variable is set."
    echo "Example: export PG_DSN='postgresql://user:pass@host:port/dbname'"
    exit 1
fi

if [ -z "$PG_DSN" ]; then
    echo "Error: PG_DSN not found in environment variables."
    exit 1
fi

echo "Executing revert SQL script..."

# Execute the revert script
psql "$PG_DSN" -f revert_new_system_changes.sql

if [ $? -eq 0 ]; then
    echo
    echo "✅ Revert completed successfully!"
    echo
    echo "The following have been removed:"
    echo "  ✓ dense_embedding column and index"
    echo "  ✓ colbert_doc_embeddings column"
    echo "  ✓ reasoning_context column"
    echo "  ✓ financial_entities column and index"
    echo "  ✓ content_hash column"
    echo "  ✓ processing_metadata column"
    echo
    echo "Your database storage should now be significantly reduced."
else
    echo "❌ Revert failed. Please check the error messages above."
    exit 1
fi 