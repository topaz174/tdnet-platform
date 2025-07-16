"""Baseline migration representing current TDnet schema

This migration represents the current state of the TDnet database schema
as captured in tdnet_schema_dump.sql. It serves as the baseline for all
future migrations.

Current schema includes:
- company_master table (qualitative company data with vector support)
- disclosures table (financial disclosures with embeddings and extraction tracking)
- document_chunks table (chunked document content with vector embeddings)
- reports table (basic reports linking)
- PostgreSQL extensions: uuid-ossp, vector
- Comprehensive indexing including vector indexes (hnsw, ivfflat)

Revision ID: 001
Revises: 
Create Date: 2024-12-28 12:00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    This is a baseline migration - the schema already exists.
    
    To use this migration system:
    1. Ensure your database matches tdnet_schema_dump.sql
    2. Run: alembic stamp 001
    3. This will mark the database as being at this baseline
    4. Future migrations can then be applied normally
    """
    pass


def downgrade() -> None:
    """
    Cannot downgrade from baseline - this would require dropping the entire schema.
    If you need to reset, use the database reset tools instead.
    """
    raise NotImplementedError("Cannot downgrade from baseline migration. Use database reset tools if needed.") 