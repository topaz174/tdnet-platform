"""Migration to unify schemas

This migration applies the schema unification by executing the SQL script
from scripts/unify_schemas.sql. This adds new tables, columns, and relationships
to support the unified XBRL schema while preserving existing data.

Revision ID: 002
Revises: 001
Create Date: 2024-12-28 13:00:00

"""
from alembic import op
import sqlalchemy as sa
from pathlib import Path
import os


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Execute the unify_schemas.sql script to add new schema elements.
    """
    # Get the path to the SQL file relative to the project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sql_file_path = project_root / 'scripts' / 'unify_schemas.sql'
    
    if not sql_file_path.exists():
        raise FileNotFoundError(f"SQL file not found at {sql_file_path}")
    
    # Read and execute the SQL file
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Execute the SQL content
    op.execute(sql_content)


def downgrade() -> None:
    """
    Cannot easily downgrade this migration as it involves complex schema changes.
    If you need to revert, use database reset tools instead.
    """
    raise NotImplementedError("Cannot downgrade this migration. Use database reset tools if needed.")
