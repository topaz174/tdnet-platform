"""Migration to make company_master the canonical companies table

This migration:
1. Adds a surrogate integer primary key to company_master
2. Drops the old companies table
3. Renames company_master to companies

Revision ID: 003
Revises: 002
Create Date: 2024-12-28 14:00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Make company_master the canonical companies table.
    """
    # Add surrogate integer primary key to company_master
    op.add_column('company_master', sa.Column('id', sa.Integer(), nullable=False, server_default=sa.text("nextval('company_master_id_seq'::regclass)")))
    op.create_primary_key('company_master_pkey', 'company_master', ['id'])
    
    # Create sequence for the new id column
    op.execute("CREATE SEQUENCE IF NOT EXISTS company_master_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1")
    op.execute("ALTER TABLE company_master ALTER COLUMN id SET DEFAULT nextval('company_master_id_seq'::regclass)")
    op.execute("ALTER SEQUENCE company_master_id_seq OWNED BY company_master.id")
    
    # Drop the old companies table
    op.drop_table('companies')
    
    # Rename company_master to companies
    op.rename_table('company_master', 'companies')
    
    # Rename the sequence
    op.execute("ALTER SEQUENCE company_master_id_seq RENAME TO companies_id_seq")


def downgrade() -> None:
    """
    Revert the migration by recreating the old structure.
    """
    # Rename companies back to company_master
    op.rename_table('companies', 'company_master')
    
    # Rename the sequence back
    op.execute("ALTER SEQUENCE companies_id_seq RENAME TO company_master_id_seq")
    
    # Drop the surrogate key from company_master
    op.drop_constraint('company_master_pkey', 'company_master', type_='primary')
    op.drop_column('company_master', 'id')
    
    # Recreate the old companies table (basic structure)
    op.create_table('companies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('company_code', sa.String(length=5), nullable=False),
        sa.Column('name_en', sa.Text(), nullable=False),
        sa.Column('name_ja', sa.Text(), nullable=True),
        sa.Column('exchange_id', sa.Integer(), nullable=True),
        sa.Column('sector_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('company_code')
    )