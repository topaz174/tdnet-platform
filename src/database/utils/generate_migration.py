#!/usr/bin/env python3
"""
Migration Generation Script

This script generates a migration from the current TDnet schema to the target schema
using the following process:

1. Creates a temporary database with the target schema
2. Uses migra to compare current database with target schema  
3. Generates migration SQL
4. Creates an Alembic migration file with the generated SQL
5. Cleans up temporary resources

Usage:
    python scripts/generate_migration.py [--dry-run] [--temp-db-name DBNAME]
"""

import os
import sys
import subprocess
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import unified config
from config.config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MigrationGenerator:
    def __init__(self, temp_db_name=None, dry_run=False):
        self.temp_db_name = temp_db_name or f"tdnet_migration_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dry_run = dry_run
        
        # Database URLs
        self.source_db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        self.temp_db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{self.temp_db_name}"
        
        # File paths
        self.target_schema_file = project_root / "target_schema.sql"
        self.migration_sql_file = project_root / "migration_output.sql"
        
        logger.info(f"Source DB: {DB_CONFIG['database']}")
        logger.info(f"Temp DB: {self.temp_db_name}")
        logger.info(f"Dry run: {self.dry_run}")
    
    def create_temp_database(self):
        """Create temporary database for target schema"""
        logger.info(f"Creating temporary database: {self.temp_db_name}")
        
        # Connect to PostgreSQL server to create database
        server_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
        
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            conn = psycopg2.connect(server_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Drop database if it exists (cleanup from previous runs)
            cursor.execute(f"DROP DATABASE IF EXISTS {self.temp_db_name}")
            
            # Create new temporary database
            cursor.execute(f"CREATE DATABASE {self.temp_db_name}")
            
            cursor.close()
            conn.close()
            
            logger.info(f"✓ Created temporary database: {self.temp_db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create temporary database: {e}")
            return False
    
    def load_target_schema(self):
        """Load target schema into temporary database"""
        logger.info("Loading target schema into temporary database...")
        
        if not self.target_schema_file.exists():
            logger.error(f"Target schema file not found: {self.target_schema_file}")
            return False
        
        try:
            # Use psql to load the schema
            cmd = [
                'psql',
                self.temp_db_url,
                '-f', str(self.target_schema_file),
                '-q'  # Quiet mode
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to load target schema: {result.stderr}")
                return False
            
            logger.info("✓ Target schema loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading target schema: {e}")
            return False
    
    def generate_migration_sql(self):
        """Use migra to generate migration SQL"""
        logger.info("Generating migration SQL with migra...")
        
        try:
            # Use migra to compare schemas and generate SQL
            cmd = [
                'migra',
                self.source_db_url,  # FROM (current schema)
                self.temp_db_url,    # TO (target schema)
                '--unsafe',          # Allow potentially unsafe operations
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Migra failed: {result.stderr}")
                return None
            
            migration_sql = result.stdout.strip()
            
            if not migration_sql:
                logger.warning("No schema differences detected - migration would be empty")
                return ""
            
            # Save migration SQL to file for review
            with open(self.migration_sql_file, 'w') as f:
                f.write(migration_sql)
            
            logger.info(f"✓ Migration SQL generated: {self.migration_sql_file}")
            logger.info(f"Migration SQL length: {len(migration_sql)} characters")
            
            return migration_sql
            
        except Exception as e:
            logger.error(f"Error generating migration SQL: {e}")
            return None
    
    def create_alembic_migration(self, migration_sql):
        """Create Alembic migration file with the generated SQL"""
        logger.info("Creating Alembic migration file...")
        
        if not migration_sql.strip():
            logger.warning("Migration SQL is empty - no migration file created")
            return False
        
        # Generate timestamp for migration
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
        
        # Create migration content
        migration_content = f'''"""Migrate from TDnet schema to unified XBRL schema

This migration transforms the TDnet-focused schema to a comprehensive
XBRL processing schema with enhanced financial data capabilities.

Key changes:
- Restructure disclosures table for XBRL workflow
- Add comprehensive XBRL entities (companies, concepts, filings, etc.)
- Add financial facts storage with normalized structure
- Add classification and categorization tables
- Add materialized views for performance
- Maintain vector capabilities for qualitative data

Generated automatically by migra comparing:
- Source: Current TDnet schema (tdnet_schema_dump.sql)
- Target: Unified XBRL schema (target_schema.sql)

Revision ID: 002
Revises: 001
Create Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply migration from TDnet schema to unified XBRL schema"""
    # Generated migration SQL from migra
    op.execute("""
{migration_sql}
    """)


def downgrade() -> None:
    """
    Downgrade is not supported for this major schema transformation.
    
    This migration represents a fundamental restructuring of the database
    schema from a document-focused design to a comprehensive XBRL processing
    system. Reversing this would require complex data transformation logic
    that is beyond the scope of automated migrations.
    
    If you need to revert, consider:
    1. Restoring from a database backup taken before migration
    2. Re-initializing with the original schema (tdnet_schema_dump.sql)
    """
    raise NotImplementedError(
        "Downgrade not supported for major schema transformation. "
        "Use database backup restoration instead."
    )
'''
        
        # Write migration file
        migration_file = project_root / "src" / "database" / "migrations" / "versions" / f"{timestamp}_002_unified_xbrl_schema.py"
        
        try:
            # Ensure migrations/versions directory exists
            migration_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(migration_file, 'w') as f:
                f.write(migration_content)
            
            logger.info(f"✓ Alembic migration created: {migration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration file: {e}")
            return False
    
    def cleanup_temp_database(self):
        """Remove temporary database"""
        logger.info(f"Cleaning up temporary database: {self.temp_db_name}")
        
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            server_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
            conn = psycopg2.connect(server_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            cursor.execute(f"DROP DATABASE IF EXISTS {self.temp_db_name}")
            
            cursor.close()
            conn.close()
            
            logger.info("✓ Temporary database cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary database: {e}")
            return False
    
    def run(self):
        """Execute the complete migration generation process"""
        logger.info("=== Starting Migration Generation Process ===")
        
        try:
            # Step 1: Create temporary database
            if not self.create_temp_database():
                return False
            
            # Step 2: Load target schema
            if not self.load_target_schema():
                self.cleanup_temp_database()
                return False
            
            # Step 3: Generate migration SQL
            migration_sql = self.generate_migration_sql()
            if migration_sql is None:
                self.cleanup_temp_database()
                return False
            
            # Step 4: Create Alembic migration (unless dry run)
            if not self.dry_run:
                if not self.create_alembic_migration(migration_sql):
                    self.cleanup_temp_database()
                    return False
            else:
                logger.info("Dry run mode - skipping Alembic migration creation")
            
            # Step 5: Cleanup
            self.cleanup_temp_database()
            
            logger.info("=== Migration Generation Completed Successfully ===")
            
            if not self.dry_run:
                logger.info("\nNext steps:")
                logger.info("1. Review the generated migration file")
                logger.info("2. Review migration_output.sql for the raw SQL")
                logger.info("3. Test the migration on a backup database")
                logger.info("4. Apply with: alembic upgrade head")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration generation failed: {e}")
            self.cleanup_temp_database()
            return False


def main():
    parser = argparse.ArgumentParser(description="Generate database migration using migra")
    parser.add_argument('--dry-run', action='store_true', 
                       help="Generate SQL but don't create Alembic migration file")
    parser.add_argument('--temp-db-name', 
                       help="Name for temporary database (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Check requirements
    try:
        subprocess.run(['migra', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run([sys.executable, '-m', 'migra', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("migra is not installed or not in PATH")
        logger.error("Install with: pip install migra")
        return 1
    
    try:
        subprocess.run(['psql', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("psql is not installed or not in PATH")
        logger.error("Please install PostgreSQL client tools")
        return 1
    
    # Run migration generation
    generator = MigrationGenerator(
        temp_db_name=args.temp_db_name,
        dry_run=args.dry_run
    )
    
    success = generator.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 