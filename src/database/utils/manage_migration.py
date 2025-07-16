#!/usr/bin/env python3
"""
Database Migration Management Script

This script provides a complete workflow for managing database schema migrations
from the current TDnet schema to the target unified XBRL schema.

Commands:
    init        - Initialize Alembic and stamp baseline
    generate    - Generate migration using migra
    validate    - Validate current database state
    apply       - Apply migrations
    status      - Show migration status
    backup      - Create database backup before migration

Usage:
    python scripts/manage_migration.py <command> [options]
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Database configuration imports
try:
    from tdnet_scraper.config.config import DB_CONFIG, DB_URL
    db_config = DB_CONFIG
    db_url = DB_URL
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    
    db_config = {
        'user': os.getenv('TDNET_DB_USER', os.getenv('DB_USER', 'alex')),
        'password': os.getenv('TDNET_DB_PASSWORD', os.getenv('DB_PASSWORD', 'alex')),
        'host': os.getenv('TDNET_DB_HOST', os.getenv('DB_HOST', 'localhost')),
        'port': os.getenv('TDNET_DB_PORT', os.getenv('DB_PORT', '5432')),
        'database': os.getenv('TDNET_DB_NAME', os.getenv('DB_NAME', 'tdnet'))
    }
    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MigrationManager:
    def __init__(self):
        self.db_url = db_url
        self.db_config = db_config
        self.project_root = project_root
        
    def validate_environment(self):
        """Validate that all required tools and configurations are available"""
        logger.info("Validating environment...")
        
        errors = []
        
        # Check database connection
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            conn.close()
            logger.info("✓ Database connection successful")
        except Exception as e:
            errors.append(f"Database connection failed: {e}")
        
        # Check required tools
        tools = ['alembic', 'psql', 'pg_dump']
        for tool in tools:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
                logger.info(f"✓ {tool} available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                errors.append(f"{tool} not found in PATH")
        
        # Check migra specifically (it might be in venv)
        try:
            # Try direct migra command
            subprocess.run(['migra', '--help'], capture_output=True, check=True)
            logger.info("✓ migra available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try with python -m migra
                subprocess.run([sys.executable, '-m', 'migra', '--help'], capture_output=True, check=True)
                logger.info("✓ migra available (via python -m)")
            except (subprocess.CalledProcessError, FileNotFoundError):
                errors.append("migra not found in PATH or as Python module")
        
        # Check required files
        required_files = [
            self.project_root / "alembic.ini",
            self.project_root / "target_schema.sql",
            self.project_root / "tdnet_schema_dump.sql",
            self.project_root / "src" / "database" / "migrations" / "env.py"
        ]
        
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"✓ {file_path.name} exists")
            else:
                errors.append(f"Required file missing: {file_path}")
        
        if errors:
            logger.error("Environment validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✓ Environment validation passed")
        return True
    
    def init_alembic(self):
        """Initialize Alembic and stamp baseline migration"""
        logger.info("Initializing Alembic and stamping baseline...")
        
        try:
            # Check if already initialized
            result = subprocess.run(['alembic', 'current'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info("Alembic already initialized")
                logger.info(f"Current revision: {result.stdout.strip()}")
                return True
            
            # Stamp the baseline revision
            logger.info("Stamping baseline revision 001...")
            result = subprocess.run(['alembic', 'stamp', '001'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Failed to stamp baseline: {result.stderr}")
                return False
            
            logger.info("✓ Alembic initialized and baseline stamped")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Alembic: {e}")
            return False
    
    def generate_migration(self, dry_run=False):
        """Generate migration using the migration generation script"""
        logger.info("Generating migration...")
        
        script_path = self.project_root / "src" / "database" / "utils" / "generate_migration.py"
        cmd = [sys.executable, str(script_path)]
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error generating migration: {e}")
            return False
    
    def show_status(self):
        """Show current migration status"""
        logger.info("Migration Status")
        logger.info("================")
        
        try:
            # Current revision
            result = subprocess.run(['alembic', 'current'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                current = result.stdout.strip()
                logger.info(f"Current revision: {current if current else 'None (uninitialized)'}")
            else:
                logger.info("Current revision: Unknown (Alembic error)")
            
            # Available revisions
            result = subprocess.run(['alembic', 'history'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("Available revisions:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            # Check for pending migrations
            result = subprocess.run(['alembic', 'heads'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                heads = result.stdout.strip()
                logger.info(f"Head revision: {heads}")
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")
    
    def create_backup(self, backup_name=None):
        """Create database backup before migration using the existing backup system"""
        try:
            # Import the existing backup manager
            from src.database.utils.backup import TDnetBackupManager
            
            backup_manager = TDnetBackupManager()
            
            if backup_name:
                # If a specific name is provided, use the full backup method but rename the file
                backup_path = backup_manager.create_database_backup("full")
                if backup_path and backup_name:
                    # Rename to custom name
                    new_path = backup_path.parent / backup_name
                    backup_path.rename(new_path)
                    backup_path = new_path
            else:
                # Use the standard backup method
                backup_path = backup_manager.create_database_backup("full")
            
            if backup_path:
                logger.info(f"✓ Migration backup created: {backup_path}")
                logger.info(f"Backup size: {backup_path.stat().st_size / 1024 / 1024:.1f} MB")
                return str(backup_path)
            else:
                logger.error("Failed to create backup")
                return False
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def apply_migrations(self, target_revision="head"):
        """Apply migrations to target revision"""
        logger.info(f"Applying migrations to: {target_revision}")
        
        try:
            result = subprocess.run(['alembic', 'upgrade', target_revision], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Migration failed: {result.stderr}")
                return False
            
            logger.info("✓ Migrations applied successfully")
            logger.info(result.stdout)
            return True
            
        except Exception as e:
            logger.error(f"Error applying migrations: {e}")
            return False
    
    def validate_schema(self):
        """Validate current database schema matches expectations"""
        logger.info("Validating database schema...")
        
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Check for key tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(tables)} tables: {', '.join(tables)}")
            
            # Check for vector extension
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            vector_ext = cursor.fetchone()
            
            if vector_ext:
                logger.info("✓ Vector extension available")
            else:
                logger.warning("Vector extension not found")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Database Migration Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize Alembic and stamp baseline')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate migration using migra')
    gen_parser.add_argument('--dry-run', action='store_true', 
                           help='Generate SQL but don\'t create migration file')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate database and environment')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply migrations')
    apply_parser.add_argument('--target', default='head', 
                             help='Target revision (default: head)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup using existing backup system')
    backup_parser.add_argument('--name', help='Custom backup file name')
    backup_parser.add_argument('--list', action='store_true', help='List existing backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = MigrationManager()
    
    # Always validate environment first
    if not manager.validate_environment():
        logger.error("Environment validation failed. Please fix the issues above.")
        return 1
    
    if args.command == 'init':
        success = manager.init_alembic()
    elif args.command == 'generate':
        success = manager.generate_migration(dry_run=args.dry_run)
    elif args.command == 'validate':
        success = manager.validate_schema()
    elif args.command == 'apply':
        success = manager.apply_migrations(args.target)
    elif args.command == 'status':
        manager.show_status()
        success = True
    elif args.command == 'backup':
        if args.list:
            # List existing backups
            try:
                from src.database.utils.backup import TDnetBackupManager
                backup_manager = TDnetBackupManager()
                backup_manager.list_backups()
                success = True
            except Exception as e:
                logger.error(f"Failed to list backups: {e}")
                success = False
        else:
            # Create backup
            success = bool(manager.create_backup(args.name))
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 