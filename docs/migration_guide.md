# Database Migration Guide

This guide provides step-by-step instructions for migrating from the current TDnet schema to the unified XBRL processing schema using Alembic and migra.

## Overview

The migration transforms the database from a document-focused TDnet schema to a comprehensive XBRL processing platform that supports:

- **Enhanced Financial Data Processing**: Normalized XBRL facts storage
- **Comprehensive Company Management**: Master company data with sector classification
- **Advanced Classification**: Automated categorization and labeling
- **Performance Optimization**: Materialized views for fast queries
- **Maintained Vector Capabilities**: Continued support for qualitative data embeddings

## Migration Architecture

### Current Schema (Source)
- `company_master`: Company data with vector support
- `disclosures`: Document-focused disclosure tracking with embeddings
- `document_chunks`: Chunked content with vector embeddings
- `reports`: Basic report linkage

### Target Schema (Destination)
- **XBRL Core**: `companies`, `xbrl_filings`, `filing_sections`, `financial_facts`
- **Concept Management**: `concepts`, `concept_overrides`
- **Classification**: `disclosure_categories`, `disclosure_subcategories`, `disclosure_labels`
- **Context Handling**: `context_dims` for period and consolidation tracking
- **Performance**: Materialized views `mv_flat_facts`, `mv_disclosures_classified`
- **Reference Data**: `exchanges`, `sectors`, `units`

## Prerequisites

### 1. Environment Setup

Ensure the following tools are installed:

```bash
# PostgreSQL client tools
sudo apt-get install postgresql-client-14

# Python packages (already in requirements.txt)
pip install alembic>=1.13.0 migra>=1.1.0 psycopg2-binary
```

### 2. Database Configuration

Set up your database connection in one of these ways:

**Option 1: Environment Variables (recommended)**
```bash
# In your .env file
TDNET_DB_USER=your_username
TDNET_DB_PASSWORD=your_password
TDNET_DB_HOST=localhost
TDNET_DB_PORT=5432
TDNET_DB_NAME=tdnet
```

**Option 2: PG_DSN Format**
```bash
# In your .env file
PG_DSN=postgresql://username:password@localhost/tdnet
```

### 3. Pre-Migration Validation

Ensure your current database matches the expected baseline schema:

```bash
# Validate environment and database
python src/database/utils/manage_migration.py validate
```

## Migration Process

### Step 1: Initialize Migration System

```bash
# Initialize Alembic and stamp the current schema as baseline
python src/database/utils/manage_migration.py init
```

This will:
- Set up Alembic tracking in your database
- Mark the current schema as revision `001` (baseline)
- Prepare for future migrations

### Step 2: Create Database Backup

```bash
# Create timestamped backup (uses existing backup system)
python src/database/utils/manage_migration.py backup

# Create named backup
python src/database/utils/manage_migration.py backup --name pre_migration_backup.sql

# List existing backups
python src/database/utils/manage_migration.py backup --list

# For advanced backup options (schema-only, table-specific, etc.)
python src/database/utils/backup.py backup --type full
```

Backups are stored in the configured backup directory from `directories.json` (uses the existing backup system).

### Step 3: Generate Migration

```bash
# Generate migration (dry run first to review)
python src/database/utils/manage_migration.py generate --dry-run

# Generate actual migration file
python src/database/utils/manage_migration.py generate
```

This process:
1. Creates temporary database with target schema
2. Uses migra to compare schemas
3. Generates comprehensive migration SQL
4. Creates Alembic migration file
5. Cleans up temporary resources

### Step 4: Review Migration

Review the generated files:

```bash
# Review the raw migration SQL
less migration_output.sql

# Review the Alembic migration file
ls src/database/migrations/versions/
less src/database/migrations/versions/*_unified_xbrl_schema.py
```

### Step 5: Test Migration (Recommended)

Test on a copy of your database:

```bash
# Create test database
createdb tdnet_test
pg_dump tdnet | psql tdnet_test

# Test migration on copy
DATABASE_URL=postgresql://user:pass@localhost/tdnet_test python src/database/utils/manage_migration.py apply
```

### Step 6: Apply Migration

```bash
# Check current status
python src/database/utils/manage_migration.py status

# Apply migration to production
python src/database/utils/manage_migration.py apply
```

### Step 7: Verify Migration

```bash
# Validate the migrated schema
python src/database/utils/manage_migration.py validate

# Check migration status
python src/database/utils/manage_migration.py status
```

## Migration Details

### Data Transformation

The migration includes automatic data transformation:

1. **Company Data**: `company_master` → `companies` table transformation
2. **Disclosure Restructuring**: Modified `disclosures` table for XBRL workflow
3. **Index Recreation**: All indexes recreated for optimal performance
4. **Extension Maintenance**: PostgreSQL extensions preserved

### Key Changes

1. **Enhanced Company Management**
   - Normalized company data structure
   - Sector and exchange relationships
   - Improved indexing for performance

2. **XBRL Processing Capability**
   - Full XBRL filing workflow support
   - Normalized financial facts storage
   - Context and period management

3. **Advanced Classification**
   - Automated disclosure categorization
   - Hierarchical subcategory support
   - Pattern-based classification rules

4. **Performance Optimizations**
   - Materialized views for common queries
   - Comprehensive indexing strategy
   - Optimized foreign key relationships

### Preserved Capabilities

- Vector embeddings for qualitative analysis
- Full-text search capabilities
- Document chunking and processing
- Extraction pipeline compatibility

## Troubleshooting

### Common Issues

**1. Migration Generation Fails**
```bash
# Check migra is installed
migra --version

# Check temporary database creation permissions
psql -c "CREATE DATABASE test_permissions; DROP DATABASE test_permissions;"
```

**2. Missing Extensions**
```bash
# Install vector extension
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"
```

**3. Permission Issues**
```bash
# Ensure database user has required permissions
psql -c "GRANT ALL PRIVILEGES ON DATABASE tdnet TO your_user;"
```

### Recovery Procedures

**If Migration Fails:**
1. Stop the migration process
2. Restore from backup:
   ```bash
   dropdb tdnet
   createdb tdnet
   psql tdnet < backups/your_backup.sql
   ```
3. Investigate and fix the issue
4. Retry migration

**If Migration Succeeds but Issues Found:**
1. Document the issues
2. Create hotfix migration if needed:
   ```bash
   alembic revision -m "hotfix_description"
   ```

## Post-Migration Tasks

### 1. Update Application Code

Review and update any application code that:
- Directly queries the old schema structure
- Relies on specific table/column names that changed
- Uses deprecated indexes or views

### 2. Performance Monitoring

Monitor database performance after migration:
- Check query execution plans
- Monitor materialized view refresh times
- Verify index usage

### 3. Data Validation

Validate that data was correctly transformed:
```sql
-- Check company count consistency
SELECT COUNT(*) FROM companies;

-- Verify disclosure data integrity
SELECT COUNT(*) FROM disclosures WHERE has_xbrl = true;

-- Check vector data preservation
SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL;
```

## Advanced Usage

### Manual Migration Steps

If you need to perform migration steps manually:

```bash
# 1. Stamp baseline without running init
alembic stamp 001

# 2. Generate migration SQL only
python src/database/utils/generate_migration.py --dry-run

# 3. Apply specific revision
alembic upgrade 002

# 4. Check migration history
alembic history --verbose
```

### Custom Migration SQL

To add custom SQL to the migration:

1. Edit the generated migration file
2. Add custom SQL in the `upgrade()` function
3. Test thoroughly before applying

## Support and Maintenance

### Log Files

Migration logs are written to:
- Console output (real-time)
- Standard Python logging (if configured)

### Schema Dumps

After successful migration, create updated schema dumps:

```bash
# Create post-migration schema dump
pg_dump --schema-only tdnet > post_migration_schema.sql
```

### Regular Maintenance

- **Materialized Views**: Refresh periodically for current data
- **Statistics**: Update table statistics after migration
- **Indexes**: Monitor and optimize as data grows

## Contact and Resources

For migration support:
- Review logs in detail
- Check PostgreSQL logs for database-level errors
- Consult project documentation for application-specific guidance

**Important Files:**
- `alembic.ini`: Alembic configuration
- `src/database/migrations/env.py`: Migration environment setup
- `src/database/migrations/versions/`: Migration files
- `src/database/utils/manage_migration.py`: Migration management
- `src/database/utils/generate_migration.py`: Migration generation 