# Database Migration System

Complete Alembic + migra migration system for transforming TDnet schema to unified XBRL schema.

## Quick Start

```bash
# 1. Validate environment
python src/database/utils/manage_migration.py validate

# 2. Create backup
python src/database/utils/manage_migration.py backup

# 3. Initialize migration system
python src/database/utils/manage_migration.py init

# 4. Generate migration (dry run first)
python src/database/utils/manage_migration.py generate --dry-run
python src/database/utils/manage_migration.py generate

# 5. Apply migration
python src/database/utils/manage_migration.py apply
```

## Files Structure

```
├── alembic.ini                              # Alembic configuration
├── src/database/
│   ├── migrations/
│   │   ├── env.py                          # Migration environment
│   │   ├── script.py.mako                  # Migration template
│   │   ├── 001_schema.sql                  # Existing migrations
│   │   ├── ...                             # (29+ existing migrations)
│   │   └── versions/
│   │       ├── 001_baseline_tdnet_schema.py # Baseline migration
│   │       └── *_unified_xbrl_schema.py     # Generated migration
│   └── utils/
│       ├── manage_migration.py             # Main management script
│       └── generate_migration.py           # Migration generation
├── docs/
│   └── migration_guide.md                  # Comprehensive guide
├── tdnet_schema_dump.sql                   # Current schema
├── target_schema.sql                       # Target schema
└── migration_output.sql                    # Generated migration SQL
```

## Commands

| Command | Description |
|---------|-------------|
| `validate` | Check environment and database |
| `init` | Initialize Alembic and stamp baseline |
| `generate` | Generate migration using migra |
| `apply` | Apply migrations to database |
| `status` | Show current migration status |
| `backup` | Create database backup (uses existing backup system) |

## Key Features

- **Automated Schema Comparison**: Uses migra to generate precise migration SQL
- **Safe Migration Process**: Baseline stamping, backups, and validation
- **Comprehensive Management**: Single script for all migration operations
- **Detailed Documentation**: Complete migration guide with troubleshooting

## Requirements

- PostgreSQL client tools (`psql`, `pg_dump`)
- Python packages: `alembic>=1.13.0`, `migra>=1.1.0`, `psycopg2-binary`
- Database connection configured via environment variables

## Documentation

- **Complete Guide**: [docs/migration_guide.md](docs/migration_guide.md)
- **Architecture Details**: See migration guide for schema transformation details
- **Troubleshooting**: Common issues and recovery procedures in guide

## Schema Transformation

**From** (TDnet Schema):
- Document-focused disclosure tracking
- Basic company and vector support
- Simple extraction workflow

**To** (Unified XBRL Schema):
- Comprehensive XBRL processing
- Normalized financial facts storage
- Advanced classification and categorization
- Performance-optimized materialized views

## Safety Features

- ✅ **Baseline Migration**: Current schema preserved as starting point
- ✅ **Automatic Backups**: Database backup before migration
- ✅ **Dry Run Mode**: Review migration SQL before applying
- ✅ **Validation Tools**: Environment and schema validation
- ✅ **Recovery Procedures**: Clear rollback instructions

---

## Integration with Existing Systems

This Alembic migration system integrates seamlessly with your existing infrastructure:

### Database Migrations
- Works alongside your existing SQL migration files in `src/database/migrations/`
- The existing 29+ migrations (001_schema.sql through 029_fix_ytd_fiscal_span_constraint.sql) represent your current database state
- The new Alembic system manages the transition to the unified XBRL schema

### Backup System
- Uses your existing `src/database/utils/backup.py` for all backup operations
- Respects the backup directory configured in `directories.json`
- Provides same comprehensive backup features (full database, schema-only, table-specific)

**⚠️ Important**: Always backup your database before running migrations in production! 