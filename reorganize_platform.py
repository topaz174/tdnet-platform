#!/usr/bin/env python3
"""
TDnet Financial Intelligence Platform Reorganization Script
===========================================================

This script reorganizes the codebase from separate tdnet_scraper and FinancialIntelligence
repos into a unified, industry-standard directory structure for a financial intelligence platform.

Directory Structure:
    src/
    ├── scraper/          # TDnet data collection
    ├── quantitative/     # Numeric data processing (XBRL)
    ├── qualitative/      # Text data processing (PDFs/XBRL text)
    ├── classifier/       # Disclosure classification
    ├── database/         # Database management & migrations
    ├── analytics/        # Advanced analytics & AI agents
    ├── interface/        # Future API/UI interfaces
    └── shared/           # Shared utilities & configuration
"""

import os
import shutil
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

class PlatformReorganizer:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.new_structure = {
            # Core processing modules
            "src/scraper/tdnet_search": [],
            "src/scraper/core": [],
            "src/scraper/utils": [],
            
            "src/quantitative/xbrl": [],
            "src/quantitative/etl": [],
            "src/quantitative/parsers": [],
            
            "src/qualitative/extraction": [],
            "src/qualitative/pipelines": [],
            "src/qualitative/analytics": [],
            
            "src/classifier/rules": [],
            "src/classifier/ml": [],
            
            "src/database/migrations": [],
            "src/database/views": [],
            "src/database/models": [],
            "src/database/utils": [],
            
            "src/analytics/financial": [],
            "src/analytics/retrieval": [],
            "src/analytics/agents": [],
            
            "src/interface/api": [],
            "src/interface/web": [],
            "src/interface/cli": [],
            
            "src/shared/utils": [],
            "src/shared/config": [],
            "src/shared/logging": [],
            
            # Supporting directories
            "config": [],
            "scripts": [],
            "tests": [],
            "docs": [],
            "data": [],
            "logs": [],
        }
        
        # File mapping: (source_path, target_directory, new_name_if_different)
        self.file_mappings = self._define_file_mappings()
        
    def _define_file_mappings(self) -> List[Tuple[str, str, str]]:
        """Define how files should be moved and renamed"""
        mappings = []
        
        # === SCRAPER MODULE ===
        # TDnet search scraper components
        mappings.extend([
            ("tdnet_scraper/src/scraper_tdnet_search/tdnet_search_scraper.py", "src/scraper/tdnet_search", "scraper.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/debug_scraper.py", "src/scraper/tdnet_search", "debug_scraper.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/backup_tdnet_search.py", "src/scraper/tdnet_search", "backup.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/reset_tdnet_search.py", "src/scraper/tdnet_search", "reset.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/setup_tdnet_search.py", "src/scraper/tdnet_search", "setup.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/google_auth.py", "src/scraper/tdnet_search", "google_auth.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/init_db_search.py", "src/scraper/tdnet_search", "init_db.py"),
            ("tdnet_scraper/src/scraper_tdnet_search/README.md", "src/scraper/tdnet_search", "README.md"),
        ])
        
        # Core scraper components
        mappings.extend([
            ("tdnet_scraper/src/scraper/tdnet_scraper.py", "src/scraper/core", "scraper.py"),
            ("tdnet_scraper/src/scraper/download_existing_xbrls.py", "src/scraper/core", "download_xbrls.py"),
            ("tdnet_scraper/src/scraper/scrape_titles.py", "src/scraper/core", "scrape_titles.py"),
        ])
        
        # === QUANTITATIVE MODULE ===
        # XBRL parsing and processing
        mappings.extend([
            ("tdnet_scraper/src/xbrl_parser_test/arelle_parser.py", "src/quantitative/xbrl", "arelle_parser.py"),
            ("tdnet_scraper/src/xbrl_parser_test/parse_xbrl_statements.py", "src/quantitative/xbrl", "statement_parser.py"),
            ("tdnet_scraper/src/xbrl_parser_test/generate_financial_table.py", "src/quantitative/xbrl", "table_generator.py"),
            ("tdnet_scraper/src/xbrl_parser_test/contexts.json", "src/quantitative/xbrl", "contexts.json"),
        ])
        
        # ETL components for numeric data
        mappings.extend([
            ("tdnet_scraper/src/etl/load_facts.py", "src/quantitative/etl", "load_facts.py"),
            ("tdnet_scraper/src/etl/load_concepts.py", "src/quantitative/etl", "load_concepts.py"),
            ("tdnet_scraper/src/etl/load_xbrl_filings.py", "src/quantitative/etl", "load_filings.py"),
            ("tdnet_scraper/src/etl/load_companies.py", "src/quantitative/etl", "load_companies.py"),
            ("tdnet_scraper/src/etl/concepts.json", "src/quantitative/etl", "concepts.json"),
            ("tdnet_scraper/src/etl/contexts.json", "src/quantitative/etl", "contexts.json"),
            ("tdnet_scraper/src/etl/xsd_elements_universal_with_labels.json", "src/quantitative/etl", "xsd_elements_universal.json"),
            ("tdnet_scraper/src/etl/xsd_elements_organized.json", "src/quantitative/etl", "xsd_elements_organized.json"),
        ])
        
        # === QUALITATIVE MODULE ===
        # Essential qualitative processing files from FinancialIntelligence
        mappings.extend([
            ("FinancialIntelligence/src/unified_extraction_pipeline.py", "src/qualitative/pipelines", "unified_pipeline.py"),
            ("FinancialIntelligence/src/xbrl_qualitative_extractor.py", "src/qualitative/extraction", "xbrl_extractor.py"),
            ("FinancialIntelligence/src/pdf_extraction_pipeline.py", "src/qualitative/extraction", "pdf_pipeline.py"),
            ("FinancialIntelligence/src/parallel_pdf_extraction_pipeline.py", "src/qualitative/pipelines", "parallel_pdf_pipeline.py"),
            ("FinancialIntelligence/financial_data_extraction_agent.py", "src/qualitative/extraction", "data_extraction_agent.py"),
            ("FinancialIntelligence/advanced_financial_analytics.py", "src/qualitative/analytics", "financial_analytics.py"),
            ("FinancialIntelligence/enhanced_retrieval_system.py", "src/qualitative/analytics", "retrieval_system.py"),
        ])
        
        # Move embedding functionality to qualitative
        mappings.extend([
            ("tdnet_scraper/src/embedding/embed_disclosures.py", "src/qualitative/analytics", "embed_disclosures.py"),
            ("tdnet_scraper/src/embedding/embed_disclosures_fixed.py", "src/qualitative/analytics", "embed_disclosures_enhanced.py"),
            ("tdnet_scraper/src/embedding/embed_disclosures_improved.py", "src/qualitative/analytics", "embed_disclosures_improved.py"),
        ])
        
        # === CLASSIFIER MODULE ===
        mappings.extend([
            ("tdnet_scraper/src/classifier/tdnet_classifier.py", "src/classifier/rules", "classifier.py"),
            ("tdnet_scraper/src/classifier/classification_rules.py", "src/classifier/rules", "rules.py"),
            ("tdnet_scraper/src/classifier/test_classify_latest.py", "src/classifier/rules", "test_classifier.py"),
            ("tdnet_scraper/src/classifier/tdnet_ml_classifier.py", "src/classifier/ml", "ml_classifier.py"),
        ])
        
        # === DATABASE MODULE ===
        # Database utilities and initialization
        mappings.extend([
            ("tdnet_scraper/src/database/init_db.py", "src/database/utils", "init_db.py"),
            ("tdnet_scraper/src/database/reset_db.py", "src/database/utils", "reset_db.py"),
            ("tdnet_scraper/src/database/backup_database.py", "src/database/utils", "backup.py"),
            ("tdnet_scraper/src/database/backup_scheduler.py", "src/database/utils", "backup_scheduler.py"),
            ("tdnet_scraper/src/database/truncate_facts.py", "src/database/utils", "truncate_facts.py"),
            ("tdnet_scraper/src/database/populate_classification_tables.py", "src/database/utils", "populate_classification.py"),
            ("tdnet_scraper/src/database/add_classification_columns.py", "src/database/utils", "add_classification_columns.py"),
            ("tdnet_scraper/src/database/migrate_tdnet_search.py", "src/database/utils", "migrate_tdnet_search.py"),
        ])
        
        # === ANALYTICS MODULE ===
        # Move advanced analytics and agents
        mappings.extend([
            ("FinancialIntelligence/advanced_financial_analytics.py", "src/analytics/financial", "analytics.py"),
            ("FinancialIntelligence/enhanced_retrieval_system.py", "src/analytics/retrieval", "enhanced_retrieval.py"),
            ("FinancialIntelligence/financial_data_extraction_agent.py", "src/analytics/agents", "extraction_agent.py"),
        ])
        
        # === SHARED UTILITIES ===
        mappings.extend([
            ("tdnet_scraper/src/utils/path_derivation.py", "src/shared/utils", "path_derivation.py"),
            ("tdnet_scraper/src/utils/change_dir.py", "src/shared/utils", "change_dir.py"),
        ])
        
        # === CONFIGURATION ===
        mappings.extend([
            ("tdnet_scraper/directories.json", "config", "directories.json"),
            ("tdnet_scraper/tdnet_cookies.json", "config", "tdnet_cookies.json"),
        ])
        
        # === DOCUMENTATION ===
        mappings.extend([
            ("tdnet_scraper/README.md", "docs", "tdnet_scraper_original.md"),
            ("FinancialIntelligence/UNIFIED_PIPELINE.md", "docs", "unified_pipeline.md"),
            ("FinancialIntelligence/PDF_EXTRACTION_IMPROVEMENTS.md", "docs", "pdf_extraction_improvements.md"),
            ("FinancialIntelligence/XBRL_EXTRACTION_IMPROVEMENTS.md", "docs", "xbrl_extraction_improvements.md"),
        ])
        
        return mappings
    
    def create_directory_structure(self):
        """Create the new directory structure"""
        print("Creating new directory structure...")
        
        for dir_path in self.new_structure.keys():
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files for Python packages
            if dir_path.startswith("src/"):
                (full_path / "__init__.py").touch()
        
        print("✓ Directory structure created")
    
    def move_files(self):
        """Move and rename files according to the mapping"""
        print("Moving files to new structure...")
        
        moved_files = []
        
        for source_path, target_dir, new_name in self.file_mappings:
            source = self.base_dir / source_path
            target_directory = self.base_dir / target_dir
            target = target_directory / new_name
            
            if source.exists():
                try:
                    # Ensure target directory exists
                    target_directory.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to new location
                    shutil.copy2(source, target)
                    moved_files.append((source_path, f"{target_dir}/{new_name}"))
                    print(f"  ✓ {source_path} → {target_dir}/{new_name}")
                except Exception as e:
                    print(f"  ✗ Failed to move {source_path}: {e}")
            else:
                print(f"  ⚠ Source file not found: {source_path}")
        
        # Copy all migration files
        migrations_source = self.base_dir / "tdnet_scraper/src/database/migrations"
        migrations_target = self.base_dir / "src/database/migrations"
        
        if migrations_source.exists():
            for migration_file in migrations_source.glob("*"):
                if migration_file.is_file():
                    target_file = migrations_target / migration_file.name
                    shutil.copy2(migration_file, target_file)
                    moved_files.append((str(migration_file.relative_to(self.base_dir)), 
                                     f"src/database/migrations/{migration_file.name}"))
                    print(f"  ✓ {migration_file.relative_to(self.base_dir)} → src/database/migrations/{migration_file.name}")
        
        # Copy database views
        views_source = self.base_dir / "tdnet_scraper/src/database/views"
        views_target = self.base_dir / "src/database/views"
        
        if views_source.exists():
            for view_file in views_source.glob("*"):
                if view_file.is_file():
                    target_file = views_target / view_file.name
                    shutil.copy2(view_file, target_file)
                    moved_files.append((str(view_file.relative_to(self.base_dir)), 
                                     f"src/database/views/{view_file.name}"))
                    print(f"  ✓ {view_file.relative_to(self.base_dir)} → src/database/views/{view_file.name}")
        
        return moved_files
    
    def update_import_paths(self, moved_files: List[Tuple[str, str]]):
        """Update import paths in moved files to match new structure"""
        print("Updating import paths...")
        
        # Create mapping of old paths to new paths for imports
        path_mappings = {}
        
        # Build comprehensive path mappings
        path_mappings.update({
            # Scraper imports
            "src.scraper_tdnet_search": "src.scraper.tdnet_search",
            "scraper_tdnet_search": "src.scraper.tdnet_search",
            "src.scraper.tdnet_scraper": "src.scraper.core.scraper",
            
            # Quantitative imports
            "src.etl": "src.quantitative.etl",
            "src.xbrl_parser_test": "src.quantitative.xbrl",
            
            # Classifier imports
            "src.classifier": "src.classifier.rules",
            
            # Database imports
            "src.database": "src.database.utils",
            
            # Utils imports
            "src.utils": "src.shared.utils",
            
            # Embedding imports
            "src.embedding": "src.qualitative.analytics",
        })
        
        # Patterns for different types of imports
        import_patterns = [
            (r'from\s+([a-zA-Z0-9_.]+)\s+import', r'from {} import'),
            (r'import\s+([a-zA-Z0-9_.]+)', r'import {}'),
        ]
        
        files_updated = 0
        
        for _, new_path in moved_files:
            file_path = self.base_dir / new_path
            
            if file_path.suffix == '.py' and file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    original_content = content
                    
                    # Update imports
                    for old_import, new_import in path_mappings.items():
                        content = re.sub(
                            f'from\\s+{re.escape(old_import)}',
                            f'from {new_import}',
                            content
                        )
                        content = re.sub(
                            f'import\\s+{re.escape(old_import)}',
                            f'import {new_import}',
                            content
                        )
                    
                    # Update relative imports to use absolute imports
                    content = self._fix_relative_imports(content, new_path)
                    
                    if content != original_content:
                        file_path.write_text(content, encoding='utf-8')
                        files_updated += 1
                        print(f"  ✓ Updated imports in {new_path}")
                
                except Exception as e:
                    print(f"  ✗ Failed to update imports in {new_path}: {e}")
        
        print(f"✓ Updated imports in {files_updated} files")
    
    def _fix_relative_imports(self, content: str, file_path: str) -> str:
        """Convert relative imports to absolute imports based on new structure"""
        
        # Determine the module path from file path
        path_parts = file_path.split('/')
        if 'src' in path_parts:
            src_index = path_parts.index('src')
            module_parts = path_parts[src_index:-1]  # Exclude filename
            current_module = '.'.join(module_parts)
        else:
            return content
        
        # Fix common relative import patterns
        patterns = [
            (r'from\s+\.\.([a-zA-Z0-9_.]*)\s+import', lambda m: f'from src.{m.group(1)} import'),
            (r'from\s+\.([a-zA-Z0-9_.]*)\s+import', lambda m: f'from {current_module}.{m.group(1)} import'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def create_requirements_txt(self):
        """Merge requirements from both projects"""
        print("Creating unified requirements.txt...")
        
        requirements = set()
        
        # Read from tdnet_scraper
        tdnet_req = self.base_dir / "tdnet_scraper/requirements.txt"
        if tdnet_req.exists():
            try:
                requirements.update(tdnet_req.read_text(encoding='utf-8').strip().split('\n'))
            except UnicodeDecodeError:
                # Try with different encoding or skip if corrupted
                try:
                    requirements.update(tdnet_req.read_text(encoding='latin-1').strip().split('\n'))
                except:
                    print(f"  ⚠ Could not read {tdnet_req}, skipping")
        
        # Read from FinancialIntelligence
        fi_req = self.base_dir / "FinancialIntelligence/requirements.txt"
        if fi_req.exists():
            try:
                requirements.update(fi_req.read_text(encoding='utf-8').strip().split('\n'))
            except UnicodeDecodeError:
                try:
                    requirements.update(fi_req.read_text(encoding='latin-1').strip().split('\n'))
                except:
                    print(f"  ⚠ Could not read {fi_req}, skipping")
        
        # Read from existing requirements.txt
        existing_req = self.base_dir / "requirements.txt"
        if existing_req.exists():
            try:
                requirements.update(existing_req.read_text(encoding='utf-8').strip().split('\n'))
            except UnicodeDecodeError:
                try:
                    requirements.update(existing_req.read_text(encoding='latin-1').strip().split('\n'))
                except:
                    print(f"  ⚠ Could not read {existing_req}, skipping")
        
        # Clean and sort requirements
        clean_requirements = sorted([req.strip() for req in requirements if req.strip() and not req.startswith('#')])
        
        # Write unified requirements
        with open(self.base_dir / "requirements.txt", 'w') as f:
            f.write("# TDnet Financial Intelligence Platform\n")
            f.write("# Unified requirements from tdnet_scraper and FinancialIntelligence\n\n")
            f.write('\n'.join(clean_requirements))
        
        print("✓ Unified requirements.txt created")
    
    def create_main_readme(self):
        """Create a comprehensive README for the unified platform"""
        readme_content = """# TDnet Financial Intelligence Platform

A comprehensive platform for processing TDnet disclosures, combining quantitative data extraction from XBRLs with qualitative analysis of PDFs and text content.

## Architecture

This platform consists of several modular components:

### Core Modules

- **Scraper** (`src/scraper/`): TDnet data collection and downloading
- **Quantitative** (`src/quantitative/`): XBRL parsing and numeric data extraction
- **Qualitative** (`src/qualitative/`): Text extraction and narrative analysis
- **Classifier** (`src/classifier/`): Disclosure categorization and classification
- **Database** (`src/database/`): Schema management, migrations, and utilities
- **Analytics** (`src/analytics/`): Advanced analytics, AI agents, and retrieval systems

### Directory Structure

```
src/
├── scraper/           # TDnet data collection
│   ├── tdnet_search/  # TDnet search scraping
│   ├── core/          # Core scraping logic
│   └── utils/         # Scraping utilities
├── quantitative/      # Numeric data processing
│   ├── xbrl/          # XBRL parsing and processing
│   ├── etl/           # ETL for numeric data
│   └── parsers/       # Financial statement parsers
├── qualitative/       # Text data processing
│   ├── extraction/    # Text extraction pipelines
│   ├── pipelines/     # Processing pipelines
│   └── analytics/     # Text analytics and embeddings
├── classifier/        # Disclosure classification
│   ├── rules/         # Rule-based classification
│   └── ml/           # Machine learning classification
├── database/          # Database management
│   ├── migrations/    # Database schema migrations
│   ├── views/         # Database views
│   ├── models/        # Data models
│   └── utils/         # Database utilities
├── analytics/         # Advanced analytics
│   ├── financial/     # Financial analytics
│   ├── retrieval/     # Information retrieval
│   └── agents/        # AI agents
├── interface/         # API and UI interfaces
│   ├── api/          # REST API endpoints
│   ├── web/          # Web interface
│   └── cli/          # Command line interface
└── shared/           # Shared utilities
    ├── utils/        # General utilities
    ├── config/       # Configuration management
    └── logging/      # Logging utilities
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure database and directories in `config/`

## Usage

### Scraping TDnet Data
```bash
python -m src.scraper.tdnet_search.scraper date 2024-01-15
```

### Processing XBRL Data
```bash
python -m src.quantitative.etl.load_filings
```

### Extracting Qualitative Data
```bash
python -m src.qualitative.pipelines.unified_pipeline
```

### Running Classification
```bash
python -m src.classifier.rules.classifier
```

## Database Management

The platform uses PostgreSQL with a comprehensive migration system:

```bash
# Initialize database
python -m src.database.utils.init_db

# Run migrations
python -m src.database.utils.migrate

# Create backup
python -m src.database.utils.backup
```

## Development

This platform is designed for modularity and scalability. Each component can be developed and tested independently.

## Contributing

When adding new functionality:
1. Follow the modular structure
2. Add appropriate tests
3. Update documentation
4. Follow the existing import patterns
"""
        
        with open(self.base_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("✓ Main README.md created")
    
    def create_gitignore(self):
        """Create a comprehensive .gitignore file"""
        gitignore_content = """# TDnet Financial Intelligence Platform .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log
log/

# Data files
data/
*.csv
*.xlsx
*.json
!config/*.json
!src/**/*.json

# Database
*.db
*.sqlite3

# Security
.env
.env.local
.env.production
*.pem
*.key
credentials.json
*_cookies.json

# Large files
*.pdf
*.zip
*.xbrl
*.htm

# Temporary files
tmp/
temp/
.tmp/

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Coverage reports
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
"""
        
        with open(self.base_dir / ".gitignore", 'w') as f:
            f.write(gitignore_content)
        
        print("✓ .gitignore created")
    
    def initialize_git_and_commit(self):
        """Initialize git repository and make initial commit"""
        print("Initializing git repository...")
        
        try:
            # Initialize git if not already initialized
            subprocess.run(["git", "init"], cwd=self.base_dir, check=True, capture_output=True)
            
            # Add all files
            subprocess.run(["git", "add", "."], cwd=self.base_dir, check=True, capture_output=True)
            
            # Make initial commit
            commit_message = """Initial commit: TDnet Financial Intelligence Platform

Reorganized codebase from separate tdnet_scraper and FinancialIntelligence repos
into unified platform with modular architecture:

- src/scraper/: TDnet data collection
- src/quantitative/: XBRL parsing and numeric data processing
- src/qualitative/: Text extraction and narrative analysis  
- src/classifier/: Disclosure classification
- src/database/: Schema management and migrations
- src/analytics/: Advanced analytics and AI agents
- src/shared/: Common utilities and configuration

This structure supports scalable development and deployment of the
financial intelligence platform."""
            
            subprocess.run(["git", "commit", "-m", commit_message], 
                         cwd=self.base_dir, check=True, capture_output=True)
            
            print("✓ Git repository initialized and initial commit made")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Git operation failed: {e}")
            print("You may need to configure git user settings:")
            print("  git config user.name 'Your Name'")
            print("  git config user.email 'your.email@example.com'")
    
    def run_reorganization(self):
        """Execute the complete reorganization process"""
        print("=" * 60)
        print("TDnet Financial Intelligence Platform Reorganization")
        print("=" * 60)
        print()
        
        # Step 1: Create directory structure
        self.create_directory_structure()
        print()
        
        # Step 2: Move files
        moved_files = self.move_files()
        print()
        
        # Step 3: Update import paths
        self.update_import_paths(moved_files)
        print()
        
        # Step 4: Create unified requirements
        self.create_requirements_txt()
        print()
        
        # Step 5: Create main README
        self.create_main_readme()
        print()
        
        # Step 6: Create .gitignore
        self.create_gitignore()
        print()
        
        # Step 7: Initialize git and commit
        self.initialize_git_and_commit()
        print()
        
        print("=" * 60)
        print("✓ Reorganization completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Review the new directory structure in src/")
        print("2. Test key functionality to ensure imports work")
        print("3. Update any remaining hardcoded paths")
        print("4. Consider database migration consolidation")
        print("5. Develop the interface/ module for deployment")

if __name__ == "__main__":
    reorganizer = PlatformReorganizer()
    reorganizer.run_reorganization() 