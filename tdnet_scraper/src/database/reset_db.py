from sqlalchemy import create_engine
from init_db import Base
import os
import shutil
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_URL

# Connect to database
engine = create_engine(DB_URL)

# Drop all tables
Base.metadata.drop_all(engine)

# Recreate tables
Base.metadata.create_all(engine)
# Get confirmation before proceeding
confirm = input("WARNING: This will delete all data. Are you sure you want to proceed? (y/n): ")
if confirm.lower() != 'y':
    print("Operation cancelled.")
    sys.exit()

# Get PDF directory from config
config_file = 'directories.json'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    pdf_dir = config['pdf_directory']
    
    # Delete PDFs directory if it exists
    if os.path.exists(pdf_dir): 
        shutil.rmtree(pdf_dir)
        print(f"Deleted PDFs directory: {pdf_dir}")
else:
    # Fallback to default directory
    pdf_dir = os.path.join(os.getcwd(), 'pdfs')
    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)
        print(f"Deleted PDFs directory: {pdf_dir}")

print("Database has been reset successfully.")