import os
import shutil
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from init_db import Disclosure, Base, engine

# Constants
CONFIG_FILE = 'directories.json'

def load_config():
    """Load the current PDF location configuration"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default configuration - PDFs are in a 'pdfs' subdirectory of current directory
        default_path = os.path.join(os.getcwd(), 'pdfs')
        config = {'pdf_directory': default_path}
        save_config(config)
        return config

def save_config(config):
    """Save the PDF location configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def get_new_path(old_path, old_base, new_base):
    """Generate the new file path by replacing the old base path with the new one"""
    # Ensure paths use consistent separators and handle path normalization
    old_path = os.path.normpath(old_path)
    old_base = os.path.normpath(old_base)
    new_base = os.path.normpath(new_base)
    
    # Get the relative path from the old base
    rel_path = os.path.relpath(old_path, old_base)
    
    # Join with the new base
    new_path = os.path.join(new_base, rel_path)
    
    return new_path

def move_pdfs_physically(old_location, new_location):
    """
    Move PDF files from old location to new location.
    
    Args:
        old_location (str): Current base directory for PDF files
        new_location (str): New base directory for PDF files
    """
    # Ensure the new directory exists
    if not os.path.exists(new_location):
        os.makedirs(new_location, exist_ok=True)
    
    print(f"Moving PDFs from {old_location} to {new_location}")
    
    # Create session for database queries
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all disclosures from the database
        disclosures = session.query(Disclosure).all()
        
        files_moved = 0
        
        for disclosure in disclosures:
            if disclosure.pdf_path and os.path.exists(disclosure.pdf_path):
                # Generate the new path
                new_path = get_new_path(disclosure.pdf_path, old_location, new_location)
                
                # Create directory structure if it doesn't exist
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # Move the file
                try:
                    shutil.move(disclosure.pdf_path, new_path)
                    files_moved += 1
                except Exception as e:
                    print(f"Error moving file {disclosure.pdf_path}: {e}")
        
        print(f"Successfully moved {files_moved} files.")
        
    except Exception as e:
        print(f"Error moving PDF files: {e}")
    finally:
        session.close()

def update_database_paths():
    """
    Update all database paths to use the directory from directories.json
    """
    # Load the target directory from config
    config = load_config()
    target_base_dir = config['pdf_directory']
    
    print(f"Updating database paths to use base directory: {target_base_dir}")
    
    # Create session for database updates
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all disclosures from the database
        disclosures = session.query(Disclosure).all()
        
        paths_updated = 0
        
        for disclosure in disclosures:
            if disclosure.pdf_path:
                # Extract the relative part (everything after the base directory)
                old_path = os.path.normpath(disclosure.pdf_path)
                
                # Try to find the date folder pattern (e.g., "2025-05-20")
                path_parts = old_path.split(os.sep)
                date_folder_index = -1
                
                for i, part in enumerate(path_parts):
                    # Look for date pattern YYYY-MM-DD
                    if len(part) == 10 and part.count('-') == 2:
                        try:
                            year, month, day = part.split('-')
                            if (len(year) == 4 and len(month) == 2 and len(day) == 2 and
                                year.isdigit() and month.isdigit() and day.isdigit()):
                                date_folder_index = i
                                break
                        except ValueError:
                            continue
                
                if date_folder_index != -1:
                    # Rebuild path with new base directory
                    relative_parts = path_parts[date_folder_index:]
                    new_path = os.path.join(target_base_dir, *relative_parts)
                    new_path = os.path.normpath(new_path)
                    
                    # Update the database path
                    disclosure.pdf_path = new_path
                    paths_updated += 1
                else:
                    print(f"Could not find date folder pattern in path: {disclosure.pdf_path}")
        
        # Commit the changes to the database
        session.commit()
        
        print(f"Successfully updated {paths_updated} database records.")
        
    except Exception as e:
        session.rollback()
        print(f"Error updating database paths: {e}")
    finally:
        session.close()

def update_scraper_code(new_location):
    """
    Update the PDF path in tdnet_scraper.py
    
    Args:
        new_location (str): New base directory for PDF files
    """
    try:
        # Read the current scraper file
        with open('tdnet_scraper.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find and replace the PDF directory definition
        for i, line in enumerate(lines):
            if "pdf_dir = os.path.join(os.getcwd(), 'pdfs', date_folder)" in line:
                # Use a raw string to avoid escape issues
                config_code = f"    # Load PDF directory from config\n    with open('{CONFIG_FILE}', 'r') as f:\n        config = json.load(f)\n    pdf_base_dir = config['pdf_directory']\n    pdf_dir = os.path.join(pdf_base_dir, date_folder)\n"
                lines[i] = config_code
                
                # Add json import if needed
                if "import json" not in "".join(lines[:20]):
                    for j, import_line in enumerate(lines):
                        if "import" in import_line and "os" in import_line:
                            lines.insert(j+1, "import json\n")
                            break
                
                break
        
        # Write back the modified file
        with open('tdnet_scraper.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print("Updated tdnet_scraper.py to use the new PDF location")
    except Exception as e:
        print(f"Error updating tdnet_scraper.py: {e}")

if __name__ == "__main__":
    print("Change PDF Storage Location")
    print("---------------------------")
    
    # Get current location
    config = load_config()
    current_location = config['pdf_directory']
    print(f"Current PDF storage location: {current_location}")
    
    # Ask for the new location
    new_location = input("Enter new PDF storage location (absolute path): ")
    
    # Normalize path (handle backslashes)
    new_location = os.path.normpath(new_location)
    
    if not os.path.isabs(new_location):
        print("Please enter an absolute path.")
    else:
        # Update the configuration first
        config['pdf_directory'] = new_location
        save_config(config)
        print(f"Updated configuration to use: {new_location}")
        
        # Ask if user wants to move PDFs physically
        move_files = input("Do you want to move PDF files to the new location? (y/n): ").lower().strip()
        if move_files == 'y':
            move_pdfs_physically(current_location, new_location)
            # Update tdnet_scraper.py to use the new path
            update_scraper_code(new_location)
        
        # Ask if user wants to update database paths
        update_db = input("Do you want to update database paths to use the new location? (y/n): ").lower().strip()
        if update_db == 'y':
            update_database_paths() 