#!/usr/bin/env python3
"""
Launcher script for TDnet Search scraper.
Run this from the project root directory.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the TDnet Search scraper with the provided arguments."""
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    scraper_script = project_root / "src" / "scraper_tdnet_search" / "tdnet_search_scraper.py"
    
    # Pass all arguments to the main scraper
    cmd = [sys.executable, str(scraper_script)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Scraper failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main() 