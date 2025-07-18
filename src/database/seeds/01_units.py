"""
Seed the units table with standard financial units for XBRL data.
"""

import sys
from pathlib import Path
import psycopg2

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import unified config
from config.config import DB_URL

def seed_units():
    """Seed the units table with standard financial units"""

    # Unit data: (id, currency, scale, unit_code, note)
    units_data = [
    (1, 'JPY', 6, 'JPY_Mil', 'Japanese Yen, in millions'),
    (2, 'JPY', 0, 'JPY', 'Japanese Yen, unscaled'),
    (3, 'JPY', 3, 'JPY_Thou', 'Japanese Yen, in thousands'),
    (4, 'USD', 6, 'USD_Mil', 'US Dollars, in millions'),
    (5, 'SHR', 0, 'Shares', 'Number of shares'),
        (6, 'PUR', 0, 'Pure', 'Pure number/ratio without unit'),
        (7, 'PUR', -2, 'Percent', 'Percentage values (scale -2, divide by 100)')
    ]
    
    print("Seeding units table...")
    
    # Connect to database
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    try:
        # Clear existing data
        print("Clearing existing units data...")
        cursor.execute("TRUNCATE TABLE units RESTART IDENTITY CASCADE")
        
        # Insert units data
        insert_query = """
            INSERT INTO units (id, currency, scale, unit_code, note)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        for unit in units_data:
            cursor.execute(insert_query, unit)
        
        # Commit the transaction
        conn.commit()
        print(f"Successfully seeded {len(units_data)} units")
        
    except Exception as e:
        conn.rollback()
        print(f"Error seeding units: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    seed_units()