import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to project root
load_dotenv(BASE_DIR / '.env')


# TDnet Search database configuration (separate database)
DB_CONFIG_SEARCH = {
    'user': os.environ['TDNET_DB_USER'],
    'password': os.environ['TDNET_DB_PASSWORD'],
    'host': os.environ['TDNET_DB_HOST'],
    'port': os.environ['TDNET_DB_PORT'],
    'database': 'tdnet_search'  # Different database name
}


DB_URL_SEARCH = f"postgresql://{DB_CONFIG_SEARCH['user']}:{DB_CONFIG_SEARCH['password']}@{DB_CONFIG_SEARCH['host']}:{DB_CONFIG_SEARCH['port']}/{DB_CONFIG_SEARCH['database']}" 