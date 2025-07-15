import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


DB_CONFIG = {
    'user': os.environ['TDNET_DB_USER'],
    'password': os.environ['TDNET_DB_PASSWORD'],
    'host': os.environ['TDNET_DB_HOST'],
    'port': os.environ['TDNET_DB_PORT'],
    'database': os.environ['TDNET_DB_NAME']
}


DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"