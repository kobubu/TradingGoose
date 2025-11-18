# helpers/test_db.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# 👇 NEW: load .env like your main app probably does
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.db import PG_DSN
import psycopg2

print("Trying to connect with DSN:")
print(PG_DSN)

try:
    conn = psycopg2.connect(PG_DSN)
    print("✅ Connected successfully")
    conn.close()
except psycopg2.OperationalError as e:
    print("❌ OperationalError:")
    print(repr(e))
