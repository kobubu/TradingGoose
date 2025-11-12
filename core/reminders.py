# core/reminders.py
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Tuple

# ---------------- Path / setup ----------------
DB_PATH = os.path.abspath(
    os.getenv(
        "REMINDERS_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "artifacts", "reminders.sqlite")
    )
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ---------------- DB helpers ----------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ---------------- Schema ----------------
def init_reminders():
    """Создаёт таблицу напоминаний, если нет."""
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            ticker    TEXT NOT NULL,
            variant   TEXT NOT NULL,     -- 'best' | 'top3' | 'all'
            when_ts   INTEGER NOT NULL,  -- UTC timestamp
            sent      INTEGER NOT NULL DEFAULT 0
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rem_when ON reminders(when_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rem_user ON reminders(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rem_sent ON reminders(sent)")

# ---------------- Core ops ----------------
def add_reminder(user_id: int, ticker: str, variant: str, when_ts: int) -> int:
    """
    Добавляет напоминание.
    Возвращает ID (lastrowid) новой записи.
    """
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reminders(user_id, ticker, variant, when_ts, sent)
            VALUES(?, ?, ?, ?, 0)
        """, (user_id, ticker.upper(), variant, int(when_ts)))
        conn.commit()
        return cur.lastrowid  # <-- именно с курсора, не conn.lastrowid!

def count_active(user_id: int) -> int:
    """Возвращает количество активных (неотправленных и будущих) напоминаний."""
    now = int(time.time())
    with db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) FROM reminders
            WHERE user_id=? AND sent=0 AND when_ts > ?
        """, (user_id, now)).fetchone()
        return row[0] if row else 0

def due_for_day(start_ts: int, end_ts: int) -> List[Tuple[int, int, str, str, int]]:
    """
    Возвращает список [(id, user_id, ticker, variant, when_ts)] напоминаний,
    которые попадают во временное окно [start_ts, end_ts) и ещё не отправлены.
    """
    with db() as conn:
        rows = conn.execute("""
            SELECT id, user_id, ticker, variant, when_ts
            FROM reminders
            WHERE sent=0 AND when_ts >= ? AND when_ts < ?
        """, (start_ts, end_ts)).fetchall()
    return rows

def mark_sent(rem_id: int) -> None:
    """Отмечает напоминание как отправленное."""
    with db() as conn:
        conn.execute("UPDATE reminders SET sent=1 WHERE id=?", (rem_id,))
