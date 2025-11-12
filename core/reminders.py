# reminders.py
import sqlite3
from contextlib import closing

_DB = "artifacts/reminders.db"

def _conn():
    return sqlite3.connect(_DB, check_same_thread=False)

def init_reminders():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              ticker TEXT NOT NULL,
              variant TEXT NOT NULL,
              when_ts INTEGER NOT NULL,
              sent INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

def add_reminder(user_id: int, ticker: str, variant: str, when_ts: int) -> int:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reminders (user_id, ticker, variant, when_ts, sent) VALUES (?, ?, ?, ?, 0)",
            (user_id, ticker, variant, when_ts)
        )
        rid = cur.lastrowid          # ✅ берём у курсора
        conn.commit()
        return rid

def count_active(user_id: int) -> int:
    with _conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM reminders WHERE user_id=? AND sent=0", (user_id,))
        (n,) = cur.fetchone()
        return int(n)

def due_for_day(start_ts: int, end_ts: int):
    with _conn() as conn:
        cur = conn.execute(
            "SELECT id, user_id, ticker, variant, when_ts FROM reminders "
            "WHERE when_ts>=? AND when_ts<? AND sent=0",
            (start_ts, end_ts)
        )
        return cur.fetchall()

def mark_sent(rem_id: int):
    with _conn() as conn:
        conn.execute("UPDATE reminders SET sent=1 WHERE id=?", (rem_id,))
        conn.commit()
