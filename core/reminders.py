# core/reminders.py
import time
from typing import List, Tuple

from core.db import db


def init_reminders():
    """Создаёт таблицу напоминаний в Postgres, если нет."""
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id        BIGSERIAL PRIMARY KEY,
            user_id   BIGINT  NOT NULL,
            ticker    TEXT    NOT NULL,
            variant   TEXT    NOT NULL,     -- 'best' | 'top3' | 'all'
            when_ts   BIGINT  NOT NULL,     -- UTC timestamp
            sent      BOOLEAN NOT NULL DEFAULT FALSE
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rem_when ON reminders(when_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rem_user ON reminders(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rem_sent ON reminders(sent)")


def add_reminder(user_id: int, ticker: str, variant: str, when_ts: int) -> int:
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reminders(user_id, ticker, variant, when_ts, sent)
            VALUES(%s, %s, %s, %s, FALSE)
            RETURNING id
        """, (user_id, ticker.upper(), variant, int(when_ts)))
        rem_id, = cur.fetchone()
        return rem_id


def count_active(user_id: int) -> int:
    now = int(time.time())
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM reminders
            WHERE user_id=%s AND sent=FALSE AND when_ts > %s
        """, (user_id, now))
        row = cur.fetchone()
        return row[0] if row else 0


def due_for_day(start_ts: int, end_ts: int) -> List[Tuple[int, int, str, str, int]]:
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_id, ticker, variant, when_ts
            FROM reminders
            WHERE sent=FALSE AND when_ts >= %s AND when_ts < %s
        """, (start_ts, end_ts))
        rows = cur.fetchall()
    return rows


def mark_sent(rem_id: int) -> None:
    with db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE reminders SET sent=TRUE WHERE id=%s", (rem_id,))
