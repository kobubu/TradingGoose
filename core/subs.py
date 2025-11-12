# core/subs.py
import os, sqlite3, time
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.getenv("SUBS_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "artifacts", "subs.sqlite"))
FREE_DAILY = int(os.getenv("FREE_DAILY", "3"))
PRO_DAILY  = int(os.getenv("PRO_DAILY", "10"))

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

def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            tier TEXT NOT NULL DEFAULT 'free',
            sub_until INTEGER NOT NULL DEFAULT 0,
            daily_count INTEGER NOT NULL DEFAULT 0,
            last_reset INTEGER NOT NULL DEFAULT 0,
            signal_enabled INTEGER NOT NULL DEFAULT 0
        )""")

def get_limits(user_id: int) -> int:
    return PRO_DAILY if is_pro(user_id) else FREE_DAILY

def get_status(user_id: int):
    with db() as conn:
        r = conn.execute("SELECT tier, sub_until, daily_count, signal_enabled FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            return {"tier": "free", "sub_until": 0, "daily_count": 0, "signal_enabled": False}
        tier, sub_until, daily_count, sig = r
    return {"tier": tier, "sub_until": sub_until, "daily_count": daily_count, "signal_enabled": bool(sig)}

def set_signal(user_id: int, enabled: bool):
    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO users(user_id) VALUES(?)", (user_id,))
        conn.execute("UPDATE users SET signal_enabled=? WHERE user_id=?", (1 if enabled else 0, user_id))

def _ensure_user(user_id: int):
    with db() as conn:
        cur = conn.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not cur.fetchone():
            conn.execute("INSERT INTO users(user_id) VALUES(?)", (user_id,))

def is_pro(user_id: int) -> bool:
    _ensure_user(user_id)
    now = int(time.time())
    with db() as conn:
        row = conn.execute("SELECT tier, sub_until FROM users WHERE user_id=?", (user_id,)).fetchone()
    tier, sub_until = row
    if sub_until and now > sub_until and tier == "pro":
        set_tier(user_id, "free", 0)
        return False
    return tier == "pro" and sub_until > now

def set_tier(user_id: int, tier: str, sub_until: int):
    _ensure_user(user_id)
    with db() as conn:
        conn.execute("UPDATE users SET tier=?, sub_until=? WHERE user_id=?", (tier, sub_until, user_id))

def get_limits(user_id: int):
    return (PRO_DAILY if is_pro(user_id) else FREE_DAILY)

def _maybe_reset_counter(user_id: int):
    _ensure_user(user_id)
    now = int(time.time())
    # reset в 00:00 UTC (или локальное — на твой вкус)
    with db() as conn:
        row = conn.execute("SELECT daily_count, last_reset FROM users WHERE user_id=?", (user_id,)).fetchone()
        cnt, last = row
        # простой дневной сброс по 86400 сек
        if now - (last or 0) >= 86400:
            conn.execute("UPDATE users SET daily_count=0, last_reset=? WHERE user_id=?", (now, user_id))

def can_consume(user_id: int) -> bool:
    _maybe_reset_counter(user_id)
    with db() as conn:
        cnt, = conn.execute("SELECT daily_count FROM users WHERE user_id=?", (user_id,)).fetchone()
    return cnt < get_limits(user_id)

def consume_one(user_id: int):
    _maybe_reset_counter(user_id)
    with db() as conn:
        conn.execute("UPDATE users SET daily_count = daily_count + 1 WHERE user_id=?", (user_id,))

def get_status(user_id: int):
    _ensure_user(user_id)
    with db() as conn:
        tier, sub_until, daily_count, last_reset, signal_enabled = conn.execute(
            "SELECT tier, sub_until, daily_count, last_reset, signal_enabled FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    return dict(tier=tier, sub_until=sub_until, daily_count=daily_count,
                last_reset=last_reset, signal_enabled=bool(signal_enabled))

def set_signal(user_id: int, enabled: bool):
    _ensure_user(user_id)
    with db() as conn:
        conn.execute("UPDATE users SET signal_enabled=? WHERE user_id=?", (1 if enabled else 0, user_id))

def pro_users_for_signal():
    now = int(time.time())
    with db() as conn:
        rows = conn.execute(
            "SELECT user_id FROM users WHERE tier='pro' AND sub_until>?", (now,)
        ).fetchall()
    return [r[0] for r in rows]
