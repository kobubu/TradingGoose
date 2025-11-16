# core/subs.py
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional

# ---------------- Config ----------------
DB_PATH = os.path.abspath(
    os.getenv(
        "SUBS_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "artifacts", "subs.sqlite")
    )
)
FREE_DAILY = int(os.getenv("FREE_DAILY", "3"))
PRO_DAILY  = int(os.getenv("PRO_DAILY", "10"))

BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0") or "0")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ---------------- DB helpers ----------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    # Немного оптимизаций для SQLite в боте
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _has_column(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def _ensure_user_row(conn: sqlite3.Connection, user_id: int) -> None:
    cur = conn.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if not cur.fetchone():
        conn.execute("INSERT INTO users(user_id) VALUES(?)", (user_id,))

# ---------------- Schema / migrations ----------------
def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id        INTEGER PRIMARY KEY,
                tier           TEXT     NOT NULL DEFAULT 'free',
                sub_until      INTEGER  NOT NULL DEFAULT 0,
                daily_count    INTEGER  NOT NULL DEFAULT 0,
                last_reset     INTEGER  NOT NULL DEFAULT 0,
                signal_enabled INTEGER  NOT NULL DEFAULT 0
            )
        """)
        cur = conn.cursor()
        # миграции недостающих колонок
        if not _has_column(cur, "users", "signal_cats"):
            conn.execute("ALTER TABLE users ADD COLUMN signal_cats TEXT DEFAULT 'all'")
        if not _has_column(cur, "users", "signal_list"):
            conn.execute("ALTER TABLE users ADD COLUMN signal_list TEXT DEFAULT ''")

        # 👉 новая таблица для зафиксированных платежей
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                tx_hash     TEXT PRIMARY KEY,
                user_id     INTEGER,
                amount_ton  REAL,
                created_at  INTEGER NOT NULL
            )
        """)


# ---------------- Tier / limits ----------------
def is_pro(user_id: int) -> bool:
    now = int(time.time())
    with db() as conn:
        _ensure_user_row(conn, user_id)
        tier, sub_until = conn.execute(
            "SELECT tier, sub_until FROM users WHERE user_id=?",
            (user_id,)
        ).fetchone()
        # авто-даунгрейд, если просрочено
        if tier == "pro" and sub_until and now > int(sub_until):
            conn.execute("UPDATE users SET tier='free', sub_until=0 WHERE user_id=?", (user_id,))
            return False
        return tier == "pro" and int(sub_until) > now

def set_tier(user_id: int, tier: str, sub_until: int) -> None:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        conn.execute(
            "UPDATE users SET tier=?, sub_until=? WHERE user_id=?",
            (tier, int(sub_until or 0), user_id)
        )

def get_limits(user_id: int) -> int:
    # владелец бота — условно бесконечный лимит
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return 10**9  # можно любое большое число

    return PRO_DAILY if is_pro(user_id) else FREE_DAILY


def _maybe_reset_counter(user_id: int) -> None:
    now = int(time.time())
    with db() as conn:
        _ensure_user_row(conn, user_id)
        daily_count, last_reset = conn.execute(
            "SELECT daily_count, last_reset FROM users WHERE user_id=?",
            (user_id,)
        ).fetchone()
        last_reset = int(last_reset or 0)
        # простой «каждые 86400 сек»; при желании можно заменить на сброс по полуночам таймзоны
        if now - last_reset >= 86400:
            conn.execute(
                "UPDATE users SET daily_count=0, last_reset=? WHERE user_id=?",
                (now, user_id)
            )

def can_consume(user_id: int) -> bool:
    # владелец бота всегда может
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return True

    _maybe_reset_counter(user_id)
    with db() as conn:
        cnt, = conn.execute(
            "SELECT daily_count FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    return int(cnt) < get_limits(user_id)


def consume_one(user_id: int) -> None:
    # для владельца не трогаем счётчик вообще
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return

    _maybe_reset_counter(user_id)
    with db() as conn:
        _ensure_user_row(conn, user_id)
        conn.execute(
            "UPDATE users SET daily_count = daily_count + 1 WHERE user_id=?",
            (user_id,),
        )


# ---------------- Status / signal toggle ----------------
def get_status(user_id: int) -> dict:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        row = conn.execute("""
            SELECT tier, sub_until, daily_count, last_reset, signal_enabled, signal_cats, signal_list
            FROM users WHERE user_id=?
        """, (user_id,)).fetchone()
    tier, sub_until, daily_count, last_reset, signal_enabled, signal_cats, signal_list = row
    return {
        "tier": tier or "free",
        "sub_until": int(sub_until or 0),
        "daily_count": int(daily_count or 0),
        "last_reset": int(last_reset or 0),
        "signal_enabled": bool(signal_enabled),
        "signal_cats": (signal_cats or "all"),
        "signal_list": (signal_list or "")
    }

def set_signal(user_id: int, enabled: bool) -> None:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        conn.execute(
            "UPDATE users SET signal_enabled=? WHERE user_id=?",
            (1 if enabled else 0, user_id)
        )

def pro_users_for_signal() -> List[int]:
    """Возвращает user_id всех валидных PRO (подписка активна)."""
    now = int(time.time())
    with db() as conn:
        rows = conn.execute(
            "SELECT user_id FROM users WHERE tier='pro' AND sub_until>?",
            (now,)
        ).fetchall()
    return [r[0] for r in rows]

# ---------------- Signal preferences (категории / списки) ----------------
def set_signal_cats(user_id: int, cats: str) -> None:
    """
    cats: 'all' | 'stocks' | 'crypto' | 'forex' | 'custom'
    """
    cats = (cats or "all").lower()
    if cats not in {"all", "stocks", "crypto", "forex", "custom"}:
        cats = "all"
    with db() as conn:
        _ensure_user_row(conn, user_id)
        conn.execute(
            "UPDATE users SET signal_cats=? WHERE user_id=?",
            (cats, user_id)
        )

def get_signal_cats(user_id: int) -> str:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        row = conn.execute("SELECT signal_cats FROM users WHERE user_id=?", (user_id,)).fetchone()
        return (row[0] or "all") if row else "all"

def set_signal_list(user_id: int, csv_or_iterable) -> None:
    """
    Принимает либо строку 'AAPL,MSFT,BTC', либо любой итерируемый список тикеров.
    Хранится как CSV в верхнем регистре, без пробелов.
    """
    if isinstance(csv_or_iterable, str):
        tickers = [t.strip().upper() for t in csv_or_iterable.split(",") if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in csv_or_iterable if str(t).strip()]
    norm_csv = ",".join(tickers)
    with db() as conn:
        _ensure_user_row(conn, user_id)
        conn.execute(
            "UPDATE users SET signal_list=? WHERE user_id=?",
            (norm_csv, user_id)
        )

def get_signal_list(user_id: int) -> List[str]:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        row = conn.execute("SELECT signal_list FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not row or not row[0]:
            return []
        return [t.strip().upper() for t in row[0].split(",") if t.strip()]

# ---------------- Payments (TON) ----------------

def is_payment_processed(tx_hash: str) -> bool:
    with db() as conn:
        cur = conn.execute(
            "SELECT 1 FROM payments WHERE tx_hash=?",
            (tx_hash,)
        )
        return cur.fetchone() is not None


def mark_payment_processed(tx_hash: str, user_id: int, amount_ton: float) -> None:
    now = int(time.time())
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO payments(tx_hash, user_id, amount_ton, created_at) "
            "VALUES(?, ?, ?, ?)",
            (tx_hash, user_id, float(amount_ton), now)
        )
