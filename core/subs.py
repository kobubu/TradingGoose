# core/subs.py
import os
import time
from typing import List

from core.db import db  # ← используем общий Postgres-коннектор

FREE_DAILY = int(os.getenv("FREE_DAILY", "3"))
PRO_DAILY  = int(os.getenv("PRO_DAILY", "10"))

BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0") or "0")


# ---------------- DB helpers ----------------

def _ensure_user_row(conn, user_id: int) -> None:
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE user_id=%s", (user_id,))
    if not cur.fetchone():
        cur.execute("INSERT INTO users(user_id) VALUES(%s)", (user_id,))


# ---------------- Schema / migrations ----------------

def init_db():
    """Создаёт таблицы в Postgres, если их ещё нет."""
    with db() as conn:
        cur = conn.cursor()

        # users: финальная версия схемы, без ALTER’ов
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id        BIGINT PRIMARY KEY,
                tier           TEXT    NOT NULL DEFAULT 'free',
                sub_until      BIGINT  NOT NULL DEFAULT 0,
                daily_count    INTEGER NOT NULL DEFAULT 0,
                last_reset     BIGINT  NOT NULL DEFAULT 0,
                signal_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                signal_cats    TEXT    NOT NULL DEFAULT 'all',
                signal_list    TEXT    NOT NULL DEFAULT ''
            )
        """)

        # зафиксированные платежи
        cur.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                tx_hash     TEXT PRIMARY KEY,
                user_id     BIGINT,
                amount_ton  DOUBLE PRECISION,
                created_at  BIGINT NOT NULL
            )
        """)


# ---------------- Tier / limits ----------------

def is_pro(user_id: int) -> bool:
    now = int(time.time())
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT tier, sub_until FROM users WHERE user_id=%s",
            (user_id,)
        )
        tier, sub_until = cur.fetchone()
        sub_until = int(sub_until or 0)

        if tier == "pro" and sub_until and now > sub_until:
            cur.execute(
                "UPDATE users SET tier='free', sub_until=0 WHERE user_id=%s",
                (user_id,)
            )
            return False
        return tier == "pro" and sub_until > now


def set_tier(user_id: int, tier: str, sub_until: int) -> None:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET tier=%s, sub_until=%s WHERE user_id=%s",
            (tier, int(sub_until or 0), user_id)
        )


def get_limits(user_id: int) -> int:
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return 10**9
    return PRO_DAILY if is_pro(user_id) else FREE_DAILY


def _maybe_reset_counter(user_id: int) -> None:
    now = int(time.time())
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT daily_count, last_reset FROM users WHERE user_id=%s",
            (user_id,)
        )
        daily_count, last_reset = cur.fetchone()
        last_reset = int(last_reset or 0)
        if now - last_reset >= 86400:
            cur.execute(
                "UPDATE users SET daily_count=0, last_reset=%s WHERE user_id=%s",
                (now, user_id)
            )


def can_consume(user_id: int) -> bool:
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return True
    _maybe_reset_counter(user_id)
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT daily_count FROM users WHERE user_id=%s",
            (user_id,),
        )
        cnt, = cur.fetchone()
    return int(cnt) < get_limits(user_id)


def consume_one(user_id: int) -> None:
    if BOT_OWNER_ID and user_id == BOT_OWNER_ID:
        return
    _maybe_reset_counter(user_id)
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET daily_count = daily_count + 1 WHERE user_id=%s",
            (user_id,),
        )


# ---------------- Status / signal toggle ----------------

def get_status(user_id: int) -> dict:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute("""
            SELECT tier, sub_until, daily_count, last_reset,
                   signal_enabled, signal_cats, signal_list
            FROM users WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()

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
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET signal_enabled=%s WHERE user_id=%s",
            (enabled, user_id)
        )


def pro_users_for_signal() -> List[int]:
    now = int(time.time())
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id FROM users WHERE tier='pro' AND sub_until>%s",
            (now,)
        )
        rows = cur.fetchall()
    return [r[0] for r in rows]


# ---------------- Signal preferences ----------------

def set_signal_cats(user_id: int, cats: str) -> None:
    cats = (cats or "all").lower()
    if cats not in {"all", "stocks", "crypto", "forex", "custom"}:
        cats = "all"
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET signal_cats=%s WHERE user_id=%s",
            (cats, user_id)
        )


def get_signal_cats(user_id: int) -> str:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT signal_cats FROM users WHERE user_id=%s",
            (user_id,)
        )
        row = cur.fetchone()
        return (row[0] or "all") if row else "all"


def set_signal_list(user_id: int, csv_or_iterable) -> None:
    if isinstance(csv_or_iterable, str):
        tickers = [t.strip().upper() for t in csv_or_iterable.split(",") if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in csv_or_iterable if str(t).strip()]
    norm_csv = ",".join(tickers)
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET signal_list=%s WHERE user_id=%s",
            (norm_csv, user_id)
        )


def get_signal_list(user_id: int) -> List[str]:
    with db() as conn:
        _ensure_user_row(conn, user_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT signal_list FROM users WHERE user_id=%s",
            (user_id,)
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return []
        return [t.strip().upper() for t in row[0].split(",") if t.strip()]


# ---------------- Payments (TON) ----------------

def is_payment_processed(tx_hash: str) -> bool:
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM payments WHERE tx_hash=%s",
            (tx_hash,)
        )
        return cur.fetchone() is not None


def mark_payment_processed(tx_hash: str, user_id: int, amount_ton: float) -> None:
    now = int(time.time())
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO payments(tx_hash, user_id, amount_ton, created_at)
            VALUES(%s, %s, %s, %s)
            ON CONFLICT (tx_hash) DO NOTHING
            """,
            (tx_hash, user_id, float(amount_ton), now)
        )
