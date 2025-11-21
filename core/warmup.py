# core/warmup.py
import asyncio
import logging
import os
import random
from contextlib import contextmanager
from datetime import datetime, time as dt_time
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger("core.warmup")

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
WARMUP_IDLE_SEC = int(os.getenv("WARMUP_IDLE_SEC", "10"))
WARMUP_CHUNK = int(os.getenv("WARMUP_CHUNK", "5"))
WARMUP_SHUFFLE = os.getenv("WARMUP_SHUFFLE", "1") == "1"
WARMUP_PATTERN = os.getenv("WARMUP_PATTERN", "")

# heavy window (локальное время)
WARMUP_HEAVY_START = os.getenv("WARMUP_HEAVY_START", "01:00")
WARMUP_HEAVY_END = os.getenv("WARMUP_HEAVY_END", "09:00")

# лимит тикеров (0 = без лимита)
WARMUP_LIMIT = int(os.getenv("WARMUP_LIMIT", "0"))

# сколько секунд после пользовательской активности warmup не трогаем
WARMUP_USER_IDLE_SEC = int(os.getenv("WARMUP_USER_IDLE_SEC", "20"))

# -----------------------------------------------------------------------------
# Backward compatible inflight-checker + activity API
# -----------------------------------------------------------------------------
_inflight_checker: Optional[Callable[[], bool]] = None
_last_user_activity_ts: float = 0.0


def set_inflight_checker(fn: Callable[[], bool]) -> None:
    """
    bot.py прокидывает сюда функцию, которая возвращает True,
    если сейчас идёт важный forecast/запрос и warmup надо пропустить.
    """
    global _inflight_checker
    _inflight_checker = fn
    logger.info(
        "warmup: inflight checker registered: %s",
        getattr(fn, "__name__", str(fn))
    )


def mark_user_activity() -> None:
    """
    Backward compatible hook.
    bot.py вызывает это при любом пользовательском действии
    (forecast/callback), чтобы warmup не работал несколько секунд.
    """
    global _last_user_activity_ts
    _last_user_activity_ts = datetime.now().timestamp()
    logger.debug("warmup: user activity marked at ts=%.0f", _last_user_activity_ts)


def _user_active_recently(now: Optional[datetime] = None) -> bool:
    if WARMUP_USER_IDLE_SEC <= 0:
        return False
    now_ts = (now or datetime.now()).timestamp()
    return (now_ts - _last_user_activity_ts) < WARMUP_USER_IDLE_SEC


def _has_inflight() -> bool:
    try:
        if _inflight_checker is None:
            return False
        return bool(_inflight_checker())
    except Exception:
        return False


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _parse_hhmm(s: str) -> dt_time:
    try:
        hh, mm = s.strip().split(":")
        return dt_time(hour=int(hh), minute=int(mm))
    except Exception:
        return dt_time(1, 0)


def _now_local() -> datetime:
    return datetime.now()


def _is_heavy_now(now: Optional[datetime] = None) -> bool:
    now = now or _now_local()
    start_t = _parse_hhmm(WARMUP_HEAVY_START)
    end_t = _parse_hhmm(WARMUP_HEAVY_END)
    cur_t = now.time()

    if start_t <= end_t:
        return start_t <= cur_t < end_t
    return (cur_t >= start_t) or (cur_t < end_t)


@contextmanager
def _apply_night_overrides(enabled: bool):
    """
    enabled=True -> временно подменяем os.environ ключами NIGHT_*
    """
    if not enabled:
        yield
        return

    backups = {}
    try:
        for k, v in os.environ.items():
            if not k.startswith("NIGHT_"):
                continue
            base_key = k[len("NIGHT_"):]
            backups[base_key] = os.environ.get(base_key)
            os.environ[base_key] = v

        logger.info("warmup: NIGHT overrides applied (%d keys)", len(backups))
        yield
    finally:
        for base_key, old_val in backups.items():
            if old_val is None:
                os.environ.pop(base_key, None)
            else:
                os.environ[base_key] = old_val

        logger.info("warmup: NIGHT overrides reverted")


# -----------------------------------------------------------------------------
# tickers source
# -----------------------------------------------------------------------------
def _tickers_from_env() -> List[str]:
    raw = os.getenv("WARMUP_TICKERS", "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _load_tickers() -> List[str]:
    tickers = _tickers_from_env()
    if tickers:
        return tickers
    try:
        from core.warmup_tickers import WARMUP_TICKERS  # type: ignore
        return list(WARMUP_TICKERS)
    except Exception:
        return []


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
_forecast_fn: Optional[Callable[..., Awaitable[object]]] = None
_warmup_lock = asyncio.Lock()


def register_forecast_fn(fn: Callable[..., Awaitable[object]]) -> None:
    global _forecast_fn
    _forecast_fn = fn
    logger.info(
        "warmup: forecast function registered: %s",
        getattr(fn, "__name__", str(fn))
    )


# backward compatible alias for bot.py
def set_forecast_fn(fn: Callable[..., Awaitable[object]]) -> None:
    register_forecast_fn(fn)


async def warmup_models(context=None) -> None:
    if _forecast_fn is None:
        logger.warning("warmup: no forecast fn registered; skipping")
        return

    now = _now_local()

    if _has_inflight():
        logger.info("warmup: skipped due to inflight activity")
        return

    if _user_active_recently(now):
        logger.info(
            "warmup: skipped due to recent user activity (<%ds)",
            WARMUP_USER_IDLE_SEC
        )
        return

    if _warmup_lock.locked():
        logger.warning("warmup: skipped because previous run still active")
        return

    async with _warmup_lock:
        heavy = _is_heavy_now(now)
        logger.info(
            "warmup: start | heavy=%s | now=%s | window=%s-%s",
            heavy,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            WARMUP_HEAVY_START,
            WARMUP_HEAVY_END,
        )

        with _apply_night_overrides(heavy):
            await _warmup_impl(heavy=heavy)

        logger.info("warmup: finished run")


# ---- backward compatible entrypoint for scheduler ----
async def warmup_job(context=None):
    """
    Старый entrypoint, который ожидает bot.py:
      warmup.warmup_job
    """
    await warmup_models(context)


async def _warmup_impl(heavy: bool) -> None:
    tickers = _load_tickers()

    if not tickers:
        logger.warning("warmup: no tickers found. Nothing to do.")
        return

    if WARMUP_SHUFFLE:
        random.shuffle(tickers)

    if WARMUP_LIMIT > 0:
        tickers = tickers[:WARMUP_LIMIT]

    logger.info(
        "warmup: tickers=%d chunk=%d idle=%ds heavy=%s",
        len(tickers),
        WARMUP_CHUNK,
        WARMUP_IDLE_SEC,
        heavy,
    )

    total_batches = (len(tickers) + WARMUP_CHUNK - 1) // WARMUP_CHUNK

    for i in range(0, len(tickers), WARMUP_CHUNK):
        batch_no = i // WARMUP_CHUNK + 1
        batch = tickers[i:i + WARMUP_CHUNK]

        logger.info(
            "warmup: batch %d/%d -> %s",
            batch_no,
            total_batches,
            ", ".join(batch)
        )

        results = await asyncio.gather(
            *(warmup_one(t) for t in batch),
            return_exceptions=True
        )

        for tck, res in zip(batch, results):
            if isinstance(res, Exception):
                logger.error("warmup: batch error for %s: %r", tck, res)

        await asyncio.sleep(WARMUP_IDLE_SEC)


async def warmup_one(user_ticker: str) -> None:
    try:
        from core.data import load_ticker_history
    except Exception:
        logger.exception("warmup: cannot import load_ticker_history")
        return

    try:
        df, resolved = await load_ticker_history(user_ticker)
    except TypeError:
        try:
            df, resolved = load_ticker_history(user_ticker)
        except Exception:
            logger.exception("warmup: load_ticker_history failed for %s", user_ticker)
            return
    except Exception:
        logger.exception("warmup: load_ticker_history failed for %s", user_ticker)
        return

    logger.info("warmup: start training %s", resolved)

    try:
        await _forecast_fn(df, resolved)  # type: ignore
        logger.info("warmup: done for %s", resolved)
    except Exception:
        logger.exception("warmup: failed for %s", resolved)
