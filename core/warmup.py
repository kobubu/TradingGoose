# core/warmup.py
import asyncio
import logging
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

WARMUP_CHUNK = int(os.getenv("WARMUP_CHUNK", "5"))

def _interleave_chunks(crypto, stocks, forex, chunk_size: int = 5):
    """
    Склеиваем списки кусками:
    5 крипты, 5 акций, 5 форекс, снова 5 крипты, 5 акций, 5 форекс, ...
    """
    res = []
    i = j = k = 0
    n_c, n_s, n_f = len(crypto), len(stocks), len(forex)

    while i < n_c or j < n_s or k < n_f:
        if i < n_c:
            res.extend(crypto[i:i + chunk_size])
            i += chunk_size
        if j < n_s:
            res.extend(stocks[j:j + chunk_size])
            j += chunk_size
        if k < n_f:
            res.extend(forex[k:k + chunk_size])
            k += chunk_size

    # убираем дубликаты с сохранением порядка (на всякий случай)
    seen = set()
    out = []
    for t in res:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

import time
from typing import Awaitable, Callable, Optional

from core.data import load_ticker_history, resolve_user_ticker

logger = logging.getLogger(__name__)

# --- конфиг из ENV ---
IDLE_SEC_FOR_WARMUP = int(os.getenv("WARMUP_IDLE_SEC", "10"))      # сколько секунд тишины считаем "idle"
WARMUP_INTERVAL_SEC = int(os.getenv("WARMUP_INTERVAL_SEC", "30"))  # только для информации, сам интервал задаём в bot.py

# --- список тикеров, которые будем греть ---
try:
    from handlers_pro import SUPPORTED_TICKERS, SUPPORTED_CRYPTO, SUPPORTED_FOREX

    WARMUP_TICKERS = _interleave_chunks(
        list(SUPPORTED_CRYPTO),
        list(SUPPORTED_TICKERS),
        list(SUPPORTED_FOREX),
        chunk_size=WARMUP_CHUNK,
    )
    logger.info(
        "warmup: built WARMUP_TICKERS with pattern %d/%d/%d, total=%d",
        min(WARMUP_CHUNK, len(SUPPORTED_CRYPTO)),
        min(WARMUP_CHUNK, len(SUPPORTED_TICKERS)),
        min(WARMUP_CHUNK, len(SUPPORTED_FOREX)),
        len(WARMUP_TICKERS),
    )
except Exception:
    logger.exception("warmup: failed to import SUPPORTED_* from handlers_pro, warmup list is empty")
    WARMUP_TICKERS = []

# --- состояние warmup-цикла ---
WARMUP_INDEX = 0
WARMUP_LOCK = asyncio.Lock()
LAST_USER_ACTIVITY_TS = time.time()

# сюда мы из bot.py подадим ссылку на _get_shared_forecast
_forecast_fn: Optional[Callable[[object, str], Awaitable[object]]] = None

WARMUP_CURRENT_TICKER: Optional[str] = None

def get_current_ticker() -> Optional[str]:
    """Для отладки: вернуть тикер, который сейчас тренируется warmup'ом (или None)."""
    return WARMUP_CURRENT_TICKER

_inflight_checker = None

def set_inflight_checker(fn):
    global _inflight_checker
    _inflight_checker = fn

def set_forecast_fn(fn: Callable[[object, str], Awaitable[object]]) -> None:
    """
    Регистрируем функцию, которая умеет считать прогноз:
      async fn(df, resolved_ticker) -> (best, metrics, fb, fa, ft)
    В bot.py мы сюда передадим _get_shared_forecast.
    """
    global _forecast_fn
    _forecast_fn = fn
    logger.info("warmup: forecast function registered: %s", getattr(fn, "__name__", str(fn)))


def mark_user_activity() -> None:
    """
    Вызывай в /forecast и callback'ах, чтобы warmup знал,
    что недавно были пользовательские запросы.
    """
    global LAST_USER_ACTIVITY_TS
    LAST_USER_ACTIVITY_TS = time.time()


async def warmup_one() -> None:
    global WARMUP_INDEX, WARMUP_CURRENT_TICKER

    if _forecast_fn is None:
        return

    now = time.time()
    if now - LAST_USER_ACTIVITY_TS < IDLE_SEC_FOR_WARMUP:
        return

    if not WARMUP_TICKERS:
        return

    async with WARMUP_LOCK:
        ticker = WARMUP_TICKERS[WARMUP_INDEX % len(WARMUP_TICKERS)]
        WARMUP_INDEX += 1

    try:
        resolved = resolve_user_ticker(ticker)
    except Exception:
        resolved = ticker

    df = load_ticker_history(resolved)
    if df is None or df.empty:
        logger.warning("warmup: no data for ticker=%s (resolved=%s)", ticker, resolved)
        return

    # 👇 тут фиксируем, что именно сейчас считаем
    WARMUP_CURRENT_TICKER = resolved
    logger.info("warmup: start training %s", resolved)

    try:
        await _forecast_fn(df, resolved)
        logger.info("warmup: finished training %s", resolved)
    except Exception:
        logger.exception("warmup: failed for %s", resolved)
    finally:
        # по-любому очищаем
        WARMUP_CURRENT_TICKER = None

    


async def warmup_job(context) -> None:
    """
    Обёртка для JobQueue (подпись (context) обязательна).
    """
    await warmup_one()

def get_debug_info(max_tickers: int = 30) -> dict:
    """
    Возвращает диагностическую информацию о warmup-цикле
    для /debug_warmup.
    """
    try:
        last_iso = datetime.fromtimestamp(LAST_USER_ACTIVITY_TS).isoformat()
    except Exception:
        last_iso = f"{LAST_USER_ACTIVITY_TS}"

    return {
        "idle_sec_for_warmup": IDLE_SEC_FOR_WARMUP,
        "interval_sec": WARMUP_INTERVAL_SEC,
        "last_user_activity_ts": LAST_USER_ACTIVITY_TS,
        "last_user_activity_iso": last_iso,
        "current_ticker": WARMUP_CURRENT_TICKER,
        "index": WARMUP_INDEX,
        "total_tickers": len(WARMUP_TICKERS),
        "tickers_preview": WARMUP_TICKERS[:max_tickers],
    }
