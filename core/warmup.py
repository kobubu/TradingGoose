# core/warmup.py
import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Optional

from core.data import load_ticker_history, resolve_user_ticker

logger = logging.getLogger(__name__)

# --- конфиг из ENV ---
IDLE_SEC_FOR_WARMUP = int(os.getenv("WARMUP_IDLE_SEC", "10"))      # сколько секунд тишины считаем "idle"
WARMUP_INTERVAL_SEC = int(os.getenv("WARMUP_INTERVAL_SEC", "30"))  # только для информации, сам интервал задаём в bot.py

# --- список тикеров, которые будем греть ---
try:
    # можно использовать те же списки, что и в Signal Mode
    from handlers_pro import SUPPORTED_TICKERS, SUPPORTED_CRYPTO, SUPPORTED_FOREX

    _all = list(dict.fromkeys(
        list(SUPPORTED_TICKERS) + list(SUPPORTED_CRYPTO) + list(SUPPORTED_FOREX)
    ))
except Exception:
    logger.warning("warmup: failed to import SUPPORTED_* from handlers_pro, warmup list is empty")
    _all = []

WARMUP_TICKERS = _all

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

    if _inflight_checker is not None and not _inflight_checker():
        # есть активные train_select_and_forecast — подождём
        return

    if _forecast_fn is None:
        # ещё не зарегистрировали исполнитель — ничего не делаем
        return

    now = time.time()
    if now - LAST_USER_ACTIVITY_TS < IDLE_SEC_FOR_WARMUP:
        # недавно была активность — не мешаем реальным пользователям
        return

    if not WARMUP_TICKERS:
        return

    # выбираем следующий тикер по кругу
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

    logger.info("warmup: start for %s", resolved)

    try:
        # _forecast_fn — это _get_shared_forecast из bot.py,
        # он сам позовёт train_select_and_forecast и использует общий реестр INFLIGHT_FORECASTS.
        await _forecast_fn(df, resolved)
        logger.info("warmup: done for %s", resolved)
    except Exception:
        logger.exception("warmup: failed for ticker=%s", resolved)


async def warmup_job(context) -> None:
    """
    Обёртка для JobQueue (подпись (context) обязательна).
    """
    await warmup_one()
