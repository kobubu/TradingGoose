"""data.py — загрузка исторических данных по тикерам с кэшированием."""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)  # будет core.data

SAVE_CSV = os.getenv("SAVE_CSV", "0") == "1"
CACHE_DAYS = int(os.getenv("CACHE_DAYS", "1"))
ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto").lower()

os.makedirs(ART_DIR, exist_ok=True)
# -------------------- Крипта: маппинг на тикеры Yahoo --------------------
# BTC-USD, ETH-USD, BNB-USD, SOL-USD, ... — минимум 40 штук
CRYPTO_MAP: Dict[str, str] = {
    # Топ-10, как было
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "DOGE": "DOGE-USD",
    "TON":  "TON-USD",
    "BNB":  "BNB-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "ADA":  "ADA-USD",
    "TRX":  "TRX-USD",
    "LTC":  "LTC-USD",

    # Добавляем ещё ~30 ликвидных монет
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
    "MATIC": "MATIC-USD",
    "DOT":  "DOT-USD",
    "BCH":  "BCH-USD",
    "XLM":  "XLM-USD",
    "XMR":  "XMR-USD",
    "ETC":  "ETC-USD",
    "ATOM": "ATOM-USD",
    "NEAR": "NEAR-USD",
    "HBAR": "HBAR-USD",
    "ALGO": "ALGO-USD",
    "APT":  "APT-USD",
    "ARB":  "ARB-USD",
    "OP":   "OP-USD",
    "SAND": "SAND-USD",
    "AXS":  "AXS-USD",
    "RUNE": "RUNE-USD",
    "MKR":  "MKR-USD",
    "LDO":  "LDO-USD",
    "STX":  "STX-USD",
    "IMX":  "IMX-USD",
    "SEI":  "SEI-USD",
    "PYTH": "PYTH-USD",
    "WIF":  "WIF-USD",
    "PEPE": "PEPE-USD",
    "INJ":  "INJ-USD",
    "FIL":  "FIL-USD",
    "GMX":  "GMX-USD",
    "ORDI": "ORDI-USD",
}
# Удобный порядок для кнопок в боте
MAIN_CRYPTO: List[str] = list(CRYPTO_MAP.keys())
# -------------------------------------------------------------------------------------------

# -------------------- Форекс-пары (Yahoo: '=X') --------------------
FOREX_MAP: Dict[str, str] = {
    # Мажоры
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",

    # EUR-кроссы
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "EURCHF": "EURCHF=X",
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "EURCAD": "EURCAD=X",

    # GBP-кроссы
    "GBPJPY": "GBPJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",

    # Прочие кроссы / сырьевые
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "AUDCAD": "AUDCAD=X",
    "AUDNZD": "AUDNZD=X",
}
MAIN_FOREX: List[str] = list(FOREX_MAP.keys())
# ------------------------------------------------------------------


def _now_utc_date() -> datetime.date:
    """Возвращает текущую дату в UTC"""
    return datetime.utcnow().date()


# -------------------- Резолвер тикера от пользователя --------------------
def resolve_user_ticker(user_ticker: str) -> str:
    """
    Принимает 'AAPL' / 'BTC' / 'EURUSD' и возвращает валидный тикер для загрузки.
    - Крипта: BTC -> BTC-USD
    - Форекс: EURUSD -> EURUSD=X
    - Остальное: возвращаем как есть
    """
    t = (user_ticker or "").strip().upper()
    if t in CRYPTO_MAP:
        return CRYPTO_MAP[t]
    if t in FOREX_MAP:
        return FOREX_MAP[t]
    return t
# -------------------------------------------------------------------------------


def _cache_read_if_fresh(ticker: str) -> Optional[pd.DataFrame]:
    """Пытается взять свежий кеш и вернуть DataFrame с колонкой Close"""
    try:
        from glob import glob

        pattern = os.path.join(ART_DIR, f"cache_{ticker}_*.csv")
        cached = sorted(glob(pattern))
        if not cached:
            logger.debug("Cache miss (no files) for ticker=%s", ticker)
            return None

        latest = cached[-1]
        mtime = os.path.getmtime(latest)
        age_days = (datetime.utcnow().timestamp() - mtime) / 86400.0
        if age_days > CACHE_DAYS:
            logger.debug(
                "Cache expired for ticker=%s file=%s age_days=%.3f > CACHE_DAYS=%s",
                ticker, latest, age_days, CACHE_DAYS,
            )
            return None

        cdf = pd.read_csv(latest, parse_dates=True, index_col=0)
        cdf.index = pd.to_datetime(cdf.index).tz_localize(None)

        if "Close" in cdf.columns:
            cdf = cdf[["Close"]]
        elif "close" in cdf.columns:
            cdf = cdf[["close"]].rename(columns={"close": "Close"})
        else:
            found = None
            for col in cdf.columns:
                if "close" in str(col).lower():
                    found = col
                    break
            if found is not None:
                cdf = cdf[[found]].rename(columns={found: "Close"})

        if "Close" not in cdf.columns:
            logger.warning(
                "Cache file for ticker=%s has no Close-like column. path=%s cols=%s",
                ticker, latest, list(cdf.columns),
            )
            return None

        cdf = cdf.dropna()
        cdf = cdf[~cdf.index.duplicated(keep="last")]
        cdf = cdf.sort_index()

        if cdf.empty:
            logger.warning("Cache file for ticker=%s is empty after cleaning; path=%s", ticker, latest)
            return None

        logger.info(
            "Cache HIT for ticker=%s file=%s rows=%d range=[%s .. %s]",
            ticker, latest, len(cdf), cdf.index[0].date(), cdf.index[-1].date(),
        )
        return cdf
    except Exception:
        logger.exception("Error reading cache for ticker=%s", ticker)
        return None


def _cache_write(ticker: str, df: pd.DataFrame) -> None:
    """Сохраняет данные в кэш"""
    try:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cache_out = os.path.join(ART_DIR, f"cache_{ticker}_{stamp}.csv")
        df.to_csv(cache_out, encoding="utf-8")
        logger.info(
            "Cache WRITE for ticker=%s path=%s rows=%d range=[%s .. %s]",
            ticker, cache_out, len(df), df.index[0].date(), df.index[-1].date(),
        )
        if SAVE_CSV:
            hist_out = os.path.join(ART_DIR, f"history_{ticker}_{stamp}.csv")
            df.to_csv(hist_out, encoding="utf-8")
            logger.debug("History CSV saved for ticker=%s path=%s", ticker, hist_out)
    except Exception:
        logger.exception("Error writing cache for ticker=%s", ticker)


def _ensure_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит данные к стандартному формату с колонкой Close"""
    if df is None or df.empty:
        raise ValueError("Empty dataframe")

    close_col: Union[str, tuple, None] = None

    # MultiIndex case
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            try:
                if col[-1] == "Close":
                    close_col = col
                    break
            except Exception:
                pass
        if close_col is None:
            for col in df.columns:
                if "Close" in str(col):
                    close_col = col
                    break
    else:
        if "Adj Close" in df.columns:
            close_col = "Adj Close"
        elif "Close" in df.columns:
            close_col = "Close"
        else:
            for col in df.columns:
                if "close" in str(col).lower():
                    close_col = col
                    break

    if close_col is None:
        raise ValueError(f"Couldn't find Close column in columns={list(df.columns)}")

    series = df[close_col]
    if isinstance(series, pd.Series):
        out = series.to_frame(name="Close")
    else:
        out = pd.DataFrame(series)
        if list(out.columns) != ["Close"]:
            out.columns = ["Close"]

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    if len(out) < 90:
        logger.warning(
            "Short series for ticker dataframe: rows=%d (<90). Downstream may decide to reject.",
            len(out),
        )
    return out


def _fetch_yahoo_clean(ticker: str, years: int = 2) -> pd.DataFrame:
    """Загрузка данных с Yahoo Finance с несколькими fallback методами"""
    ticker = ticker.upper().strip()
    period_days = max(365 * years + 10, 730)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=period_days)

    tries = 3
    last_err = None

    logger.info(
        "Yahoo fetch: ticker=%s start=%s end=%s period_days=%d",
        ticker, start_dt.date(), end_dt.date(), period_days,
    )

    # 1) download(start, end)
    for attempt in range(1, tries + 1):
        try:
            logger.debug("Yahoo attempt#%d with explicit dates for %s", attempt, ticker)
            df = yf.download(
                tickers=ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=10,
                threads=False,
            )
            if df is not None and not df.empty:
                logger.info("Yahoo success (start/end) for %s rows=%d", ticker, len(df))
                return _ensure_close_frame(df)
            raise ValueError("yf.download returned empty (start/end)")
        except Exception as e:
            last_err = e
            logger.warning("Yahoo attempt#%d failed for %s: %s", attempt, ticker, e)
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    # 2) download(period)
    for attempt in range(1, tries + 1):
        try:
            logger.debug("Yahoo attempt#%d with period=%dd for %s", attempt, period_days, ticker)
            df = yf.download(
                tickers=ticker,
                period=f"{period_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=10,
                threads=False,
            )
            if df is not None and not df.empty:
                logger.info("Yahoo success (period) for %s rows=%d", ticker, len(df))
                return _ensure_close_frame(df)
            raise ValueError("yf.download(period) returned empty")
        except Exception as e:
            last_err = e
            logger.warning("Yahoo(period) attempt#%d failed for %s: %s", attempt, ticker, e)
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    # 3) Ticker.history
    for attempt in range(1, tries + 1):
        try:
            logger.debug("Yahoo Ticker.history attempt#%d for %s", attempt, ticker)
            tkr = yf.Ticker(ticker)
            df = tkr.history(period=f"{period_days}d", interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                logger.info("Yahoo Ticker.history success for %s rows=%d", ticker, len(df))
                return _ensure_close_frame(df)
            raise ValueError("Ticker.history returned empty")
        except Exception as e:
            last_err = e
            logger.warning("Yahoo Ticker.history attempt#%d failed for %s: %s", attempt, ticker, e)
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    raise ValueError(f"Yahoo fetch failed for {ticker}: {last_err}")


def _fetch_stooq_close(ticker: str) -> pd.DataFrame:
    """Загрузка данных с Stooq как fallback источник"""
    logger.info("Stooq fetch: ticker=%s", ticker)
    import pandas_datareader.data as pdr

    df = pdr.DataReader(ticker, "stooq")
    if df is None or df.empty:
        raise ValueError("stooq returned empty")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    if "Close" not in df.columns and "close" in df.columns:
        df.columns = [c.capitalize() for c in df.columns]
    if "Close" not in df.columns:
        for col in df.columns:
            if "close" in str(col).lower():
                df = df[[col]].rename(columns={col: "Close"})
                break
    else:
        df = df[["Close"]]

    if "Close" not in df.columns:
        raise ValueError(f"stooq columns have no Close: {list(df.columns)}")

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    if len(df) < 90:
        raise ValueError(f"stooq: too few rows after cleaning: {len(df)}")

    logger.info(
        "Stooq success: ticker=%s rows=%d range=[%s .. %s]",
        ticker, len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


def load_ticker_history(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Основная функция загрузки исторических данных с кэшированием"""
    try:
        # резолвим юзерский тикер заранее
        user_input = ticker
        ticker = resolve_user_ticker(ticker)

        logger.info(
            "load_ticker_history: user_ticker=%s resolved=%s years=%d ART_DIR=%s CACHE_DAYS=%s DATA_SOURCE=%s",
            user_input, ticker, years, ART_DIR, CACHE_DAYS, DATA_SOURCE,
        )

        cached = _cache_read_if_fresh(ticker)
        if cached is not None:
            return cached

        # загрузка с основного источника
        if DATA_SOURCE == "stooq":
            df = _fetch_stooq_close(ticker)
        elif DATA_SOURCE == "yahoo":
            df = _fetch_yahoo_clean(ticker, years=years)
        else:
            # auto mode: сначала Yahoo, потом Stooq
            try:
                df = _fetch_yahoo_clean(ticker, years=years)
            except Exception as e_y:
                logger.warning("Yahoo failed in auto mode for %s: %s; trying Stooq", ticker, e_y)
                df = _fetch_stooq_close(ticker)

        if df is None or df.empty:
            logger.warning("Fetched df is None/empty for ticker=%s", ticker)
            return None

        logger.info(
            "load_ticker_history success: ticker=%s rows=%d range=[%s .. %s]",
            ticker, len(df), df.index[0].date(), df.index[-1].date(),
        )
        _cache_write(ticker, df)
        return df

    except Exception:
        logger.exception("Exception in load_ticker_history(ticker=%s)", ticker)
        return None


# -------------------- Удобные врапперы для криптовалют --------------------

def load_crypto_history(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """
    Загружает историю по крипте по символу из MAIN_CRYPTO (например, 'BTC').
    """
    yf_ticker = resolve_user_ticker(symbol)
    return load_ticker_history(yf_ticker, years=years)


def load_top_crypto_histories(years: int = 2) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Пакетная загрузка по всем 10 основным криптовалютам.
    Возвращает dict: {'BTC': df, 'ETH': df, ...}
    """
    out: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in MAIN_CRYPTO:
        try:
            out[sym] = load_crypto_history(sym, years=years)
        except Exception:
            logger.exception("Error loading crypto history for %s", sym)
            out[sym] = None
    return out
# -------------------------------------------------------------------------------
