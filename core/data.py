import os
import time
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List

import pandas as pd
import yfinance as yf

SAVE_CSV = os.getenv("SAVE_CSV", "0") == "1"
CACHE_DAYS = int(os.getenv("CACHE_DAYS", "1"))
ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto").lower()

# -------------------- NEW: топ-10 криптовалют и маппинг на тикеры Yahoo --------------------
# Берём монеты с устойчивыми тикерами в Yahoo Finance:
# BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD, ADA-USD, DOGE-USD, TRX-USD, AVAX-USD, LTC-USD
CRYPTO_MAP: Dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "TRX": "TRX-USD",
    "AVAX": "AVAX-USD",
    "LTC": "LTC-USD",
}
# Удобный порядок для кнопок в боте
MAIN_CRYPTO: List[str] = list(CRYPTO_MAP.keys())
# -------------------------------------------------------------------------------------------

# -------------------- НОВОЕ: основные форекс-пары (Yahoo: '=X') --------------------
# Символы для пользователя без суффиксов: EURUSD, GBPUSD, USDJPY и т.п.
FOREX_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}
MAIN_FOREX: List[str] = list(FOREX_MAP.keys())

def _now_utc_date() -> datetime.date:
    """Возвращает текущую дату в UTC"""
    return datetime.utcnow().date()

# -------------------- NEW: резолвер тикера от пользователя --------------------
def resolve_user_ticker(user_ticker: str) -> str:
    """
    Принимает, к примеру, 'AAPL' / 'BTC' / 'EURUSD' и возвращает валидный тикер для загрузки.
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
        os.makedirs(ART_DIR, exist_ok=True)
        from glob import glob
        pattern = os.path.join(ART_DIR, f"cache_{ticker}_*.csv")
        cached = sorted(glob(pattern))
        if not cached:
            return None
        latest = cached[-1]
        mtime = os.path.getmtime(latest)
        age_days = (datetime.utcnow().timestamp() - mtime) / 86400.0
        if age_days > CACHE_DAYS:
            return None

        cdf = pd.read_csv(latest, parse_dates=True, index_col=0)
        cdf.index = pd.to_datetime(cdf.index).tz_localize(None)
        if "Close" in cdf.columns:
            cdf = cdf[["Close"]]
        elif "close" in cdf.columns:
            cdf = cdf[["close"]].rename(columns={"close": "Close"})
        else:
            for col in cdf.columns:
                if "close" in str(col).lower():
                    cdf = cdf[[col]].rename(columns={col: "Close"})
                    break
        if "Close" not in cdf.columns:
            return None

        cdf = cdf.dropna()
        cdf = cdf[~cdf.index.duplicated(keep="last")]
        cdf = cdf.sort_index()
        return cdf if not cdf.empty else None
    except Exception:
        return None

def _cache_write(ticker: str, df: pd.DataFrame) -> None:
    """Сохраняет данные в кэш"""
    try:
        os.makedirs(ART_DIR, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cache_out = os.path.join(ART_DIR, f"cache_{ticker}_{stamp}.csv")
        df.to_csv(cache_out, encoding="utf-8")
        if SAVE_CSV:
            hist_out = os.path.join(ART_DIR, f"history_{ticker}_{stamp}.csv")
            df.to_csv(hist_out, encoding="utf-8")
    except Exception:
        pass

def _ensure_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит данные к стандартному формату с колонкой Close"""
    if df is None or df.empty:
        raise ValueError("Empty dataframe")

    close_col: Union[str, tuple, None] = None
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
        if "Close" in df.columns:
            close_col = "Close"
        elif "Adj Close" in df.columns:
            close_col = "Adj Close"
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
        raise ValueError(f"Too few rows after cleaning: {len(out)} (<90)")
    return out

def _fetch_yahoo_clean(ticker: str, years: int = 2) -> pd.DataFrame:
    """Загрузка данных с Yahoo Finance с несколькими fallback методами"""
    ticker = ticker.upper().strip()
    period_days = max(365 * years + 10, 730)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=period_days)

    tries = 3
    last_err = None

    for attempt in range(1, tries + 1):
        try:
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
                return _ensure_close_frame(df)
            raise ValueError("yf.download returned empty")
        except Exception as e:
            last_err = e
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    for attempt in range(1, tries + 1):
        try:
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
                return _ensure_close_frame(df)
            raise ValueError("yf.download(period) returned empty")
        except Exception as e:
            last_err = e
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    for attempt in range(1, tries + 1):
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(period=f"{period_days}d", interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                return _ensure_close_frame(df)
            raise ValueError("Ticker.history returned empty")
        except Exception as e:
            last_err = e
            if attempt < tries:
                time.sleep(1 << (attempt - 1))

    raise ValueError(f"Yahoo fetch failed for {ticker}: {last_err}")

def _fetch_stooq_close(ticker: str) -> pd.DataFrame:
    """Загрузка данных с Stooq как fallback источник"""
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
    return df



def load_ticker_history(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Основная функция загрузки исторических данных с кэшированием"""
    try:
        # -------------------- NEW: резолвим юзерский тикер заранее --------------------
        ticker = resolve_user_ticker(ticker)
        # -----------------------------------------------------------------------------
        print(f"DEBUG: load_ticker_history(ticker={ticker}, years={years})")
        print(f"DEBUG: ART_DIR={ART_DIR}, CACHE_DAYS={CACHE_DAYS}, DATA_SOURCE={DATA_SOURCE}")

        cached = _cache_read_if_fresh(ticker)
        if cached is not None:
            print("DEBUG: using fresh cache; shape:", cached.shape)
            return cached

        if DATA_SOURCE == "stooq":
            df = _fetch_stooq_close(ticker)
        elif DATA_SOURCE == "yahoo":
            df = _fetch_yahoo_clean(ticker, years=years)
        else:
            try:
                df = _fetch_yahoo_clean(ticker, years=years)
            except Exception as e_y:
                print(f"DEBUG: Yahoo failed in auto mode: {e_y}")
                df = _fetch_stooq_close(ticker)
        
        print("DEBUG: df result preview:\n", df.head())
        print("DEBUG: df shape:", df.shape)
        
        if df is None or df.empty:
            print("DEBUG: fetched df is None/empty")
            return None

        _cache_write(ticker, df)
        return df

    except Exception as e:
        print(f"DEBUG: exception in load_ticker_history: {e}")
        return None


# -------------------- NEW: удобные врапперы для криптовалют --------------------
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
            out[sym] = None
    return out
# -------------------------------------------------------------------------------