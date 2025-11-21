"""
forecast.py — пайплайн обучения моделей и построения 30-дневного прогноза.
Содержит:
- кэш моделей и прогнозов
- мягкая санитизация прогноза
- защита от "битых" кэшей (слишком плоских / упавших в lo / странных)
- (опционально) кэш PNG графиков рядом с forecasts
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional, Tuple, List, Any

import logging
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm

from . import model_cache
from .models import refit_and_forecast_30d, select_and_fit_with_candidates

logger = logging.getLogger(__name__)
models_logger = logging.getLogger("models")

# ---- ВАЖНО: версия модели для кэша ----
MODEL_VERSION = str(os.getenv("MODEL_VERSION", "v5")).strip()

WF_HORIZON = int(os.getenv("WF_HORIZON", "5"))
MODEL_CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))
ENSEMBLES_ENABLED = os.getenv("ENSEMBLES_ENABLED", "1") == "1"

# Куда складываем статистику по моделям
STATS_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "models_stats.csv"
)

# ---------- Data signature для кэша ----------


def _make_data_signature(df: pd.DataFrame, tail_len: int = 256) -> str:
    """
    Сигнатура данных для кэша моделей/прогнозов.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe in _make_data_signature")

    idx = pd.to_datetime(df.index)
    n = int(len(df))
    start_dt = idx[0].date().isoformat()
    end_dt = idx[-1].date().isoformat()

    closes = df["Close"].astype("float32").to_numpy()
    tail = closes[-tail_len:] if n > tail_len else closes

    tail_bytes = tail.tobytes()
    tail_hash = hashlib.sha1(tail_bytes).hexdigest()

    payload = {
        "n": n,
        "start": start_dt,
        "end": end_dt,
        "tail_sha1": tail_hash,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_fc_key_and_sig(
    df: pd.DataFrame, ticker: str
) -> Tuple[str, str]:
    """
    Утилита для bot.py:
    возвращает (fc_key, data_sig)
    """
    sig = _make_data_signature(df)
    fc_key = model_cache.make_forecasts_key(ticker or "N/A", sig)
    return fc_key, sig


def _make_future_index(
    last_ts: pd.Timestamp, periods: int, ticker: str
) -> pd.DatetimeIndex:
    t = (ticker or "").upper()
    is_crypto = t.endswith("-USD")
    is_fx = t.endswith("=X")
    start = last_ts + pd.Timedelta(days=1)
    if is_crypto or is_fx:
        return pd.date_range(start=start, periods=periods, freq="D")
    return pd.bdate_range(start=start, periods=periods)


# Жёсткость фильтра можно крутить из .env
FC_MAX_MULT = float(os.getenv("FC_MAX_MULT", "5.0"))     # макс. ×5 от последней цены
FC_MIN_MULT = float(os.getenv("FC_MIN_MULT", "0.2"))     # мин. ×0.2
FC_MAX_DAILY_CHG = float(os.getenv("FC_MAX_DAILY_CHG", "0.5"))  # макс. дневной скачок 50%

# дополнительная проверка первого шага относительно last_close
FC_MAX_START_DEV = float(os.getenv("FC_MAX_START_DEV", "0.35"))  # 35% от last_close

# порог "плоскости" прогноза
FC_FLAT_PCT = float(os.getenv("FC_FLAT_PCT", "0.003"))  # 0.3% std/price


def _sanitize_forecast_array(arr: np.ndarray, last_close: float) -> np.ndarray:
    """
    Мягкая санитизация прогноза:
    - заменяем NaN/inf на last_close
    - убираем нули/отрицательные цены (ставим нижнюю границу lo)
    - режем экстремальные уровни по FC_MIN_MULT / FC_MAX_MULT
    - МЯГКО ограничиваем дневной скачок по FC_MAX_DAILY_CHG

    НИКОГДА не возвращаем None, только откорректированный массив.
    """
    arr = np.asarray(arr, dtype=float)

    if arr.size == 0:
        return arr

    # 1) базовый уровень last_close
    if not np.isfinite(last_close) or last_close <= 0:
        finite_pos = arr[np.isfinite(arr) & (arr > 0)]
        if finite_pos.size > 0:
            last_close = float(finite_pos[-1])
        else:
            last_close = 1.0

    # 2) NaN/inf → last_close
    arr = np.nan_to_num(arr, nan=last_close, posinf=last_close, neginf=last_close)

    # 3) вычисляем границы ДО любых замен
    lo = last_close * FC_MIN_MULT
    hi = last_close * FC_MAX_MULT
    if not np.isfinite(lo) or lo <= 0:
        lo = max(last_close * 0.01, 1e-6)
    if not np.isfinite(hi) or hi <= lo:
        hi = last_close * 10.0

    # 4) не даём нулевые/отрицательные цены
    arr[arr <= 0] = lo

    # 5) мягкие границы по мультипликаторам
    arr = np.clip(arr, lo, hi)

    # 6) МЯГКИЙ лимит по дневному изменению
    if arr.size >= 2:
        max_rel = FC_MAX_DAILY_CHG
        for i in range(1, len(arr)):
            prev = arr[i - 1]
            if prev <= 0:
                continue
            r = (arr[i] - prev) / prev
            if abs(r) > max_rel:
                arr[i] = prev * (1.0 + np.sign(r) * max_rel)

        rel = np.diff(arr) / arr[:-1]
        max_jump = float(np.max(np.abs(rel))) if rel.size > 0 else 0.0
        if max_jump > max_rel * 0.9:
            logger.warning(
                "Forecast had big raw jumps, clipped to max |Δ|=%.2f (final max=%.2f)",
                max_rel,
                max_jump,
            )

    return arr


def _is_too_flat(arr: np.ndarray, last_close: float, flat_pct: float = FC_FLAT_PCT) -> bool:
    arr = np.asarray(arr, float)
    if arr.size < 2:
        return True
    rel = float(np.std(arr) / max(last_close, 1e-9))
    return rel < flat_pct


def _forecast_sanity_ok(arr: np.ndarray, last_close: float) -> bool:
    """
    Более строгая проверка "нормальности" прогноза. Нужна, чтобы
    не залипать на битом кэше после рестарта.
    """
    if arr is None:
        return False
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if np.any(arr <= 0):
        return False

    lo = last_close * FC_MIN_MULT
    hi = last_close * FC_MAX_MULT
    lo = max(lo, 1e-9)

    # 1) первый шаг не должен улетать далеко от last_close
    dev0 = abs(arr[0] - last_close) / max(last_close, 1e-9)
    if dev0 > FC_MAX_START_DEV:
        return False

    # 2) не должен почти весь лежать на lo или hi
    frac_lo = float(np.mean(arr <= lo * 1.001))
    frac_hi = float(np.mean(arr >= hi * 0.999))
    if frac_lo > 0.6 or frac_hi > 0.6:
        return False

    # 3) не должен быть слишком плоским
    if _is_too_flat(arr, last_close):
        return False

    return True


# ---------- Ансамбли ----------


def _build_ensembles_from_candidates(
    y: pd.Series,
    candidates,
    future_idx: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строим 2 ансамбля:
      - среднее по всем кандидатам
      - среднее по топ-3 по RMSE
    """
    last_close = float(y.iloc[-1])
    if not np.isfinite(last_close) or last_close == 0.0:
        last_close = 1.0

    def _safe_fcst_for_candidate(cand) -> Optional[np.ndarray]:
        try:
            s = refit_and_forecast_30d(y, cand)
            arr = s.values.astype(float)
            arr = _sanitize_forecast_array(arr, last_close)
            if not _forecast_sanity_ok(arr, last_close):
                return None
            return arr
        except Exception:
            logger.exception(
                "refit_and_forecast_30d failed for candidate %s",
                getattr(cand, "name", "?"),
            )
            return None

    # --- среднее по ВСЕМ кандидатам ---
    all_fcsts: List[np.ndarray] = []
    good_candidates = []
    for c in candidates:
        arr = _safe_fcst_for_candidate(c)
        if arr is not None:
            all_fcsts.append(arr)
            good_candidates.append(c)

    if all_fcsts:
        mat = np.vstack(all_fcsts)
        avg_all = np.nanmean(mat, axis=0)
        fcst_avg_all = pd.DataFrame({"forecast": avg_all}, index=future_idx)
    else:
        fcst_avg_all = pd.DataFrame({"forecast": []})

    # --- топ-3 по RMSE среди "хороших" кандидатов ---
    if good_candidates:
        top3 = sorted(good_candidates, key=lambda c: float(c.rmse))[:3]
    else:
        top3 = []

    top3_mat: List[np.ndarray] = []
    for c in top3:
        arr = _safe_fcst_for_candidate(c)
        if arr is not None:
            top3_mat.append(arr)

    if top3_mat:
        top3_avg = np.nanmean(np.vstack(top3_mat), axis=0)
        fcst_avg_top3 = pd.DataFrame({"forecast": top3_avg}, index=future_idx)
    else:
        fcst_avg_top3 = pd.DataFrame({"forecast": []})

    return fcst_avg_all, fcst_avg_top3


def _is_fresh(meta: Dict[str, Any], ttl: int) -> bool:
    try:
        return (
            meta.get("model_version") == MODEL_VERSION
            and (time.time() - int(meta.get("trained_at", 0)) <= ttl)
        )
    except Exception:
        return False


def load_cached_plot_if_fresh(df: pd.DataFrame, ticker: str, ttl: Optional[int] = None) -> Optional[bytes]:
    """
    Достаёт PNG-байты из plot-cache, если forecasts свежие.
    Используется в bot.py.
    """
    if ttl is None:
        ttl = MODEL_CACHE_TTL_SECONDS

    try:
        fc_key, sig = make_fc_key_and_sig(df, ticker)
        fb, fa, ft, meta = model_cache.load_forecasts(fc_key)
        if meta is None or not _is_fresh(meta, ttl) or meta.get("data_sig") != sig:
            return None
        return model_cache.load_plot(fc_key)
    except Exception:
        logger.exception("load_cached_plot_if_fresh failed for %s", ticker)
        return None


def load_cached_forecasts_if_fresh(
    df: pd.DataFrame,
    ticker: Optional[str],
    ttl: Optional[int] = None,
) -> Optional[Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Быстрая попытка достать 3 прогноза (best / avg_all / avg_top3) из кэша.
    НИЧЕГО не обучает и не трогает модели.

    Возвращает:
      (best_dict, metrics, fb, fa, ft)
    или None.
    """
    if not ticker:
        return None

    if ttl is None:
        ttl = MODEL_CACHE_TTL_SECONDS

    y = df["Close"].astype(float)
    data_sig = _make_data_signature(df)
    fc_key = model_cache.make_forecasts_key(ticker or "N/A", data_sig)

    fb, fa, ft, fmeta = model_cache.load_forecasts(fc_key)
    if fb is None or fa is None or ft is None or fmeta is None:
        return None

    last_close = float(y.iloc[-1])

    # санитизируем и проверяем sanity
    arr = fb["forecast"].values.astype(float)
    arr_ok = _sanitize_forecast_array(arr, last_close)

    if (
        _is_fresh(fmeta, ttl)
        and fmeta.get("data_sig") == data_sig
        and _forecast_sanity_ok(arr_ok, last_close)
    ):
        logger.info("Forecasts cache HIT (fast path) for %s", ticker)
        models_logger.info(
            "Using cached FORECASTS (fast) for %s: best=%s rmse=%.4f",
            (ticker or "N/A"),
            fmeta.get("best_name", "cached_best"),
            float((fmeta.get("metrics") or {}).get("rmse") or float("nan")),
        )

        fb = fb.copy()
        fb["forecast"] = arr_ok

        best_dict = {"name": fmeta.get("best_name", "cached_best")}
        metrics = fmeta.get("metrics", {"rmse": None, "mape": None})
        return best_dict, metrics, fb, fa, ft

    logger.info("Forecasts cache for %s is stale/insane → ignore", ticker)
    return None


def _make_config_id(model_name: str, extra: Optional[dict]) -> str:
    """Стабильный идентификатор конфигурации модели."""
    try:
        payload = {
            "name": model_name,
            "extra": extra or {},
            "model_version": MODEL_VERSION,
        }
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(s).hexdigest()
    except Exception:
        return f"legacy::{model_name}"


def _append_model_stats(
    ticker: Optional[str],
    model_name: str,
    model_type: Optional[str],
    metrics: Dict[str, Optional[float]],
    data_sig: str,
    n_points: int,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    val_steps: int,
    extra: Optional[dict],
    source: str = "train",
) -> None:
    """Пишет одну строку в artifacts/models_stats.csv."""
    try:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

        rmse = metrics.get("rmse")
        mape = metrics.get("mape")
        config_id = _make_config_id(model_name, extra)

        row = {
            "ts": int(time.time()),
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "ticker": (ticker or "N/A").upper(),
            "model_name": model_name,
            "model_type": model_type or "",
            "config_id": config_id,
            "rmse": float(rmse) if rmse is not None else "",
            "mape": float(mape) if mape is not None else "",
            "val_steps": int(val_steps),
            "n_points": int(n_points),
            "start_date": pd.to_datetime(start_dt).date().isoformat(),
            "end_date": pd.to_datetime(end_dt).date().isoformat(),
            "data_sig": data_sig,
            "source": source,
        }

        file_exists = STATS_PATH.exists()
        with STATS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        models_logger.info(
            "Model stats: %s cfg=%s for %s type=%s rmse=%.4f source=%s",
            model_name,
            config_id,
            row["ticker"],
            model_type or "",
            float(rmse) if rmse is not None else float("nan"),
            source,
        )

    except Exception:
        logger.exception("Failed to append model stats")


# ---------- Основной пайплайн ----------

def train_select_and_forecast(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    force_retrain: bool = False,
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(
        "train_select_and_forecast: ticker=%s, len=%d, force_retrain=%s",
        ticker,
        len(df),
        force_retrain,
    )

    y = df["Close"].copy()
    val_steps = min(30, max(10, len(y) // 10))
    future_idx = _make_future_index(df.index[-1], 30, ticker or "")

    # ---------- 1. Fast path: forecasts cache ----------
    if ticker and not force_retrain:
        cached = load_cached_forecasts_if_fresh(df, ticker, MODEL_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    # ---------- 2. sig + keys ----------
    data_sig = _make_data_signature(df)
    params = {
        "val_steps": val_steps,
        "disable_lstm": os.getenv("DISABLE_LSTM", "0") == "1",
    }
    model_key = model_cache.make_cache_key(ticker or "N/A", "auto", params, data_sig)
    fc_key = model_cache.make_forecasts_key(ticker or "N/A", data_sig)

    # ---------- 3. init ----------
    best_dict: Dict[str, Optional[str]] = {"name": None}
    metrics: Dict[str, Optional[float]] = {"rmse": None, "mape": None}
    fcst_best_df: Optional[pd.DataFrame] = None
    fcst_avg_all_df: Optional[pd.DataFrame] = None
    fcst_avg_top3_df: Optional[pd.DataFrame] = None

    try:
        last_close = float(y.iloc[-1])
    except Exception:
        last_close = 1.0

    def _save_three(
        best_name: str,
        metrics_obj: Dict[str, Any],
        fb: pd.DataFrame,
        fa: pd.DataFrame,
        ft: pd.DataFrame,
    ):
        meta = {
            "best_name": best_name,
            "metrics": metrics_obj,
            "trained_at": int(time.time()),
            "model_version": MODEL_VERSION,
            "data_sig": data_sig,
            "ticker": (ticker or "N/A").upper(),
            "fc_key": fc_key,  # чтобы /history мог найти PNG
        }
        model_cache.save_forecasts(fc_key, fb, fa, ft, meta)

    # ---------- 4. Try cached MODELS ----------

    # --- sklearn (RandomForest) ---
    if ticker and not force_retrain:
        skl_model, skl_meta = model_cache.load_sklearn_model(model_key)
        if skl_model is not None and _is_fresh(skl_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                lag = int((skl_meta.get("extra") or {}).get("lag", 30))
                arr = y.values.astype(float)
                if len(arr) >= lag + 1:
                    last_window = arr[-lag:].copy()
                    preds = []
                    for _ in range(30):
                        yhat = skl_model.predict(last_window.reshape(1, -1))[0]
                        preds.append(float(yhat))
                        last_window = np.roll(last_window, -1)
                        last_window[-1] = yhat

                    preds_arr = np.array(preds, dtype=float)
                    sane = _sanitize_forecast_array(preds_arr, last_close)

                    # sanity-check cached model forecast
                    if _forecast_sanity_ok(sane, last_close):
                        fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)
                        best_dict["name"] = skl_meta.get("name", "cached_sklearn")
                        metrics = skl_meta.get("metrics", {"rmse": None, "mape": None})
                        models_logger.info(
                            "Using cached SKLEARN model=%s for %s rmse=%.4f",
                            best_dict["name"],
                            (ticker or "N/A"),
                            float(metrics.get("rmse") or float("nan")),
                        )
                    else:
                        logger.warning("Cached RF forecast insane/flat -> ignore cache for %s", ticker)
            except Exception:
                logger.exception("Error using cached sklearn model for %s", ticker)

    # --- SARIMAX ---
    if fcst_best_df is None and ticker and not force_retrain:
        sm_res, sm_meta = model_cache.load_statsmodels_result(model_key)
        if sm_res is not None and _is_fresh(sm_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                fcst = sm_res.get_forecast(steps=30).predicted_mean.values
                sane = _sanitize_forecast_array(fcst, last_close)

                if _forecast_sanity_ok(sane, last_close):
                    fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)
                    best_dict["name"] = sm_meta.get("name", "cached_sarimax")
                    metrics = sm_meta.get("metrics", {"rmse": None, "mape": None})
                    models_logger.info(
                        "Using cached SARIMAX model=%s for %s rmse=%.4f",
                        best_dict["name"],
                        (ticker or "N/A"),
                        float(metrics.get("rmse") or float("nan")),
                    )
                else:
                    logger.warning("Cached SARIMAX forecast insane/flat -> ignore cache for %s", ticker)
            except Exception:
                logger.exception("Error using cached SARIMAX model for %s", ticker)

    # --- LSTM ---
    if fcst_best_df is None and ticker and not force_retrain:
        tf_model, tf_meta = model_cache.load_tf_model(model_key)
        if tf_model is not None and _is_fresh(tf_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                extra = tf_meta.get("extra") or {}
                mu = float(extra["mu"])
                sigma = float(extra["sigma"])
                window = int(extra["window"])

                # защита от sigma=0/NaN в старом кэше
                if not np.isfinite(sigma) or sigma <= 1e-9:
                    raise ValueError(f"Bad sigma in cached LSTM meta: {sigma}")

                arr = y.values.astype("float32").reshape(-1, 1)
                norm = (arr - mu) / sigma
                last_seq = norm[-window:, 0].reshape(1, window, 1)

                preds = []
                for _ in range(30):
                    yhat = tf_model.predict(last_seq, verbose=0).reshape(-1)[0]
                    preds.append(float(yhat * sigma + mu))
                    last_seq = np.concatenate(
                        [last_seq[0, 1:, 0], [yhat]]
                    ).reshape(1, window, 1)

                preds_arr = np.array(preds, dtype=float)
                sane = _sanitize_forecast_array(preds_arr, last_close)

                if _forecast_sanity_ok(sane, last_close):
                    fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)
                    best_dict["name"] = tf_meta.get("name", "cached_lstm")
                    metrics = tf_meta.get("metrics", {"rmse": None, "mape": None})
                    models_logger.info(
                        "Using cached LSTM model=%s for %s rmse=%.4f",
                        best_dict["name"],
                        (ticker or "N/A"),
                        float(metrics.get("rmse") or float("nan")),
                    )
                else:
                    logger.warning("Cached LSTM forecast insane/flat -> ignore cache for %s", ticker)

            except Exception:
                logger.exception("Error using cached LSTM model for %s", ticker)

    # ---------- 5. If best from cached model ----------
    if fcst_best_df is not None:
        if ENSEMBLES_ENABLED:
            _, candidates = select_and_fit_with_candidates(
                y,
                val_steps=val_steps,
                horizon=WF_HORIZON,
                eval_tag=ticker,
                save_plots=False,
            )
            fcst_avg_all_df, fcst_avg_top3_df = _build_ensembles_from_candidates(
                y, candidates, future_idx
            )
            if fcst_avg_all_df.empty:
                fcst_avg_all_df = fcst_best_df.copy()
            if fcst_avg_top3_df.empty:
                fcst_avg_top3_df = fcst_best_df.copy()
        else:
            fcst_avg_all_df = fcst_best_df.copy()
            fcst_avg_top3_df = fcst_best_df.copy()

        _save_three(
            best_dict["name"] or "cached_best",
            metrics,
            fcst_best_df,
            fcst_avg_all_df,
            fcst_avg_top3_df,
        )
        return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df

    # ---------- 6. Train from scratch ----------
    best, candidates = select_and_fit_with_candidates(
        y,
        val_steps=val_steps,
        horizon=WF_HORIZON,
        eval_tag=ticker,
        save_plots=True,
    )

    # Прогноз лучшей модели
    raw_fcst_30 = refit_and_forecast_30d(y, best)
    last_close = float(y.iloc[-1])

    sanitized = _sanitize_forecast_array(raw_fcst_30.values, last_close)

    # fallback на 2-го кандидата, если winner слишком плоский
    winner_used = best
    if _is_too_flat(sanitized, last_close) and len(candidates) >= 2:
        logger.warning("Forecast too flat -> fallback to 2nd best model")
        second = sorted(candidates, key=lambda c: float(c.rmse))[1]
        raw2 = refit_and_forecast_30d(y, second)
        sanitized2 = _sanitize_forecast_array(raw2.values, last_close)
        if _forecast_sanity_ok(sanitized2, last_close):
            sanitized = sanitized2
            winner_used = second

    # IMPORTANT: y_fcst_30 создаём ВСЕГДА
    y_fcst_30 = pd.Series(sanitized, index=future_idx)
    fcst_best_df = pd.DataFrame({"forecast": y_fcst_30.values}, index=future_idx)

    if ENSEMBLES_ENABLED:
        fcst_avg_all_df, fcst_avg_top3_df = _build_ensembles_from_candidates(
            y, candidates, future_idx
        )
        if fcst_avg_all_df.empty:
            fcst_avg_all_df = fcst_best_df.copy()
        if fcst_avg_top3_df.empty:
            fcst_avg_top3_df = fcst_best_df.copy()
    else:
        fcst_avg_all_df = fcst_best_df.copy()
        fcst_avg_top3_df = fcst_best_df.copy()

    # Метрики по winner_used
    rmse = mean_squared_error(
        y.iloc[-val_steps:], winner_used.yhat_val[-val_steps:], squared=False
    )
    try:
        mape = mean_absolute_percentage_error(
            y.iloc[-val_steps:], winner_used.yhat_val[-val_steps:]
        )
    except Exception:
        mape = np.nan

    best_dict = {"name": winner_used.name}
    metrics = {"rmse": float(rmse), "mape": float(mape) if mape == mape else None}

    models_logger.info(
        "Final WF metrics for %s: model=%s rmse=%.4f mape=%s",
        (ticker or "N/A"),
        winner_used.name,
        float(rmse),
        "NA" if mape != mape else f"{float(mape):.4f}",
    )

    # ---------- 7. Save winner model + forecasts ----------
    try:
        if ticker:
            meta_model = {
                "name": winner_used.name,
                "trained_at": int(time.time()),
                "metrics": metrics,
                "extra": getattr(winner_used, "extra", {}),
                "model_version": MODEL_VERSION,
                "data_sig": data_sig,
                "ticker": (ticker or "N/A").upper(),
            }

            if winner_used.extra.get("type") == "rf":
                rf_obj, lag = winner_used.model_obj
                meta_model["extra"] = {"lag": int(lag)}
                model_cache.save_sklearn_model(model_key, rf_obj, meta_model)

            elif winner_used.extra.get("type") == "sarimax":
                order, seas, trend, _ = winner_used.model_obj
                m = sm.tsa.SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seas,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = m.fit(disp=False)
                meta_model["extra"] = {
                    "order": order,
                    "seasonal_order": seas,
                    "trend": trend,
                }
                model_cache.save_statsmodels_result(model_key, res, meta_model)

            elif winner_used.extra.get("type") == "lstm":
                mo = winner_used.model_obj
                if len(mo) == 4 and mo[-1] == "returns":
                    model, (mu, sigma), window, _ = mo
                else:
                    model, (mu, sigma), window = mo
                meta_model["extra"] = {
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "window": int(window),
                }
                model_cache.save_tf_model(model_key, model, meta_model)

            _save_three(
                winner_used.name,
                metrics,
                fcst_best_df,
                fcst_avg_all_df,
                fcst_avg_top3_df,
            )

    except Exception:
        logger.exception("Saving model/forecasts failed for %s", ticker)

    # ---------- 8. Log stats ----------
    try:
        model_type = None
        extra = getattr(winner_used, "extra", None)
        if isinstance(extra, dict):
            model_type = extra.get("type")

        _append_model_stats(
            ticker=ticker,
            model_name=winner_used.name,
            model_type=model_type,
            metrics=metrics,
            data_sig=data_sig,
            n_points=len(df),
            start_dt=df.index[0],
            end_dt=df.index[-1],
            val_steps=val_steps,
            extra=extra,
            source="train",
        )
    except Exception:
        logger.exception("Failed to log model stats for ticker=%s", ticker)

    logger.info(
        "Trained from scratch: ticker=%s winner=%s rmse=%.4f",
        ticker,
        best_dict["name"],
        metrics["rmse"] if metrics["rmse"] is not None else float("nan"),
    )
    return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df
