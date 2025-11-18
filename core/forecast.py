"""forecast.py — пайплайн обучения моделей и построения 30-дневного прогноза."""

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

MODEL_VERSION = "v3"
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
FC_MAX_MULT = float(os.getenv("FC_MAX_MULT", "5.0"))  # макс. ×5 от последней цены
FC_MIN_MULT = float(os.getenv("FC_MIN_MULT", "0.2"))  # мин. ×0.2
FC_MAX_DAILY_CHG = float(os.getenv("FC_MAX_DAILY_CHG", "0.5"))  # макс. дневной скачок 50%


def _sanitize_forecast_array(arr: np.ndarray, last_close: float) -> Optional[np.ndarray]:
    arr = np.asarray(arr, dtype=float)

    # 1) базовая проверка на NaN/inf
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    if not np.isfinite(last_close) or last_close <= 0:
        return None

    # 2) запрет на нулевые/отрицательные цены
    if np.any(arr <= 0):
        return None

    # 3) очень мягкие границы, только от совсем выноса в космос
    lo = last_close * FC_MIN_MULT   # но FC_MIN_MULT сделай чем-то типа 0.01
    hi = last_close * FC_MAX_MULT   # а FC_MAX_MULT — 50 или 100

    if np.min(arr) < lo or np.max(arr) > hi:
        # считаем действительно поехавшим
        return None

    # 4) ДНЕВНЫЕ СКАЧКИ НЕ РУБИМ ЖЁСТКО
    # можно либо вообще убрать этот блок, либо сильно ослабить:
    rel = np.diff(arr) / arr[:-1]
    if np.any(np.abs(rel) > FC_MAX_DAILY_CHG):
        # если сильно нервничаешь — логируем, но НЕ превращаем в flat
        logger.warning(
            "Forecast has big daily moves (max=%.2f). Leaving as is.",
            float(np.max(np.abs(rel)))
        )
        # и просто возвращаем arr без обнуления
        return arr

    return arr


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

    Плюс sanity-check: выкидываем явных «поехавших» кандидатов.
    """
    last_close = float(y.iloc[-1])
    if not np.isfinite(last_close) or last_close == 0.0:
        last_close = 1.0

    def _safe_fcst_for_candidate(cand) -> Optional[np.ndarray]:
        try:
            s = refit_and_forecast_30d(y, cand)
            arr = s.values.astype(float)

            arr = _sanitize_forecast_array(arr, last_close)
            if arr is None:
                logger.warning(
                    "Skip candidate %s: insane or invalid forecast",
                    getattr(cand, "name", "?"),
                )
                return None

            return arr
        except Exception:
            logger.exception(
                "refit_and_forecast_30d failed for candidate %s",
                getattr(cand, "name", "?"),
            )
            return None

    # --- среднее по ВСЕМ кандидатам (после фильтрации) ---
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

    # 1) считаем сигнатуру и ключи кэша
    data_sig = _make_data_signature(df)
    params = {
        "val_steps": val_steps,
        "disable_lstm": os.getenv("DISABLE_LSTM", "0") == "1",
    }
    model_key = model_cache.make_cache_key(ticker or "N/A", "auto", params, data_sig)
    fc_key = model_cache.make_forecasts_key(ticker or "N/A", data_sig)

    try:
        logger.debug(
            "DSIG: ticker=%s len=%d last_dt=%s model_key=%s fc_key=%s",
            ticker,
            len(df),
            pd.to_datetime(df.index[-1]).date().isoformat(),
            model_key,
            fc_key,
        )
    except Exception:
        pass

    # ---------- 2. Пробуем достать 3 прогноза из кэша ----------
    if ticker and not force_retrain:
        fb, fa, ft, fmeta = model_cache.load_forecasts(fc_key)
        if fb is not None and fa is not None and ft is not None and fmeta is not None:
            last_close = float(df["Close"].iloc[-1])

            # sanity-check кэшированного best-прогноза
            arr = fb["forecast"].values.astype(float)
            arr_ok = _sanitize_forecast_array(arr, last_close)

            if (
                _is_fresh(fmeta, MODEL_CACHE_TTL_SECONDS)
                and fmeta.get("data_sig") == data_sig
                and arr_ok is not None
            ):
                logger.info("Forecasts cache HIT (fresh + sane) for %s", ticker)
                best_dict = {"name": fmeta.get("best_name", "cached_best")}
                metrics = fmeta.get("metrics", {"rmse": None, "mape": None})
                models_logger.info(
                    "Using cached FORECASTS for %s: best=%s rmse=%.4f",
                    (ticker or "N/A"),
                    fmeta.get("best_name", "cached_best"),
                    float((fmeta.get("metrics") or {}).get("rmse") or float("nan")),
                )
                fb = fb.copy()
                fb["forecast"] = arr_ok
                return best_dict, metrics, fb, fa, ft
            else:
                logger.info(
                    "Forecasts cache for %s is stale or insane → recompute", ticker
                )

    # ---------- 3. Пробуем достать модель-победителя из кэша ----------
    best_dict: Dict[str, Optional[str]] = {"name": None}
    metrics: Dict[str, Optional[float]] = {"rmse": None, "mape": None}
    fcst_best_df: Optional[pd.DataFrame] = None
    fcst_avg_all_df: Optional[pd.DataFrame] = None
    fcst_avg_top3_df: Optional[pd.DataFrame] = None

    # последняя цена — нужна для sanity-check прогнозов из кэшированных моделей
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
        }
        model_cache.save_forecasts(fc_key, fb, fa, ft, meta)

    # --- sklearn (RF) ---
    if ticker and not force_retrain:
        skl_model, skl_meta = model_cache.load_sklearn_model(model_key)
        if skl_model is not None and _is_fresh(
            skl_meta or {}, MODEL_CACHE_TTL_SECONDS
        ):
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
                    if sane is None:
                        logger.warning(
                            "Cached RF model produced insane forecast for %s — falling back to flat.",
                            ticker,
                        )
                        sane = np.full(30, last_close, dtype=float)

                    fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)

                    best_dict["name"] = skl_meta.get("name", "cached_sklearn")
                    metrics = skl_meta.get("metrics", {"rmse": None, "mape": None})
                    models_logger.info(
                        "Using cached SKLEARN model=%s for %s rmse=%.4f",
                        best_dict["name"],
                        (ticker or "N/A"),
                        float(metrics.get("rmse") or float("nan")),
                    )
                    logger.info("Loaded cached sklearn model for %s", ticker)
            except Exception:
                logger.exception("Error using cached sklearn model for %s", ticker)

    # --- SARIMAX ---
    if fcst_best_df is None and ticker and not force_retrain:
        sm_res, sm_meta = model_cache.load_statsmodels_result(model_key)
        if sm_res is not None and _is_fresh(sm_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                fcst = sm_res.get_forecast(steps=30).predicted_mean.values
                sane = _sanitize_forecast_array(fcst, last_close)
                if sane is None:
                    logger.warning(
                        "Cached SARIMAX produced insane forecast for %s — falling back to flat.",
                        ticker,
                    )
                    sane = np.full(30, last_close, dtype=float)

                fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)
                best_dict["name"] = sm_meta.get("name", "cached_sarimax")
                metrics = sm_meta.get("metrics", {"rmse": None, "mape": None})
                logger.info("Loaded cached SARIMAX model for %s", ticker)
                models_logger.info(
                    "Using cached SARIMAX model=%s for %s rmse=%.4f",
                    best_dict["name"],
                    (ticker or "N/A"),
                    float(metrics.get("rmse") or float("nan")),
                )
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
                if sane is None:
                    logger.warning(
                        "Cached LSTM model produced insane forecast for %s — falling back to flat.",
                        ticker,
                    )
                    sane = np.full(30, last_close, dtype=float)

                fcst_best_df = pd.DataFrame({"forecast": sane}, index=future_idx)
                best_dict["name"] = tf_meta.get("name", "cached_lstm")
                metrics = tf_meta.get("metrics", {"rmse": None, "mape": None})
                models_logger.info(
                    "Using cached LSTM model=%s for %s rmse=%.4f",
                    best_dict["name"],
                    (ticker or "N/A"),
                    float(metrics.get("rmse") or float("nan")),
                )
                logger.info("Loaded cached LSTM model for %s", ticker)
            except Exception:
                logger.exception("Error using cached LSTM model for %s", ticker)

        # ---------- 4. Если best есть из кэша: считаем / не считаем ансамбли и сохраняем три прогноза ----------
    if fcst_best_df is not None:
        if ENSEMBLES_ENABLED:
            # как раньше — считаем кандидатов и ансамбли
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
            # ансамбли выключены — просто используем лучший прогноз как proxy
            fcst_avg_all_df = fcst_best_df.copy()
            fcst_avg_top3_df = fcst_best_df.copy()

        _save_three(
            best_dict["name"] or "cached_best",
            metrics,
            fcst_best_df,
            fcst_avg_all_df,
            fcst_avg_top3_df,
        )
        models_logger.info(
            "Using cached BEST model=%s for %s (fresh, ENSEMBLES_ENABLED=%s)",
            best_dict.get("name") or "cached",
            (ticker or "N/A"),
            ENSEMBLES_ENABLED,
        )
        return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df


    # ---------- 5. Обучение с нуля ----------
    best, candidates = select_and_fit_with_candidates(
        y,
        val_steps=val_steps,
        horizon=WF_HORIZON,
        eval_tag=ticker,
        save_plots=True,
    )

    # Логи по всем кандидатам и победителю
    try:
        cand_lines = []
        for c in candidates:
            rmse_c = getattr(c, "rmse", None)
            cand_lines.append(
                f"{getattr(c, 'name', '?')}={rmse_c:.4f}"
                if rmse_c is not None
                else f"{getattr(c, 'name', '?')}=NA"
            )

        models_logger.info(
            "Model candidates for %s: %s",
            (ticker or "N/A"),
            ", ".join(cand_lines),
        )

        best_rmse = getattr(best, "rmse", None)
        models_logger.info(
            "Winner model for %s: %s rmse=%.4f",
            (ticker or "N/A"),
            getattr(best, "name", "?"),
            best_rmse if best_rmse is not None else float("nan"),
        )
    except Exception:
        logger.exception("Failed to log model candidates for ticker=%s", ticker)

    # Прогноз лучшей модели
    raw_fcst_30 = refit_and_forecast_30d(y, best)
    last_close = float(y.iloc[-1])

    sanitized = _sanitize_forecast_array(raw_fcst_30.values, last_close)
    if sanitized is None:
        logger.warning(
            "Winner model '%s' produced insane forecast for %s — falling back to flat.",
            best.name,
            ticker,
        )
        sanitized = np.full(30, last_close, dtype=float)

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
        # ансамбли выключены — оба варианта равны лучшей модели
        fcst_avg_all_df = fcst_best_df.copy()
        fcst_avg_top3_df = fcst_best_df.copy()

    # Метрики
    rmse = mean_squared_error(
        y.iloc[-val_steps:], best.yhat_val[-val_steps:], squared=False
    )
    try:
        mape = mean_absolute_percentage_error(
            y.iloc[-val_steps:], best.yhat_val[-val_steps:]
        )
    except Exception:
        mape = np.nan

    models_logger.info(
        "Final WF metrics for %s: model=%s rmse=%.4f mape=%s",
        (ticker or "N/A"),
        best.name,
        float(rmse),
        "NA" if mape != mape else f"{float(mape):.4f}",
    )

    best_dict = {"name": best.name}
    metrics = {"rmse": float(rmse), "mape": float(mape) if mape == mape else None}

    # ---------- 6. Сохраняем модель-победителя и три прогноза ----------
    try:
        if ticker:
            meta_model = {
                "name": best.name,
                "trained_at": int(time.time()),
                "metrics": metrics,
                "extra": getattr(best, "extra", {}),
                "model_version": MODEL_VERSION,
                "data_sig": data_sig,
                "ticker": (ticker or "N/A").upper(),
            }

            if best.extra.get("type") == "rf":
                rf_obj, lag = best.model_obj
                meta_model["extra"] = {"lag": int(lag)}
                model_cache.save_sklearn_model(model_key, rf_obj, meta_model)

            elif best.extra.get("type") == "sarimax":
                order, seas, trend, _ = best.model_obj
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

            elif best.extra.get("type") == "lstm":
                mo = best.model_obj
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
                best.name,
                metrics,
                fcst_best_df,
                fcst_avg_all_df,
                fcst_avg_top3_df,
            )

    except Exception:
        logger.exception("Saving model/forecasts failed for %s", ticker)

    # Записываем статистику эффективности победившей модели
    try:
        model_type = None
        extra = getattr(best, "extra", None)
        if isinstance(extra, dict):
            model_type = extra.get("type")

        _append_model_stats(
            ticker=ticker,
            model_name=best.name,
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
