"""forecast.py"""
import hashlib
import io
import json
import os
import time
from typing import Dict, Optional, Tuple, List, Any

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm

from . import model_cache
from .models import refit_and_forecast_30d, select_and_fit_with_candidates

logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"
WF_HORIZON = int(os.getenv("WF_HORIZON", "5"))
MODEL_CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))


# ---------- Data signature для кэша ----------

def _make_data_signature(df: pd.DataFrame, tail_len: int = 256) -> str:
    """
    Сигнатура данных для кэша моделей/прогнозов.

    Зависит от:
      - числа точек (n)
      - первой и последней даты
      - sha1-хеша последних tail_len значений Close (в float32)

    Если приходит новая свечка или меняется хвост истории — сигнатура меняется.
    При пересчитывании тех же данных — остаётся той же.
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
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _make_future_index(last_ts: pd.Timestamp, periods: int, ticker: str) -> pd.DatetimeIndex:
    t = (ticker or "").upper()
    is_crypto = t.endswith("-USD")
    is_fx = t.endswith("=X")
    start = last_ts + pd.Timedelta(days=1)
    if is_crypto or is_fx:
        return pd.date_range(start=start, periods=periods, freq="D")
    return pd.bdate_range(start=start, periods=periods)


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

    Плюс sanity-check: выкидываем явных «поехавших» кандидатов, у которых
    прогнозы в десятки раз больше/меньше текущей цены или содержат NaN/inf.
    """
    last_close = float(y.iloc[-1])
    if not np.isfinite(last_close) or last_close == 0.0:
        last_close = 1.0

    SCALE_FACTOR = 50.0  # максимум ×50 от последней цены

    def _safe_fcst_for_candidate(cand) -> Optional[np.ndarray]:
        try:
            s = refit_and_forecast_30d(y, cand)
            arr = s.values.astype(float)
            if arr.size == 0:
                return None
            if not np.all(np.isfinite(arr)):
                logger.warning("Skip candidate %s: non-finite values in forecast",
                               getattr(cand, "name", "?"))
                return None
            max_abs = float(np.max(np.abs(arr)))
            if max_abs > abs(last_close) * SCALE_FACTOR:
                logger.warning(
                    "Skip candidate %s: insane forecast scale max=%.3g last_close=%.3g",
                    getattr(cand, "name", "?"),
                    max_abs,
                    last_close,
                )
                return None
            return arr
        except Exception:
            logger.exception("refit_and_forecast_30d failed for candidate %s",
                             getattr(cand, "name", "?"))
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
            if _is_fresh(fmeta, MODEL_CACHE_TTL_SECONDS) and fmeta.get("data_sig") == data_sig:
                logger.info("Forecasts cache HIT (fresh) for %s", ticker)
                best_dict = {"name": fmeta.get("best_name", "cached_best")}
                metrics = fmeta.get("metrics", {"rmse": None, "mape": None})
                return best_dict, metrics, fb, fa, ft
            else:
                logger.info("Forecasts cache STALE for %s → recompute", ticker)

    # ---------- 3. Пробуем достать модель-победителя из кэша ----------
    best_dict: Dict[str, Optional[str]] = {"name": None}
    metrics: Dict[str, Optional[float]] = {"rmse": None, "mape": None}
    fcst_best_df: Optional[pd.DataFrame] = None
    fcst_avg_all_df: Optional[pd.DataFrame] = None
    fcst_avg_top3_df: Optional[pd.DataFrame] = None

    def _save_three(best_name: str, metrics_obj: Dict[str, Any],
                    fb: pd.DataFrame, fa: pd.DataFrame, ft: pd.DataFrame):
        """Сохранить три варианта прогноза + метаданные."""
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
                    fcst_best_df = pd.DataFrame(
                        {"forecast": np.array(preds)}, index=future_idx
                    )
                    best_dict["name"] = skl_meta.get("name", "cached_sklearn")
                    metrics = skl_meta.get("metrics", {"rmse": None, "mape": None})
                    logger.info("Loaded cached sklearn model for %s", ticker)
            except Exception:
                logger.exception("Error using cached sklearn model for %s", ticker)

    # --- SARIMAX ---
    if fcst_best_df is None and ticker and not force_retrain:
        sm_res, sm_meta = model_cache.load_statsmodels_result(model_key)
        if sm_res is not None and _is_fresh(sm_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                fcst = sm_res.get_forecast(steps=30).predicted_mean.values
                fcst_best_df = pd.DataFrame({"forecast": fcst}, index=future_idx)
                best_dict["name"] = sm_meta.get("name", "cached_sarimax")
                metrics = sm_meta.get("metrics", {"rmse": None, "mape": None})
                logger.info("Loaded cached SARIMAX model for %s", ticker)
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

                fcst_best_df = pd.DataFrame(
                    {"forecast": np.array(preds)}, index=future_idx
                )
                best_dict["name"] = tf_meta.get("name", "cached_lstm")
                metrics = tf_meta.get("metrics", {"rmse": None, "mape": None})
                logger.info("Loaded cached LSTM model for %s", ticker)
            except Exception:
                logger.exception("Error using cached LSTM model for %s", ticker)

    # ---------- 4. Если best есть из кэша: считаем ансамбли и сохраняем три прогноза ----------
    if fcst_best_df is not None:
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

        _save_three(
            best_dict["name"] or "cached_best",
            metrics,
            fcst_best_df,
            fcst_avg_all_df,
            fcst_avg_top3_df,
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

    # Прогноз лучшей модели
    y_fcst_30 = refit_and_forecast_30d(y, best)
    fcst_best_df = pd.DataFrame({"forecast": y_fcst_30.values}, index=future_idx)

    # Ансамбли
    fcst_avg_all_df, fcst_avg_top3_df = _build_ensembles_from_candidates(
        y, candidates, future_idx
    )
    if fcst_avg_all_df.empty:
        fcst_avg_all_df = fcst_best_df.copy()
    if fcst_avg_top3_df.empty:
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

            _save_three(best.name, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df)

    except Exception:
        logger.exception("Saving model/forecasts failed for %s", ticker)

    logger.info(
        "Trained from scratch: ticker=%s winner=%s rmse=%.4f",
        ticker,
        best_dict["name"],
        metrics["rmse"] if metrics["rmse"] is not None else float("nan"),
    )
    return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df


# ---------- Плоттеры ----------

def make_plot_image(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    markers: list = None,
    title_suffix: str = "",
) -> io.BytesIO:
    plt.figure(figsize=(10, 5))
    plt.plot(history_df.index, history_df["Close"], label="History")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")

    title = f"{ticker}: History & 30-Day Forecast"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)

    try:
        if markers:
            for m in markers:
                try:
                    if isinstance(m, dict):
                        side = m.get("side", "long")
                        dt = m.get("sell") if side == "short" else m.get("buy")
                    else:
                        dt, _label = m
                    if dt is None:
                        continue
                    plt.axvline(dt, linestyle="--", alpha=0.35)
                except Exception:
                    continue
    except Exception:
        pass

    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def export_plot_pdf(history_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history_df.index, history_df["Close"], label="History")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")
    plt.title(f"{ticker}: History & 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close()
