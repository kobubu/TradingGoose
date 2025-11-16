"""forecast.py"""
import hashlib
import io
import json
import os
import time
from typing import Dict, Optional, Tuple, List

import logging 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from core.subs import can_consume, consume_one, get_status, is_pro, get_limits
import statsmodels.api as sm

from . import model_cache
from .models import refit_and_forecast_30d, select_and_fit_with_candidates

logger = logging.getLogger(__name__) 

MODEL_VERSION = "v1"
WF_HORIZON = int(os.getenv("WF_HORIZON", "5"))
MODEL_CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))


def _make_data_signature(df: pd.DataFrame) -> str:
    last_dt = pd.to_datetime(df.index[-1]).date().isoformat()
    # Делаем сигнатуру устойчивее: убираем цену вовсе ИЛИ округляем грубо.
    payload = {
        "last_dt": last_dt,
        "n": int(len(df)),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _make_future_index(last_ts: pd.Timestamp, periods: int, ticker: str) -> pd.DatetimeIndex:
    t = (ticker or "").upper()
    is_crypto = t.endswith("-USD")
    is_fx = t.endswith("=X")
    start = last_ts + pd.Timedelta(days=1)
    if is_crypto or is_fx:
        return pd.date_range(start=start, periods=periods, freq="D")
    return pd.bdate_range(start=start, periods=periods)


def _build_ensembles_from_candidates(
    y: pd.Series,
    candidates,
    future_idx: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_fcsts: List[np.ndarray] = []
    for c in candidates:
        try:
            s = refit_and_forecast_30d(y, c)
            all_fcsts.append(s.values.astype(float))
        except Exception:
            pass
    if all_fcsts:
        mat = np.vstack(all_fcsts)
        avg_all = np.nanmean(mat, axis=0)
        fcst_avg_all = pd.DataFrame({"forecast": avg_all}, index=future_idx)
    else:
        fcst_avg_all = pd.DataFrame({"forecast": []})

    top3 = sorted(candidates, key=lambda c: float(c.rmse))[:3]
    top3_mat: List[np.ndarray] = []
    for c in top3:
        try:
            s = refit_and_forecast_30d(y, c)
            top3_mat.append(s.values.astype(float))
        except Exception:
            pass
    if top3_mat:
        top3_avg = np.nanmean(np.vstack(top3_mat), axis=0)
        fcst_avg_top3 = pd.DataFrame({"forecast": top3_avg}, index=future_idx)
    else:
        fcst_avg_top3 = pd.DataFrame({"forecast": []})

    return fcst_avg_all, fcst_avg_top3


def _is_fresh(meta: Dict[str, any], ttl: int) -> bool:
    try:
        return (meta.get("model_version") == MODEL_VERSION
                and (time.time() - int(meta.get("trained_at", 0)) <= ttl))
    except Exception:
        return False


def train_select_and_forecast(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    force_retrain: bool = False,
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("train_select_and_forecast: ticker=%s, len=%d, force_retrain=%s",
                ticker, len(df), force_retrain)
    y = df["Close"].copy()
    val_steps = min(30, max(10, len(y) // 10))
    future_idx = _make_future_index(df.index[-1], 30, ticker or "")

    # 1) ключи КЭША сначала
    data_sig = _make_data_signature(df)
    params = {"val_steps": val_steps, "disable_lstm": os.getenv("DISABLE_LSTM", "0") == "1"}
    model_key = model_cache.make_cache_key(ticker or "N/A", "auto", params, data_sig)
    fc_key = model_cache.make_forecasts_key(ticker or "N/A", data_sig)

    # 2) теперь безопасно логируем
    try:
        print(f"DEBUG: ticker={ticker}  len={len(df)}  last_dt={pd.to_datetime(df.index[-1]).date().isoformat()}")
        print(f"DEBUG: model_key={model_key}")
        print(f"DEBUG: fc_key={fc_key}")
    except Exception:
        pass

    # ---------- сначала пробуем достать 3 прогноза из кеша ----------
    if ticker and not force_retrain:
        fb, fa, ft, fmeta = model_cache.load_forecasts(fc_key)
        if fb is not None and fa is not None and ft is not None and fmeta is not None:
            if _is_fresh(fmeta, MODEL_CACHE_TTL_SECONDS) and fmeta.get("data_sig") == data_sig:
                print("DEBUG: forecasts cache HIT (fresh) → use saved 3x forecasts")
                best_dict = {"name": fmeta.get("best_name", "cached_best")}
                metrics = fmeta.get("metrics", {"rmse": None, "mape": None})
                return best_dict, metrics, fb, fa, ft
            else:
                # иначе просто пересчитаем и перезапишем
                print("DEBUG: forecasts cache STALE → recompute & overwrite")

    # ---------- Далее логика: пытаемся использовать кэш модели для "best" ----------
    best_dict: Dict[str, Optional[str]] = {"name": None}
    metrics: Dict[str, Optional[float]] = {"rmse": None, "mape": None}
    fcst_best_df: Optional[pd.DataFrame] = None
    fcst_avg_all_df: Optional[pd.DataFrame] = None
    fcst_avg_top3_df: Optional[pd.DataFrame] = None

    def _save_three(best_name: str, metrics_obj: Dict[str, any], fb, fa, ft):
        """Сохранить три варианта прогноза + метаданные, включая тикер."""
        meta = {
            "best_name": best_name,
            "metrics": metrics_obj,
            "trained_at": int(time.time()),
            "model_version": MODEL_VERSION,
            "data_sig": data_sig,
            "ticker": (ticker or "N/A").upper(),
        }
        model_cache.save_forecasts(fc_key, fb, fa, ft, meta)

    # --- sklearn best
    if ticker and not force_retrain:
        skl_model, skl_meta = model_cache.load_sklearn_model(model_key)
        if skl_model is not None and _is_fresh(skl_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                lag = int(skl_meta.get("extra", {}).get("lag", 30))
                arr = y.values.astype(float)
                if len(arr) >= lag + 1:
                    last_window = arr[-lag:].copy()
                    preds = []
                    for _ in range(30):
                        yhat = skl_model.predict(last_window.reshape(1, -1))[0]
                        preds.append(float(yhat))
                        last_window = np.roll(last_window, -1)
                        last_window[-1] = yhat
                    fcst_best_df = pd.DataFrame({"forecast": np.array(preds)}, index=future_idx)
                    best_dict["name"] = skl_meta.get("name", "cached_sklearn")
                    metrics = skl_meta.get("metrics", {"rmse": None, "mape": None})
            except Exception:
                pass

    # --- SARIMAX best
    if fcst_best_df is None and ticker and not force_retrain:
        sm_res, sm_meta = model_cache.load_statsmodels_result(model_key)
        if sm_res is not None and _is_fresh(sm_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                fcst = sm_res.get_forecast(steps=30).predicted_mean.values
                fcst_best_df = pd.DataFrame({"forecast": fcst}, index=future_idx)
                best_dict["name"] = sm_meta.get("name", "cached_sarimax")
                metrics = sm_meta.get("metrics", {"rmse": None, "mape": None})
            except Exception:
                pass

    # --- LSTM best
    if fcst_best_df is None and ticker and not force_retrain:
        tf_model, tf_meta = model_cache.load_tf_model(model_key)
        if tf_model is not None and _is_fresh(tf_meta or {}, MODEL_CACHE_TTL_SECONDS):
            try:
                mu = float(tf_meta["extra"]["mu"])
                sigma = float(tf_meta["extra"]["sigma"])
                window = int(tf_meta["extra"]["window"])
                arr = y.values.astype("float32").reshape(-1, 1)
                norm = (arr - mu) / sigma
                last_seq = norm[-window:, 0].reshape(1, window, 1)
                preds = []
                for _ in range(30):
                    yhat = tf_model.predict(last_seq, verbose=0).reshape(-1)[0]
                    preds.append(float(yhat * sigma + mu))
                    last_seq = np.concatenate([last_seq[0, 1:, 0], [yhat]]).reshape(1, window, 1)
                fcst_best_df = pd.DataFrame({"forecast": np.array(preds)}, index=future_idx)
                best_dict["name"] = tf_meta.get("name", "cached_lstm")
                metrics = tf_meta.get("metrics", {"rmse": None, "mape": None})
            except Exception:
                pass

    # ---------- Если "best" уже есть из кэша — считаем кандидатов и ансамбли, потом сохраняем 3 прогноза ----------
    if fcst_best_df is not None:
        _, candidates = select_and_fit_with_candidates(
            y,
            val_steps=val_steps,
            horizon=WF_HORIZON,
            eval_tag=ticker,
            save_plots=False,
        )
        fcst_avg_all_df, fcst_avg_top3_df = _build_ensembles_from_candidates(y, candidates, future_idx)
        if fcst_avg_all_df.empty:
            fcst_avg_all_df = fcst_best_df.copy()
        if fcst_avg_top3_df.empty:
            fcst_avg_top3_df = fcst_best_df.copy()

        _save_three(best_dict["name"] or "cached_best", metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df)
        return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df

    # ---------- Обучение с нуля ----------
    best, candidates = select_and_fit_with_candidates(
        y,
        val_steps=val_steps,
        horizon=WF_HORIZON,
        eval_tag=ticker,
        save_plots=True,
    )

    # Прогноз лучшей
    y_fcst_30 = refit_and_forecast_30d(y, best)
    fcst_best_df = pd.DataFrame({"forecast": y_fcst_30.values}, index=future_idx)

    # Ансамбли
    fcst_avg_all_df, fcst_avg_top3_df = _build_ensembles_from_candidates(y, candidates, future_idx)
    if fcst_avg_all_df.empty:
        fcst_avg_all_df = fcst_best_df.copy()
    if fcst_avg_top3_df.empty:
        fcst_avg_top3_df = fcst_best_df.copy()

    # Метрики
    rmse = mean_squared_error(y.iloc[-val_steps:], best.yhat_val[-val_steps:], squared=False)
    try:
        mape = mean_absolute_percentage_error(y.iloc[-val_steps:], best.yhat_val[-val_steps:])
    except Exception:
        mape = np.nan
    best_dict = {"name": best.name}
    metrics = {"rmse": float(rmse), "mape": float(mape) if mape == mape else None}

    # Сохраняем модель(и) и ТРИ прогноза
    try:
        if ticker:
            # сохранить модель-победителя (как и раньше)
            meta_model = {
                "name": best.name,
                "trained_at": int(time.time()),
                "metrics": metrics,
                "extra": getattr(best, "extra", {}),
                "model_version": MODEL_VERSION,
                "data_sig": data_sig,
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
                meta_model["extra"] = {"order": order, "seasonal_order": seas, "trend": trend}
                model_cache.save_statsmodels_result(model_key, res, meta_model)
            elif best.extra.get("type") == "lstm":
                mo = best.model_obj
                if len(mo) == 4 and mo[-1] == "returns":
                    model, (mu, sigma), window, _ = mo
                else:
                    model, (mu, sigma), window = mo
                meta_model["extra"] = {"mu": float(mu), "sigma": float(sigma), "window": int(window)}
                model_cache.save_tf_model(model_key, model, meta_model)

            # сохранить ТРИ прогноза (через наш helper, с тикером)
            _save_three(best.name, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df)

    except Exception as e:
        print(f"[ERR] save_forecasts failed: {e}")
        raise

    print(f"DEBUG: trained from scratch. winner={best_dict['name']}, rmse={metrics['rmse']:.4f}")
    return best_dict, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df


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
            ymin, ymax = plt.ylim()
            for m in markers:
                if isinstance(m, dict):
                    side = m.get("side", "long")
                    dt = m.get("sell") if side == "short" else m.get("buy")
                else:
                    try:
                        dt, _label = m
                    except Exception:
                        continue

                if dt is None:
                    continue

                plt.axvline(dt, linestyle="--", alpha=0.35)
                # подписи можно не рисовать, чтобы не захламлять
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
