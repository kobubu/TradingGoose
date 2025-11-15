# core/models.py
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger("models")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras
from telegram.error import Forbidden

# Затем предупреждения
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# Затем GPU конфигурация
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

matplotlib.use("Agg")

WF_HORIZON = int(os.getenv("WF_HORIZON", "5"))
DISABLE_LSTM = os.getenv('DISABLE_LSTM', '0') == '1'

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


def _pad_to_val_steps(preds, val_steps, fill_value):
    """Дополняет прогнозы до нужной длины fill_value"""
    preds = np.asarray(preds, dtype=float)
    if len(preds) < val_steps:
        pad = np.full(val_steps - len(preds), float(fill_value), dtype=float)
        preds = np.concatenate([pad, preds])
    return preds


def _safe_mape(y_true, y_pred):
    """Безопасный расчет MAPE с обработкой ошибок"""
    try:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        return None


def _save_eval_plot(y_idx, y_true, y_pred, title, out_png):
    """Сохраняет график сравнения прогноза с реальными значениями"""
    try:
        plt.figure(figsize=(10, 4.5))
        plt.plot(y_idx, y_true, label="True")
        plt.plot(y_idx, y_pred, label="Pred")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    except Exception:
        logger.exception("Failed to save eval plot: %s", out_png)
    finally:
        plt.close()


def _save_eval_json(meta: Dict[str, Any], out_json: str):
    """Сохраняет метаданные модели в JSON файл"""
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.exception("Failed to save eval json: %s", out_json)


def _valid_preds(arr: np.ndarray, val_steps: int) -> bool:
    """Проверяет валидность прогнозов"""
    try:
        arr = np.asarray(arr, dtype=float)
        return (arr.ndim == 1) and (len(arr) == val_steps) and np.isfinite(arr).all()
    except Exception:
        return False


@dataclass
class ModelResult:
    """Класс для хранения результатов работы модели"""
    name: str
    yhat_val: np.ndarray
    rmse: float
    model_obj: Any
    extra: Dict[str, Any]


def _wf_random_forest(
    y: pd.Series,
    max_lag: int,
    val_steps: int,
    horizon: int = 1,
    rf_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, RandomForestRegressor]:
    """Walk-forward валидация для RandomForest с лаговыми признаками"""
    y_arr = y.astype(float).values
    start = len(y_arr) - val_steps
    n_valid = min(val_steps, max(0, len(y_arr) - (start + horizon - 1)))
    preds: List[float] = []

    if rf_params is None:
        rf_params = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    def make_X_y(arr: np.ndarray, lag: int):
        """Создает матрицу признаков и целей из временного ряда"""
        X, yy = [], []
        for t in range(lag, len(arr)):
            X.append(arr[t-lag:t])
            yy.append(arr[t])
        return np.array(X, dtype=float), np.array(yy, dtype=float)

    model = RandomForestRegressor(**rf_params)

    for i in range(n_valid):
        t = start + i
        train_arr = y_arr[:t]
        if len(train_arr) <= max_lag:
            preds.append(float(train_arr[-1]))
            continue

        X_tr, y_tr = make_X_y(train_arr, max_lag)
        model.fit(X_tr, y_tr)

        last_window = train_arr[-max_lag:].copy()
        yhat = None
        for _ in range(horizon):
            yhat = float(model.predict(last_window.reshape(1, -1))[0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        preds.append(float(yhat))

    preds = np.array(preds, dtype=float)
    fill = float(y_arr[start-1]) if start-1 >= 0 else float(y_arr[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)
    return preds, model


def _choose_sarimax_config(y_train: pd.Series):
    """Подбирает оптимальную конфигурацию SARIMAX через grid search"""
    y_train = y_train.astype(float)
    if len(y_train) < 20:
        return (1, 1, 1), (0, 0, 0, 5), 'n'

    P_list = [0, 1]
    Q_list = [0, 1]
    D_list = [0]
    s_list = [5, 7]
    p_list = [0, 1, 2]
    d_list = [0, 1]
    q_list = [0, 1, 2]
    trends = ['n', 'c']

    best = None
    best_aic = np.inf
    total = 0
    tried = 0

    for s in s_list:
        for P in P_list:
            for D in D_list:
                for Q in Q_list:
                    for p in p_list:
                        for d in d_list:
                            for q in q_list:
                                for trend in trends:
                                    total += 1
                                    try:
                                        m = sm.tsa.SARIMAX(
                                            y_train,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s),
                                            trend=trend,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                        )
                                        r = m.fit(disp=False)
                                        tried += 1
                                        aic = r.aic
                                        if np.isfinite(aic) and aic < best_aic:
                                            best_aic = aic
                                            best = ((p, d, q), (P, D, Q, s), trend)
                                    except Exception:
                                        continue

    if best is None:
        best = ((1, 1, 1), (0, 0, 0, 5), 'n')

    try:
        logger.debug(
            "SARIMAX grid search tried %d/%d configs, best=%s, best_aic=%.2f",
            tried, total, best, best_aic
        )
    except Exception:
        pass
    return best


def _wf_sarimax(y: pd.Series, val_steps: int, horizon: int = 1):
    """Walk-forward валидация для SARIMAX модели"""
    y = y.astype(float)
    start = len(y) - val_steps
    (order, seas, trend) = _choose_sarimax_config(y.iloc[:max(start, 30)])
    preds: List[float] = []
    last_fit = None

    n_valid = min(val_steps, max(0, len(y) - (start + horizon - 1)))
    for i in range(n_valid):
        t = start + i
        try:
            m = sm.tsa.SARIMAX(
                y.iloc[:t],
                order=order,
                seasonal_order=seas,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            r = m.fit(disp=False)
            fc = r.get_forecast(steps=horizon).predicted_mean
            yhat = float(fc.iloc[-1])
            if not np.isfinite(yhat):
                raise ValueError("Non-finite SARIMAX yhat")
            preds.append(yhat)
            last_fit = r
        except Exception:
            preds.append(float(y.iloc[t-1]))

    preds = np.array(preds, dtype=float)
    fill = float(y.iloc[start-1]) if start-1 >= 0 else float(y.iloc[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)
    return preds, (order, seas, trend, last_fit)


def _wf_lstm(
    y: pd.Series,
    val_steps: int,
    window: int = 30,
    epochs: int = 8,
    batch_size: int = 16,
    horizon: int = 1
):
    """Walk-forward валидация для LSTM на лог-доходностях"""
    p = y.astype(float).values
    if len(p) < window + val_steps + 5:
        raise ValueError("Слишком короткий ряд для LSTM по доходностям.")

    logp = np.log(p + 1e-9)
    r = np.diff(logp).astype("float32").reshape(-1, 1)

    start = len(r) - val_steps
    n_valid = min(val_steps, max(0, len(r) - (start + horizon - 1)))
    preds_prices: List[float] = []

    def build_model():
        """Создает архитектуру LSTM модели"""
        m = keras.Sequential([
            keras.layers.Input(shape=(window, 1)),
            keras.layers.GRU(64, return_sequences=False),
            keras.layers.Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    model = None
    last_mu, last_sigma = 0.0, 1.0

    for i in range(n_valid):
        t = start + i
        train = r[:t]
        mu = float(train.mean())
        sigma = float(train.std() + 1e-6)
        norm = (train - mu) / sigma

        if len(norm) <= window:
            preds_prices.append(float(p[t]))
            model = None
            last_mu, last_sigma = mu, sigma
            continue

        X, yy = [], []
        for k in range(window, len(norm)):
            X.append(norm[k-window:k, 0])
            yy.append(norm[k, 0])
        X = np.array(X)[..., None]
        yy = np.array(yy)

        if model is None:
            model = build_model()
            base_epochs = max(12, epochs)
        else:
            base_epochs = 4
        model.fit(X, yy, epochs=base_epochs, batch_size=batch_size, verbose=0)

        last_seq = norm[-window:, 0].reshape(1, window, 1)
        rhat_norm = None
        for _ in range(horizon):
            rhat_norm = model.predict(last_seq, verbose=0).reshape(-1)[0]
            last_seq = np.concatenate([last_seq[0, 1:, 0], [rhat_norm]]).reshape(1, window, 1)

        rhat = float(rhat_norm * sigma + mu)
        last_logp = float(logp[t])
        yhat_price = float(np.exp(last_logp + rhat))
        preds_prices.append(yhat_price)

        last_mu, last_sigma = mu, sigma

    preds_prices = np.array(preds_prices, dtype=float)
    fill_price = float(p[start]) if start >= 0 else float(p[0])
    preds_prices = _pad_to_val_steps(preds_prices, val_steps, fill_price)

    return preds_prices, (model, (last_mu, last_sigma), window, 'returns')


def select_and_fit_with_candidates(
    y: pd.Series,
    val_steps: int = 30,
    horizon: int = WF_HORIZON,
    eval_tag: str = None,
    save_plots: bool = False,
    artifacts_dir: str = None
) -> Tuple[ModelResult, List[ModelResult]]:
    """Выбирает лучшую модель по RMSE на валидационном периоде"""
    if artifacts_dir is None:
        artifacts_dir = ART_DIR
    os.makedirs(artifacts_dir, exist_ok=True)

    y = y.astype(float)
    if len(y) <= max(35, val_steps + 5):
        raise ValueError("Слишком короткий ряд для надёжной walk-forward валидации.")
    y_true = y.iloc[-val_steps:].values
    y_index = y.index[-val_steps:]

    candidates: List[ModelResult] = []

    # RandomForest семейство
    for lag in (20, 30, 60, 90):
        try:
            rf_preds, rf_obj = _wf_random_forest(y, max_lag=lag, val_steps=val_steps, horizon=horizon)
            if _valid_preds(rf_preds, val_steps):
                rmse = mean_squared_error(y_true, rf_preds, squared=False)
                res = ModelResult(
                    name=f"RandomForest(lag={lag})",
                    yhat_val=rf_preds,
                    rmse=float(rmse),
                    model_obj=(rf_obj, lag),
                    extra={"type": "rf", "lag": lag}
                )
                candidates.append(res)
                logger.debug("RF candidate lag=%d rmse=%.4f", lag, rmse)
                if save_plots:
                    tag = (eval_tag or "series").upper()
                    base = f"eval_{tag}_rf_lag{lag}"
                    _save_eval_plot(
                        y_index, y_true, rf_preds,
                        f"{tag} — RandomForest(lag={lag})  RMSE={rmse:.4f}",
                        os.path.join(artifacts_dir, f"{base}.png"),
                    )
                    _save_eval_json(
                        {
                            "model": "RandomForest",
                            "lag": lag,
                            "rmse": float(rmse),
                            "mape": _safe_mape(y_true, rf_preds),
                            "val_steps": val_steps,
                            "horizon": horizon,
                        },
                        os.path.join(artifacts_dir, f"{base}.json"),
                    )
        except Exception:
            logger.exception("RF candidate failed for lag=%d", lag)

    # SARIMAX
    try:
        sarimax_preds, sarimax_obj = _wf_sarimax(y, val_steps=val_steps, horizon=horizon)
        if _valid_preds(sarimax_preds, val_steps):
            rmse = mean_squared_error(y_true, sarimax_preds, squared=False)
            res = ModelResult(
                name="SARIMAX",
                yhat_val=sarimax_preds,
                rmse=float(rmse),
                model_obj=sarimax_obj,
                extra={"type": "sarimax"}
            )
            candidates.append(res)
            logger.debug("SARIMAX candidate rmse=%.4f", rmse)
            if save_plots:
                tag = (eval_tag or "series").upper()
                base = f"eval_{tag}_sarimax"
                _save_eval_plot(
                    y_index, y_true, sarimax_preds,
                    f"{tag} — SARIMAX  RMSE={rmse:.4f}",
                    os.path.join(artifacts_dir, f"{base}.png"),
                )
                _save_eval_json(
                    {
                        "model": "SARIMAX",
                        "rmse": float(rmse),
                        "mape": _safe_mape(y_true, sarimax_preds),
                        "val_steps": val_steps,
                        "horizon": horizon,
                    },
                    os.path.join(artifacts_dir, f"{base}.json"),
                )
        else:
            logger.warning("SARIMAX produced invalid preds (nan/len mismatch)")
    except Exception as e:
        logger.exception("SARIMAX failed: %s", e)

    # LSTM (если не отключён)
    if not DISABLE_LSTM:
        best_lstm: Optional[ModelResult] = None
        for win in (60, 90, 120):
            try:
                lstm_preds, lstm_obj = _wf_lstm(
                    y, val_steps=val_steps, window=win, epochs=8,
                    batch_size=16, horizon=horizon
                )
                if _valid_preds(lstm_preds, val_steps):
                    rmse = mean_squared_error(y_true, lstm_preds, squared=False)
                    cand = ModelResult(
                        name=f"LSTM(window={win})",
                        yhat_val=lstm_preds,
                        rmse=float(rmse),
                        model_obj=lstm_obj,
                        extra={"type": "lstm", "window": win}
                    )
                    logger.debug("LSTM candidate window=%d rmse=%.4f", win, rmse)
                    if save_plots:
                        tag = (eval_tag or "series").upper()
                        base = f"eval_{tag}_lstm_win{win}"
                        _save_eval_plot(
                            y_index, y_true, lstm_preds,
                            f"{tag} — LSTM(win={win})  RMSE={rmse:.4f}",
                            os.path.join(artifacts_dir, f"{base}.png"),
                        )
                        _save_eval_json(
                            {
                                "model": "LSTM",
                                "window": win,
                                "rmse": float(rmse),
                                "mape": _safe_mape(y_true, lstm_preds),
                                "val_steps": val_steps,
                                "horizon": horizon,
                            },
                            os.path.join(artifacts_dir, f"{base}.json"),
                        )
                    if (best_lstm is None) or (cand.rmse < best_lstm.rmse):
                        best_lstm = cand
            except Exception:
                logger.exception("LSTM candidate failed for window=%d", win)
        if best_lstm is not None:
            candidates.append(best_lstm)

    try:
        logger.info(
            "Model candidates: %s",
            ", ".join(f"{c.name}={c.rmse:.4f}" for c in candidates)
        )
    except Exception:
        pass

    if not candidates:
        raise RuntimeError("Не удалось обучить ни одну модель на вал-окне.")

    best = min(candidates, key=lambda m: m.rmse)

    if not _valid_preds(best.yhat_val, val_steps):
        raise RuntimeError(f"Лучшая модель '{best.name}' вернула некорректный валидационный прогноз.")

    logger.info("Winner model: %s rmse=%.4f", best.name, best.rmse)

    if save_plots:
        tag = (eval_tag or "series").upper()
        base = f"eval_{tag}_WINNER"
        _save_eval_plot(
            y_index, y_true, best.yhat_val,
            f"{tag} — WINNER: {best.name}  RMSE={best.rmse:.4f}",
            os.path.join(artifacts_dir, f"{base}.png"),
        )
        _save_eval_json(
            {
                "winner": best.name,
                "rmse": float(best.rmse),
                "mape": _safe_mape(y_true, best.yhat_val),
                "val_steps": val_steps,
                "horizon": horizon,
                "candidates": [
                    {"name": c.name, "rmse": float(c.rmse)}
                    for c in sorted(candidates, key=lambda x: x.rmse)
                ],
            },
            os.path.join(artifacts_dir, f"{base}.json"),
        )

    return best, candidates


def refit_and_forecast_30d(y: pd.Series, best: ModelResult) -> pd.Series:
    """Переобучает лучшую модель на всех данных и строит 30-дневный прогноз"""
    y = y.astype(float)

    logger.info("refit_and_forecast_30d using model '%s' type=%s", best.name, best.extra.get("type"))

    if best.extra.get("type") == "rf":
        rf_obj, max_lag = best.model_obj
        arr = y.values

        X, yy = [], []
        for t in range(max_lag, len(arr)):
            X.append(arr[t-max_lag:t])
            yy.append(arr[t])
        X, yy = np.array(X, dtype=float), np.array(yy, dtype=float)
        rf_obj.fit(X, yy)

        last_window = arr[-max_lag:].copy()
        preds = []
        for _ in range(30):
            yhat = float(rf_obj.predict(last_window.reshape(1, -1))[0])
            preds.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        return pd.Series(preds, index=range(1, 31))

    elif best.extra.get("type") == "sarimax":
        order, seas, trend, _ = best.model_obj
        model = sm.tsa.SARIMAX(
            y, order=order, seasonal_order=seas, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fcst = res.get_forecast(steps=30).predicted_mean.values
        return pd.Series(fcst, index=range(1, 31))

    elif best.extra.get("type") == "lstm":
        mo = best.model_obj
        if len(mo) == 4 and mo[-1] == 'returns':
            model, (mu, sigma), window, _ = mo
            p = y.values
            logp = np.log(p + 1e-9)
            r = np.diff(logp).astype("float32").reshape(-1, 1)

            mu_full = float(r.mean())
            sigma_full = float(r.std() + 1e-6)
            r_norm = (r - mu_full) / sigma_full

            X, yy = [], []
            for k in range(window, len(r_norm)):
                X.append(r_norm[k-window:k, 0])
                yy.append(r_norm[k, 0])
            if len(X) == 0:
                logger.warning("LSTM refit: too few samples; returning flat series")
                return pd.Series([float(p[-1])]*30, index=range(1, 31))

            X = np.array(X)[..., None]
            yy = np.array(yy)

            cbs = [
                keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
            ]
            model.fit(X, yy, epochs=8, batch_size=16, verbose=0, callbacks=cbs)

            last_seq = r_norm[-window:, 0].reshape(1, window, 1)
            r_future = []
            for _ in range(30):
                rhat_n = model.predict(last_seq, verbose=0).reshape(-1)[0]
                r_future.append(float(rhat_n * sigma_full + mu_full))
                last_seq = np.concatenate([last_seq[0, 1:, 0], [rhat_n]]).reshape(1, window, 1)

            last_logp = float(logp[-1])
            cum = np.cumsum(np.array(r_future, dtype=float))
            prices = np.exp(last_logp + cum)
            return pd.Series(prices, index=range(1, 31))

        else:
            model, (mu, sigma), window = mo
            arr = y.values.astype("float32").reshape(-1, 1)
            y_norm = (arr - mu) / sigma

            X, yy = [], []
            for t in range(window, len(y_norm)):
                X.append(y_norm[t-window:t, 0])
                yy.append(y_norm[t, 0])
            if len(X) == 0:
                logger.warning("LSTM refit (direct): too few samples; returning flat series")
                return pd.Series([float(y.iloc[-1])]*30, index=range(1, 31))

            X = np.array(X)[..., None]
            yy = np.array(yy)

            v = max(1, len(X)//10)
            X_tr, y_tr = X[:-v], yy[:-v]
            X_vl, y_vl = X[-v:], yy[-v:]

            cbs = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            ]
            model.fit(X_tr, y_tr, validation_data=(X_vl, y_vl), epochs=10, batch_size=16, verbose=0, callbacks=cbs)

            last_seq = y_norm[-window:, 0].reshape(1, window, 1)
            preds = []
            for _ in range(30):
                yhat = model.predict(last_seq, verbose=0).reshape(-1)[0]
                preds.append(float(yhat))
                last_seq = np.concatenate([last_seq[0, 1:, 0], [yhat]]).reshape(1, window, 1)

            preds_denorm = (np.array(preds) * sigma + mu).reshape(-1)
            return pd.Series(preds_denorm, index=range(1, 31))

    last = float(y.iloc[-1])
    logger.warning("refit_and_forecast_30d: unknown model type; returning flat series")
    return pd.Series([last]*30, index=range(1, 31))


def select_and_fit(
    y: pd.Series,
    val_steps: int = 30,
    horizon: int = WF_HORIZON,
    eval_tag: str = None,
    save_plots: bool = False,
    artifacts_dir: str = None
) -> ModelResult:
    """Старая сигнатура для обратной совместимости."""
    best, _ = select_and_fit_with_candidates(
        y, val_steps=val_steps, horizon=horizon,
        eval_tag=eval_tag, save_plots=save_plots, artifacts_dir=artifacts_dir
    )
    return best
