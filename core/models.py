# core/models.py
import json
import os
import time

# ---- детерминизм: ВАЖНО до импорта tensorflow ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger("models")

import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# если есть telegram в зависимости — можно оставить
try:
    from telegram.error import Forbidden  # noqa: F401
except Exception:
    Forbidden = Exception  # fallback

# --- optional industrial GBMs ---
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBRegressor = None
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    _HAS_LGBM = False

# --- optional CatBoost (очень сильный табличный бустер) ---
try:
    from catboost import CatBoostRegressor  # type: ignore
    _HAS_CAT = True
except Exception:
    CatBoostRegressor = None
    _HAS_CAT = False

# ---- warnings ----
warnings.filterwarnings("ignore")

# ---- GPU memory growth (если GPU есть) ----
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---- глобальные сиды ----
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

matplotlib.use("Agg")

# =========================
#        CONSTA NTS
# =========================

WF_HORIZON = int(os.getenv("WF_HORIZON", "5"))

# ускорение WF: переобучать не на каждом шаге, а раз в K шагов
WF_REFIT_EVERY = int(os.getenv("WF_REFIT_EVERY", "5"))

# ограничение длины вал-окна при подборе (ускорение)
VAL_STEPS_CAP = int(os.getenv("VAL_STEPS_CAP", "15"))

DISABLE_LSTM = os.getenv("DISABLE_LSTM", "0") == "1"
DISABLE_NBEATS = os.getenv("DISABLE_NBEATS", "0") == "1"
DISABLE_TCN = os.getenv("DISABLE_TCN", "0") == "1"
DISABLE_GBM = os.getenv("DISABLE_GBM", "0") == "1"
DISABLE_CATBOOST = os.getenv("DISABLE_CATBOOST", "0") == "1"
DISABLE_TRANSFORMER = os.getenv("DISABLE_TRANSFORMER", "1") == "1"  # heavier, off by default

# подробные логи по обучению
MODELS_VERBOSE = os.getenv("MODELS_VERBOSE", "0") == "1"
MODELS_LOG_EVERY = int(os.getenv("MODELS_LOG_EVERY", "10"))  # как часто писать прогресс wf

# раннее отсечение плохих кандидатов
EARLY_CUTOFF_MULT = float(os.getenv("EARLY_CUTOFF_MULT", "1.25"))


def _log(msg: str, *args, level="info"):
    """Условные verbose-логи (MODELS_VERBOSE=1)."""
    if MODELS_VERBOSE:
        getattr(logger, level)(msg, *args)


# =========================
#        ENV GRIDS
# =========================

def _int_list_from_env(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    items: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except ValueError:
            continue
    return items


def _float_list_from_env(name: str, default: str) -> list[float]:
    raw = os.getenv(name, default)
    items: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(float(part))
        except ValueError:
            continue
    return items


# === NBEATS конфиг ===
NBEATS_BLOCKS = int(os.getenv("NBEATS_BLOCKS", "3"))
NBEATS_WIDTH = int(os.getenv("NBEATS_WIDTH", "128"))
NBEATS_HIDDEN = int(os.getenv("NBEATS_HIDDEN", "4"))
NBEATS_EPOCHS = int(os.getenv("NBEATS_EPOCHS", "8"))
NBEATS_BATCH = int(os.getenv("NBEATS_BATCH", "32"))

# быстрые гриды для NBEATS (по умолчанию)
NBEATS_WINDOW_GRID = _int_list_from_env("NBEATS_WINDOW_GRID", "60,90")
NBEATS_BLOCKS_GRID = _int_list_from_env("NBEATS_BLOCKS_GRID", "2")
NBEATS_WIDTH_GRID = _int_list_from_env("NBEATS_WIDTH_GRID", "64,128")
NBEATS_HIDDEN_GRID = _int_list_from_env("NBEATS_HIDDEN_GRID", "2")


# === GBM (XGB/LGBM/HGB) быстрый грид ===
GBM_LAG_GRID = _int_list_from_env("GBM_LAG_GRID", "30,60")
GBM_N_ESTIMATORS_GRID = _int_list_from_env("GBM_N_ESTIMATORS_GRID", "400,800")
GBM_MAX_DEPTH_GRID = _int_list_from_env("GBM_MAX_DEPTH_GRID", "3,5")
GBM_LR_GRID = _float_list_from_env("GBM_LR_GRID", "0.03")
GBM_SUBSAMPLE_GRID = _float_list_from_env("GBM_SUBSAMPLE_GRID", "0.9")
GBM_COLSAMPLE_GRID = _float_list_from_env("GBM_COLSAMPLE_GRID", "0.9")


# === CatBoost грид ===
CAT_LAG_GRID = _int_list_from_env("CAT_LAG_GRID", "30,60")
CAT_ITERS_GRID = _int_list_from_env("CAT_ITERS_GRID", "600,1200")
CAT_DEPTH_GRID = _int_list_from_env("CAT_DEPTH_GRID", "4,6")
CAT_LR_GRID = _float_list_from_env("CAT_LR_GRID", "0.03")


# === TCN быстрый грид ===
TCN_WINDOW_GRID = _int_list_from_env("TCN_WINDOW_GRID", "60,90")
TCN_FILTERS_GRID = _int_list_from_env("TCN_FILTERS_GRID", "32,64")
TCN_STACKS_GRID = _int_list_from_env("TCN_STACKS_GRID", "2")
TCN_DROPOUT_GRID = _float_list_from_env("TCN_DROPOUT_GRID", "0.1")


# === Transformer-lite (опционально) ===
TR_WINDOW_GRID = _int_list_from_env("TR_WINDOW_GRID", "60,90")
TR_DMODEL_GRID = _int_list_from_env("TR_DMODEL_GRID", "32,64")
TR_HEADS_GRID = _int_list_from_env("TR_HEADS_GRID", "2,4")
TR_LAYERS_GRID = _int_list_from_env("TR_LAYERS_GRID", "1,2")
TR_DROPOUT_GRID = _float_list_from_env("TR_DROPOUT_GRID", "0.1")


ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


# =========================
#         UTILS
# =========================

def _pad_to_val_steps(preds, val_steps, fill_value):
    """Дополняет прогнозы до val_steps слева fill_value."""
    preds = np.asarray(preds, dtype=float).reshape(-1)
    if len(preds) < val_steps:
        pad = np.full(val_steps - len(preds), float(fill_value), dtype=float)
        preds = np.concatenate([pad, preds])
    return preds


def _pad_to_orig_steps(preds: np.ndarray, orig_steps: int, fill_value: float) -> np.ndarray:
    """Пэдит preds до orig_steps (слева), если preds короче."""
    preds = np.asarray(preds, dtype=float).reshape(-1)
    if len(preds) < orig_steps:
        pad = np.full(orig_steps - len(preds), float(fill_value), dtype=float)
        preds = np.concatenate([pad, preds])
    return preds


def _safe_mape(y_true, y_pred):
    try:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        return None


def _save_eval_plot(y_idx, y_true, y_pred, title, out_png):
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
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.exception("Failed to save eval json: %s", out_json)


def _valid_preds(arr: np.ndarray, val_steps: int) -> bool:
    try:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return (arr.ndim == 1) and (len(arr) == val_steps) and np.isfinite(arr).all()
    except Exception:
        return False


def _make_lagged_xy(arr: np.ndarray, lag: int, horizon: int = 1):
    """Лаговые признаки для табличных моделей."""
    X, yy = [], []
    max_t = len(arr) - horizon + 1
    for t in range(lag, max_t):
        X.append(arr[t - lag:t])
        yy.append(arr[t:t + horizon])
    X = np.asarray(X, dtype=float)
    yy = np.asarray(yy, dtype=float)
    if horizon == 1:
        yy = yy.reshape(-1)
    return X, yy


# =========================
#         DATACLASS
# =========================

@dataclass
class ModelResult:
    name: str
    yhat_val: np.ndarray   # всегда длиной orig_val_steps
    rmse: float            # RMSE посчитан на val_steps_eff
    model_obj: Any
    extra: Dict[str, Any]


# =========================
#   INDUSTRIAL GBM (XGB/LGBM/HGB)
# =========================

def _wf_gbm(
    y: pd.Series,
    max_lag: int,
    val_steps: int,
    horizon: int = 1,
    gbm_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Walk-forward для GBM с лаговыми признаками.
    Приоритет: XGBoost -> LightGBM -> sklearn HistGB.

    Ускорение:
      - refit раз в WF_REFIT_EVERY шагов
    """
    y_arr = y.astype(float).values
    start = len(y_arr) - val_steps
    n_valid = min(val_steps, max(0, len(y_arr) - (start + horizon - 1)))
    preds: List[float] = []

    gbm_params = gbm_params or {}
    model = None
    used = None
    used_params: Dict[str, Any] = {}

    if _HAS_XGB:
        base = dict(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        base.update(gbm_params)
        used_params = base
        model = XGBRegressor(**used_params)
        used = "xgb"

    elif _HAS_LGBM:
        base = dict(
            n_estimators=900,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED,
        )
        base.update(gbm_params)
        used_params = base
        model = LGBMRegressor(**used_params)
        used = "lgbm"

    else:
        base = dict(
            max_iter=500,
            learning_rate=0.05,
            max_depth=None,
            l2_regularization=1.0,
            random_state=SEED,
        )
        translated = {}
        if "n_estimators" in gbm_params:
            translated["max_iter"] = int(gbm_params["n_estimators"])
        if "learning_rate" in gbm_params:
            translated["learning_rate"] = float(gbm_params["learning_rate"])
        if "max_depth" in gbm_params:
            translated["max_depth"] = gbm_params["max_depth"]

        base.update(translated)
        used_params = base
        model = HistGradientBoostingRegressor(**used_params)
        used = "hgb"

    _log(
        "WF_GBM start | backend=%s | lag=%d | val_steps=%d | horizon=%d | refit_every=%d | params=%s",
        used, max_lag, val_steps, horizon, WF_REFIT_EVERY, used_params
    )
    t0 = time.time()
    last_fit_i = None

    for i in range(n_valid):
        t = start + i
        train_arr = y_arr[:t]
        if len(train_arr) <= max_lag:
            preds.append(float(train_arr[-1]))
            continue

        if (last_fit_i is None) or (i % WF_REFIT_EVERY == 0):
            X_tr, y_tr = _make_lagged_xy(train_arr, max_lag, horizon=1)
            model.fit(X_tr, y_tr)
            last_fit_i = i
            _log("WF_GBM refit | i=%d/%d | train_len=%d | X=%s", i+1, n_valid, len(train_arr), X_tr.shape)

        last_window = train_arr[-max_lag:].copy()
        yhat = None
        for _ in range(horizon):
            yhat = float(model.predict(last_window.reshape(1, -1))[0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        preds.append(float(yhat))

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_GBM progress | step=%d/%d | last_pred=%.4f", i + 1, n_valid, yhat)

    preds = np.array(preds, dtype=float)
    fill = float(y_arr[start - 1]) if start - 1 >= 0 else float(y_arr[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)

    _log("WF_GBM done | elapsed=%.2fs", time.time() - t0)
    return preds, (model, max_lag, used, used_params)


# =========================
#            CATBOOST
# =========================

def _wf_catboost(
    y: pd.Series,
    max_lag: int,
    val_steps: int,
    horizon: int = 1,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any]:
    """Walk-forward для CatBoostRegressor на лагах."""
    if not _HAS_CAT:
        raise RuntimeError("CatBoost not installed")

    y_arr = y.astype(float).values
    start = len(y_arr) - val_steps
    n_valid = min(val_steps, max(0, len(y_arr) - (start + horizon - 1)))
    preds: List[float] = []

    base = dict(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
    )
    if params:
        base.update(params)

    model = CatBoostRegressor(**base)

    _log(
        "WF_CAT start | lag=%d | val_steps=%d | horizon=%d | refit_every=%d | params=%s",
        max_lag, val_steps, horizon, WF_REFIT_EVERY, base
    )
    t0 = time.time()
    last_fit_i = None

    for i in range(n_valid):
        t = start + i
        train_arr = y_arr[:t]
        if len(train_arr) <= max_lag:
            preds.append(float(train_arr[-1]))
            continue

        if (last_fit_i is None) or (i % WF_REFIT_EVERY == 0):
            X_tr, y_tr = _make_lagged_xy(train_arr, max_lag, horizon=1)
            model.fit(X_tr, y_tr)
            last_fit_i = i
            _log("WF_CAT refit | i=%d/%d | train_len=%d | X=%s", i+1, n_valid, len(train_arr), X_tr.shape)

        last_window = train_arr[-max_lag:].copy()
        yhat = None
        for _ in range(horizon):
            yhat = float(model.predict(last_window.reshape(1, -1))[0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        preds.append(float(yhat))

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_CAT progress | step=%d/%d | last_pred=%.4f", i + 1, n_valid, yhat)

    preds = np.array(preds, dtype=float)
    fill = float(y_arr[start - 1]) if start - 1 >= 0 else float(y_arr[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)

    _log("WF_CAT done | elapsed=%.2fs", time.time() - t0)
    return preds, (model, max_lag, base)


# =========================
#            LSTM/GRU (returns)
# =========================

def _wf_lstm(
    y: pd.Series,
    val_steps: int,
    window: int = 30,
    epochs: int = 6,
    batch_size: int = 16,
    horizon: int = 1
):
    """Walk-forward для GRU на лог-доходностях. refit раз в WF_REFIT_EVERY."""
    p = y.astype(float).values
    if len(p) < window + val_steps + 5:
        raise ValueError("Слишком короткий ряд для LSTM/GRU по доходностям.")

    _log(
        "WF_LSTM start | window=%d | val_steps=%d | horizon=%d | refit_every=%d",
        window, val_steps, horizon, WF_REFIT_EVERY
    )
    t0 = time.time()

    logp = np.log(p + 1e-9)
    r = np.diff(logp).astype("float32").reshape(-1, 1)

    start = len(r) - val_steps
    n_valid = min(val_steps, max(0, len(r) - (start + horizon - 1)))
    preds_prices: List[float] = []

    def build_model():
        m = keras.Sequential([
            keras.layers.Input(shape=(window, 1)),
            keras.layers.GRU(48, return_sequences=False),
            keras.layers.Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    model: Optional[keras.Model] = None
    mu_last, sigma_last = 0.0, 1.0

    for i in range(n_valid):
        t = start + i
        train = r[:t]
        mu = float(train.mean())
        sigma = float(train.std() + 1e-6)
        mu_last, sigma_last = mu, sigma
        norm = (train - mu) / sigma

        if len(norm) <= window:
            preds_prices.append(float(p[t]))
            model = None
            continue

        X, yy = [], []
        for k in range(window, len(norm)):
            X.append(norm[k - window:k, 0])
            yy.append(norm[k, 0])
        X = np.array(X)[..., None]
        yy = np.array(yy)

        if model is None:
            model = build_model()

        if i % WF_REFIT_EVERY == 0:
            base_epochs = max(8, epochs) if i == 0 else 2
            model.fit(
                X, yy,
                epochs=base_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
            )
            _log(
                "WF_LSTM refit | i=%d/%d | train_len=%d | X=%s | epochs=%d",
                i + 1, n_valid, len(train), X.shape, base_epochs
            )

        last_seq = norm[-window:, 0].reshape(1, window, 1)
        rhat_norm = None
        for _ in range(horizon):
            rhat_norm = model.predict(last_seq, verbose=0).reshape(-1)[0]
            last_seq = np.concatenate([last_seq[0, 1:, 0], [rhat_norm]]).reshape(1, window, 1)

        rhat = float(rhat_norm * sigma + mu)
        last_logp = float(logp[t])
        yhat_price = float(np.exp(last_logp + rhat))
        preds_prices.append(yhat_price)

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_LSTM progress | step=%d/%d | last_pred=%.4f", i + 1, n_valid, yhat_price)

    preds_prices = np.array(preds_prices, dtype=float)
    fill_price = float(p[start]) if start >= 0 else float(p[0])
    preds_prices = _pad_to_val_steps(preds_prices, val_steps, fill_price)

    _log("WF_LSTM done | elapsed=%.2fs", time.time() - t0)
    return preds_prices, (model, (mu_last, sigma_last), window, "returns")


# =========================
#            TCN
# =========================

def _wf_tcn(
    y: pd.Series,
    val_steps: int,
    window: int = 60,
    epochs: int = 6,
    batch_size: int = 16,
    horizon: int = 1,
    n_filters: int = 64,
    n_stacks: int = 2,
    dropout: float = 0.1,
) -> Tuple[np.ndarray, Any]:
    """Walk-forward для TCN. refit раз в WF_REFIT_EVERY."""
    p = y.astype(float).values
    if len(p) < window + val_steps + 5:
        raise ValueError("Слишком короткий ряд для TCN.")

    _log(
        "WF_TCN start | window=%d | filters=%d | stacks=%d | dropout=%.3f | val_steps=%d | horizon=%d | refit_every=%d",
        window, n_filters, n_stacks, dropout, val_steps, horizon, WF_REFIT_EVERY
    )
    t0 = time.time()

    start = len(p) - val_steps
    n_valid = min(val_steps, max(0, len(p) - (start + horizon - 1)))
    preds: List[float] = []

    def build_model(forecast_length: int):
        inp = keras.Input(shape=(window, 1))
        x = inp
        dilation = 1
        for _ in range(n_stacks):
            x = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
            )(x)
            x = keras.layers.Dropout(dropout)(x)
            dilation *= 2
        x = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(forecast_length)(x)
        m = keras.Model(inp, out, name="TCN")
        m.compile(optimizer="adam", loss="mse")
        return m

    model: Optional[keras.Model] = None

    for i in range(n_valid):
        t = start + i
        train = p[:t]

        if len(train) <= window + horizon:
            preds.append(float(train[-1]))
            model = None
            continue

        X, yy = [], []
        max_idx = len(train) - horizon + 1
        for s in range(window, max_idx):
            X.append(train[s - window:s])
            yy.append(train[s:s + horizon])

        X = np.array(X, dtype="float32")[..., None]
        yy = np.array(yy, dtype="float32")

        if X.shape[0] < 16:
            preds.append(float(train[-1]))
            model = None
            continue

        mu = float(train.mean())
        sigma = float(train.std() + 1e-6)
        X_norm = (X - mu) / sigma
        yy_norm = (yy - mu) / sigma

        if model is None:
            model = build_model(horizon)

        if i % WF_REFIT_EVERY == 0:
            base_epochs = max(8, epochs) if i == 0 else 2
            model.fit(
                X_norm, yy_norm,
                epochs=base_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
            )
            _log(
                "WF_TCN refit | i=%d/%d | train_len=%d | X=%s | epochs=%d",
                i + 1, n_valid, len(train), X_norm.shape, base_epochs
            )

        last_window = train[-window:]
        last_norm = ((last_window - mu) / sigma).reshape(1, window, 1)
        fc_norm = model.predict(last_norm, verbose=0)[0]
        yhat = float(fc_norm[-1] * sigma + mu)

        if not np.isfinite(yhat):
            yhat = float(train[-1])

        preds.append(yhat)

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_TCN progress | step=%d/%d | last_pred=%.4f", i + 1, n_valid, yhat)

    preds = np.array(preds, dtype=float)
    fill = float(p[start - 1]) if start - 1 >= 0 else float(p[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)

    _log("WF_TCN done | elapsed=%.2fs", time.time() - t0)
    tcn_obj = (model, window, horizon, n_filters, n_stacks, dropout)
    return preds, tcn_obj


# =========================
#   TRANSFORMER-LITE (optional)
# =========================

def _wf_transformer(
    y: pd.Series,
    val_steps: int,
    window: int = 60,
    epochs: int = 6,
    batch_size: int = 16,
    horizon: int = 1,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> Tuple[np.ndarray, Any]:
    """
    Лёгкий Transformer-энкодер по окну цен.
    Включать DISABLE_TRANSFORMER=0.
    """
    p = y.astype(float).values
    if len(p) < window + val_steps + 5:
        raise ValueError("Слишком короткий ряд для Transformer.")

    _log(
        "WF_TR start | window=%d d_model=%d heads=%d layers=%d drop=%.2f val_steps=%d horizon=%d refit_every=%d",
        window, d_model, n_heads, n_layers, dropout, val_steps, horizon, WF_REFIT_EVERY
    )
    t0 = time.time()

    start = len(p) - val_steps
    n_valid = min(val_steps, max(0, len(p) - (start + horizon - 1)))
    preds: List[float] = []

    def build_model():
        inp = keras.Input(shape=(window, 1))
        x = keras.layers.Dense(d_model)(inp)

        # positional encoding (learned)
        pos = tf.range(start=0, limit=window, delta=1)
        pos_emb = keras.layers.Embedding(input_dim=window, output_dim=d_model)(pos)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        x = x + pos_emb

        for _ in range(n_layers):
            attn = keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)(x, x)
            x = keras.layers.Add()([x, attn])
            x = keras.layers.LayerNormalization()(x)

            ffn = keras.Sequential([
                keras.layers.Dense(d_model * 2, activation="relu"),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(d_model),
            ])(x)

            x = keras.layers.Add()([x, ffn])
            x = keras.layers.LayerNormalization()(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(horizon)(x)
        m = keras.Model(inp, out, name="TransformerLite")
        m.compile(optimizer="adam", loss="mse")
        return m

    model: Optional[keras.Model] = None

    for i in range(n_valid):
        t = start + i
        train = p[:t]

        if len(train) <= window + horizon:
            preds.append(float(train[-1]))
            model = None
            continue

        X, yy = [], []
        max_idx = len(train) - horizon + 1
        for s in range(window, max_idx):
            X.append(train[s - window:s])
            yy.append(train[s:s + horizon])

        X = np.array(X, dtype="float32")[..., None]
        yy = np.array(yy, dtype="float32")

        if X.shape[0] < 16:
            preds.append(float(train[-1]))
            model = None
            continue

        mu = float(train.mean())
        sigma = float(train.std() + 1e-6)
        X_norm = (X - mu) / sigma
        yy_norm = (yy - mu) / sigma

        if model is None:
            model = build_model()

        if i % WF_REFIT_EVERY == 0:
            base_epochs = max(8, epochs) if i == 0 else 2
            model.fit(
                X_norm, yy_norm,
                epochs=base_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
            )
            _log("WF_TR refit | i=%d/%d | train_len=%d | X=%s epochs=%d", i+1, n_valid, len(train), X_norm.shape, base_epochs)

        last_window = train[-window:]
        last_norm = ((last_window - mu) / sigma).reshape(1, window, 1)
        fc_norm = model.predict(last_norm, verbose=0)[0]
        yhat = float(fc_norm[-1] * sigma + mu)
        if not np.isfinite(yhat):
            yhat = float(train[-1])
        preds.append(yhat)

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_TR progress | step=%d/%d | last_pred=%.4f", i+1, n_valid, yhat)

    preds = np.array(preds, dtype=float)
    fill = float(p[start - 1]) if start - 1 >= 0 else float(p[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)

    _log("WF_TR done | elapsed=%.2fs", time.time() - t0)
    tr_obj = (model, window, horizon, d_model, n_heads, n_layers, dropout)
    return preds, tr_obj


# =========================
#           SELECT + FIT
# =========================

def select_and_fit_with_candidates(
    y: pd.Series,
    val_steps: int = 30,
    horizon: int = WF_HORIZON,
    eval_tag: str = None,
    save_plots: bool = False,
    artifacts_dir: str = None
) -> Tuple[ModelResult, List[ModelResult]]:
    """
    Выбирает лучшую модель по RMSE на валидационном периоде.

    Ускорение:
      - внутри подбора используем укороченное val-окно val_steps_eff
      - но наружу всегда возвращаем yhat_val длиной orig_val_steps
    """
    if artifacts_dir is None:
        artifacts_dir = ART_DIR
    os.makedirs(artifacts_dir, exist_ok=True)

    y = y.astype(float)

    orig_val_steps = int(val_steps)
    val_steps_eff = min(orig_val_steps, VAL_STEPS_CAP)

    if len(y) <= max(35, val_steps_eff + 5):
        raise ValueError("Слишком короткий ряд для надёжной walk-forward валидации.")

    # full series для внешних пользователей и forecast.py
    y_true_full = y.iloc[-orig_val_steps:].values
    y_index_full = y.index[-orig_val_steps:]

    # короткое окно для RMSE в подборе
    y_true_eff = y.iloc[-val_steps_eff:].values

    fill_value = float(y.iloc[-orig_val_steps - 1]) if len(y) > orig_val_steps else float(y.iloc[0])

    candidates: List[ModelResult] = []
    current_best_rmse = np.inf

    _log(
        "SELECT start | len=%d | orig_val_steps=%d | val_steps_eff=%d | horizon=%d | cutoff_mult=%.2f",
        len(y), orig_val_steps, val_steps_eff, horizon, EARLY_CUTOFF_MULT
    )

    # --- GBM grid ---
    if not DISABLE_GBM:
        for lag in GBM_LAG_GRID:
            for n_est in GBM_N_ESTIMATORS_GRID:
                for depth in GBM_MAX_DEPTH_GRID:
                    for lr in GBM_LR_GRID:
                        for subs in GBM_SUBSAMPLE_GRID:
                            for cols in GBM_COLSAMPLE_GRID:
                                params = dict(
                                    n_estimators=n_est,
                                    max_depth=depth,
                                    learning_rate=lr,
                                    subsample=subs,
                                    colsample_bytree=cols,
                                )
                                try:
                                    preds_eff, obj = _wf_gbm(
                                        y, max_lag=lag, val_steps=val_steps_eff,
                                        horizon=horizon, gbm_params=params
                                    )
                                    if not _valid_preds(preds_eff, val_steps_eff):
                                        continue

                                    rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                                    if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                                        _log("CAND drop | GBM lag=%d params=%s rmse=%.6f", lag, params, rmse)
                                        continue

                                    preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)
                                    backend = obj[2]

                                    res = ModelResult(
                                        name=f"GBM[{backend}](lag={lag},n={n_est},d={depth},lr={lr})",
                                        yhat_val=preds_full,
                                        rmse=float(rmse),
                                        model_obj=obj,
                                        extra={
                                            "type": "gbm",
                                            "lag": lag,
                                            "backend": backend,
                                            "params": params,
                                            "val_steps_used": val_steps_eff,
                                        }
                                    )
                                    candidates.append(res)
                                    current_best_rmse = min(current_best_rmse, rmse)

                                    logger.debug("GBM candidate %s rmse=%.4f", res.name, rmse)
                                    _log("CANDIDATE | %s | rmse=%.6f", res.name, rmse)

                                    if save_plots:
                                        tag = (eval_tag or "series").upper()
                                        base = f"eval_{tag}_gbm_{backend}_lag{lag}_n{n_est}_d{depth}_lr{lr}"
                                        _save_eval_plot(
                                            y_index_full, y_true_full, preds_full,
                                            f"{tag} — {res.name} RMSE={rmse:.4f}",
                                            os.path.join(artifacts_dir, f"{base}.png"),
                                        )
                                        _save_eval_json(
                                            {
                                                "model": "GBM",
                                                "backend": backend,
                                                "lag": lag,
                                                "params": params,
                                                "rmse": float(rmse),
                                                "mape": _safe_mape(y_true_eff, preds_eff),
                                                "orig_val_steps": orig_val_steps,
                                                "val_steps_used": val_steps_eff,
                                                "horizon": horizon,
                                            },
                                            os.path.join(artifacts_dir, f"{base}.json"),
                                        )
                                except Exception:
                                    logger.exception("GBM candidate failed (lag=%s params=%s)", lag, params)

    # --- CatBoost grid ---
    if (not DISABLE_CATBOOST) and _HAS_CAT:
        for lag in CAT_LAG_GRID:
            for iters in CAT_ITERS_GRID:
                for depth in CAT_DEPTH_GRID:
                    for lr in CAT_LR_GRID:
                        params = dict(iterations=iters, depth=depth, learning_rate=lr)
                        try:
                            preds_eff, obj = _wf_catboost(
                                y, max_lag=lag, val_steps=val_steps_eff,
                                horizon=horizon, params=params
                            )
                            if not _valid_preds(preds_eff, val_steps_eff):
                                continue

                            rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                            if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                                _log("CAND drop | CAT lag=%d params=%s rmse=%.6f", lag, params, rmse)
                                continue

                            preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)

                            res = ModelResult(
                                name=f"CATBOOST(lag={lag},it={iters},d={depth},lr={lr})",
                                yhat_val=preds_full,
                                rmse=float(rmse),
                                model_obj=obj,
                                extra={
                                    "type": "catboost",
                                    "lag": lag,
                                    "params": params,
                                    "val_steps_used": val_steps_eff,
                                }
                            )
                            candidates.append(res)
                            current_best_rmse = min(current_best_rmse, rmse)

                            logger.debug("CAT candidate %s rmse=%.4f", res.name, rmse)
                            _log("CANDIDATE | %s | rmse=%.6f", res.name, rmse)

                        except Exception:
                            logger.exception("CAT candidate failed (lag=%s params=%s)", lag, params)

    # --- TCN grid ---
    if not DISABLE_TCN:
        best_tcn: Optional[ModelResult] = None
        for win in TCN_WINDOW_GRID:
            for nf in TCN_FILTERS_GRID:
                for ns in TCN_STACKS_GRID:
                    for dr in TCN_DROPOUT_GRID:
                        try:
                            preds_eff, obj = _wf_tcn(
                                y, val_steps=val_steps_eff, window=win,
                                epochs=6, batch_size=16, horizon=horizon,
                                n_filters=nf, n_stacks=ns, dropout=dr
                            )
                            if not _valid_preds(preds_eff, val_steps_eff):
                                continue

                            rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                            if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                                _log("CAND drop | TCN win=%d F=%d S=%d D=%.2f rmse=%.6f", win, nf, ns, dr, rmse)
                                continue

                            preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)

                            cand = ModelResult(
                                name=f"TCN(win={win},F={nf},S={ns},D={dr})",
                                yhat_val=preds_full,
                                rmse=float(rmse),
                                model_obj=obj,
                                extra={
                                    "type": "tcn",
                                    "window": win, "filters": nf, "stacks": ns, "dropout": dr,
                                    "val_steps_used": val_steps_eff
                                }
                            )

                            if (best_tcn is None) or (cand.rmse < best_tcn.rmse):
                                best_tcn = cand

                            current_best_rmse = min(current_best_rmse, rmse)

                            logger.debug("TCN candidate %s rmse=%.4f", cand.name, rmse)
                            _log("CANDIDATE | %s | rmse=%.6f", cand.name, rmse)

                        except Exception:
                            logger.exception("TCN candidate failed for win=%d F=%d S=%d D=%s", win, nf, ns, dr)

        if best_tcn is not None:
            candidates.append(best_tcn)

    # --- LSTM/GRU ---
    if not DISABLE_LSTM:
        best_lstm: Optional[ModelResult] = None
        for win in (60, 90):
            try:
                preds_eff, obj = _wf_lstm(
                    y, val_steps=val_steps_eff, window=win, epochs=6,
                    batch_size=16, horizon=horizon
                )
                if not _valid_preds(preds_eff, val_steps_eff):
                    continue

                rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                    _log("CAND drop | LSTM win=%d rmse=%.6f", win, rmse)
                    continue

                preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)

                cand = ModelResult(
                    name=f"LSTM(window={win})",
                    yhat_val=preds_full,
                    rmse=float(rmse),
                    model_obj=obj,
                    extra={"type": "lstm", "window": win, "val_steps_used": val_steps_eff}
                )

                if (best_lstm is None) or (cand.rmse < best_lstm.rmse):
                    best_lstm = cand

                current_best_rmse = min(current_best_rmse, rmse)

                logger.debug("LSTM candidate %s rmse=%.4f", cand.name, rmse)
                _log("CANDIDATE | %s | rmse=%.6f", cand.name, rmse)

            except Exception:
                logger.exception("LSTM candidate failed for window=%d", win)

        if best_lstm is not None:
            candidates.append(best_lstm)

    # --- N-BEATS grid ---
    if not DISABLE_NBEATS:
        best_nbeats: Optional[ModelResult] = None
        for win in NBEATS_WINDOW_GRID:
            for n_blocks in NBEATS_BLOCKS_GRID:
                for width in NBEATS_WIDTH_GRID:
                    for n_hidden in NBEATS_HIDDEN_GRID:
                        try:
                            preds_eff, obj = _wf_nbeats(
                                y, val_steps=val_steps_eff, window=win, horizon=horizon,
                                n_blocks=n_blocks, width=width, n_hidden=n_hidden,
                            )
                            if not _valid_preds(preds_eff, val_steps_eff):
                                continue

                            rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                            if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                                _log("CAND drop | NBEATS win=%d B=%d W=%d H=%d rmse=%.6f",
                                     win, n_blocks, width, n_hidden, rmse)
                                continue

                            preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)

                            cand = ModelResult(
                                name=f"NBEATS(win={win},B={n_blocks},W={width},H={n_hidden})",
                                yhat_val=preds_full,
                                rmse=float(rmse),
                                model_obj=obj,
                                extra={
                                    "type": "nbeats",
                                    "window": win, "blocks": n_blocks, "width": width, "hidden": n_hidden,
                                    "val_steps_used": val_steps_eff
                                }
                            )

                            if (best_nbeats is None) or (cand.rmse < best_nbeats.rmse):
                                best_nbeats = cand

                            current_best_rmse = min(current_best_rmse, rmse)

                            logger.debug("NBEATS candidate %s rmse=%.4f", cand.name, rmse)
                            _log("CANDIDATE | %s | rmse=%.6f", cand.name, rmse)

                        except Exception:
                            logger.exception("NBEATS candidate failed for win=%d B=%d W=%d H=%d", win, n_blocks, width, n_hidden)

        if best_nbeats is not None:
            candidates.append(best_nbeats)

    # --- Transformer-lite grid (optional) ---
    if not DISABLE_TRANSFORMER:
        best_tr: Optional[ModelResult] = None
        for win in TR_WINDOW_GRID:
            for dm in TR_DMODEL_GRID:
                for hds in TR_HEADS_GRID:
                    for lays in TR_LAYERS_GRID:
                        for dr in TR_DROPOUT_GRID:
                            try:
                                preds_eff, obj = _wf_transformer(
                                    y, val_steps=val_steps_eff, window=win, epochs=6, batch_size=16,
                                    horizon=horizon, d_model=dm, n_heads=hds, n_layers=lays, dropout=dr
                                )
                                if not _valid_preds(preds_eff, val_steps_eff):
                                    continue

                                rmse = mean_squared_error(y_true_eff, preds_eff, squared=False)
                                if np.isfinite(current_best_rmse) and rmse > current_best_rmse * EARLY_CUTOFF_MULT:
                                    _log("CAND drop | TR win=%d dm=%d h=%d l=%d dr=%.2f rmse=%.6f",
                                         win, dm, hds, lays, dr, rmse)
                                    continue

                                preds_full = _pad_to_orig_steps(preds_eff, orig_val_steps, fill_value)

                                cand = ModelResult(
                                    name=f"TR(win={win},dm={dm},h={hds},L={lays},D={dr})",
                                    yhat_val=preds_full,
                                    rmse=float(rmse),
                                    model_obj=obj,
                                    extra={
                                        "type": "transformer",
                                        "window": win, "d_model": dm, "heads": hds, "layers": lays, "dropout": dr,
                                        "val_steps_used": val_steps_eff
                                    }
                                )
                                if (best_tr is None) or (cand.rmse < best_tr.rmse):
                                    best_tr = cand
                                current_best_rmse = min(current_best_rmse, rmse)
                            except Exception:
                                logger.exception("TR candidate failed")

        if best_tr is not None:
            candidates.append(best_tr)

    if not candidates:
        raise RuntimeError("Не удалось обучить ни одну модель на вал-окне.")

    if MODELS_VERBOSE:
        top = sorted(candidates, key=lambda c: c.rmse)[:5]
        _log("TOP5 candidates:\n%s", "\n".join(f"  {c.name}: {c.rmse:.6f}" for c in top))

    best = min(candidates, key=lambda m: m.rmse)
    if not _valid_preds(best.yhat_val, orig_val_steps):
        raise RuntimeError(
            f"Лучшая модель '{best.name}' вернула некорректный валидационный прогноз."
        )

    logger.info("Winner model: %s rmse=%.4f (val_steps_used=%d)", best.name, best.rmse, val_steps_eff)

    if save_plots:
        tag = (eval_tag or "series").upper()
        base = f"eval_{tag}_WINNER"
        _save_eval_plot(
            y_index_full, y_true_full, best.yhat_val,
            f"{tag} — WINNER: {best.name} RMSE={best.rmse:.4f}",
            os.path.join(artifacts_dir, f"{base}.png"),
        )
        _save_eval_json(
            {
                "winner": best.name,
                "rmse": float(best.rmse),
                "mape": _safe_mape(y_true_eff, best.yhat_val[-val_steps_eff:]),
                "orig_val_steps": orig_val_steps,
                "val_steps_used": val_steps_eff,
                "horizon": horizon,
                "candidates": [
                    {"name": c.name, "rmse": float(c.rmse), "val_steps_used": c.extra.get("val_steps_used")}
                    for c in sorted(candidates, key=lambda x: x.rmse)
                ],
            },
            os.path.join(artifacts_dir, f"{base}.json"),
        )

    return best, candidates


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


# =========================
#      REFIT + 30D FORECAST
# =========================

def refit_and_forecast_30d(y: pd.Series, best: ModelResult) -> pd.Series:
    """Переобучает лучшую модель на всех данных и строит 30-дневный прогноз."""
    y = y.astype(float)

    logger.info(
        "refit_and_forecast_30d using model '%s' type=%s",
        best.name, best.extra.get("type")
    )

    if best.extra.get("type") == "gbm":
        model, max_lag, _, _ = best.model_obj
        arr = y.values

        X, yy = _make_lagged_xy(arr, max_lag, horizon=1)
        model.fit(X, yy)

        last_window = arr[-max_lag:].copy()
        preds = []
        for _ in range(30):
            yhat = float(model.predict(last_window.reshape(1, -1))[0])
            preds.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        return pd.Series(preds, index=range(1, 31))

    if best.extra.get("type") == "catboost":
        model, max_lag, _ = best.model_obj
        arr = y.values
        X, yy = _make_lagged_xy(arr, max_lag, horizon=1)
        model.fit(X, yy)

        last_window = arr[-max_lag:].copy()
        preds = []
        for _ in range(30):
            yhat = float(model.predict(last_window.reshape(1, -1))[0])
            preds.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        return pd.Series(preds, index=range(1, 31))

    if best.extra.get("type") == "tcn":
        model, window, _, n_filters, n_stacks, dropout = best.model_obj
        arr = y.values.astype(float)

        if len(arr) <= window + 10:
            last = float(arr[-1])
            logger.warning("TCN refit: too few samples; returning flat series")
            return pd.Series([last] * 30, index=range(1, 31))

        horizon_fc = 1
        X, yy = [], []
        max_idx = len(arr) - horizon_fc + 1
        for s in range(window, max_idx):
            X.append(arr[s - window:s])
            yy.append(arr[s:s + horizon_fc])

        X = np.array(X, dtype="float32")[..., None]
        yy = np.array(yy, dtype="float32")
        mu = float(arr.mean())
        sigma = float(arr.std() + 1e-6)
        X_norm = (X - mu) / sigma
        yy_norm = (yy - mu) / sigma

        def build_refit():
            inp = keras.Input(shape=(window, 1))
            x = inp
            dilation = 1
            for _ in range(n_stacks):
                x = keras.layers.Conv1D(
                    filters=n_filters, kernel_size=3,
                    padding="causal", dilation_rate=dilation,
                    activation="relu",
                )(x)
                x = keras.layers.Dropout(dropout)(x)
                dilation *= 2
            x = keras.layers.GlobalAveragePooling1D()(x)
            out = keras.layers.Dense(horizon_fc)(x)
            m = keras.Model(inp, out, name="TCN")
            m.compile(optimizer="adam", loss="mse")
            return m

        model = build_refit()
        cbs = [keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
        model.fit(
            X_norm, yy_norm,
            epochs=12, batch_size=32,
            verbose=0, callbacks=cbs,
            shuffle=False,
        )

        last_window = arr[-window:].copy()
        preds_30 = []
        for _ in range(30):
            lw_norm = ((last_window - mu) / sigma).reshape(1, window, 1)
            fc_norm = model.predict(lw_norm, verbose=0)[0]
            yhat = float(fc_norm[-1] * sigma + mu)
            if not np.isfinite(yhat):
                yhat = float(last_window[-1])
            preds_30.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        return pd.Series(preds_30, index=range(1, 31))

    if best.extra.get("type") == "lstm":
        mo = best.model_obj
        if len(mo) == 4 and mo[-1] == "returns":
            model, (_, _), window, _ = mo
            p = y.values
            logp = np.log(p + 1e-9)
            r = np.diff(logp).astype("float32").reshape(-1, 1)

            mu_full = float(r.mean())
            sigma_full = float(r.std() + 1e-6)
            r_norm = (r - mu_full) / sigma_full

            X, yy = [], []
            for k in range(window, len(r_norm)):
                X.append(r_norm[k - window:k, 0])
                yy.append(r_norm[k, 0])

            if len(X) == 0:
                logger.warning("LSTM refit: too few samples; returning flat series")
                return pd.Series([float(p[-1])] * 30, index=range(1, 31))

            X = np.array(X)[..., None]
            yy = np.array(yy)

            cbs = [keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)]
            model.fit(
                X, yy,
                epochs=8, batch_size=16,
                verbose=0, callbacks=cbs,
                shuffle=False,
            )

            last_seq = r_norm[-window:, 0].reshape(1, window, 1)
            r_future = []
            for _ in range(30):
                rhat_n = model.predict(last_seq, verbose=0).reshape(-1)[0]
                r_future.append(float(rhat_n * sigma_full + mu_full))
                last_seq = np.concatenate([last_seq[0, 1:, 0], [rhat_n]]).reshape(1, window, 1)

            last_logp = float(logp[-1])
            prices = np.exp(last_logp + np.cumsum(np.array(r_future, dtype=float)))
            return pd.Series(prices, index=range(1, 31))

    if best.extra.get("type") == "nbeats":
        window, _, n_blocks, width, n_hidden = best.model_obj
        arr = y.values.astype(float)

        if len(arr) <= window + 10:
            last = float(arr[-1])
            logger.warning("NBEATS refit: too few samples; returning flat series")
            return pd.Series([last] * 30, index=range(1, 31))

        horizon_fc = 1
        X, yy = [], []
        for t in range(window, len(arr) - horizon_fc + 1):
            X.append(arr[t - window:t])
            yy.append(arr[t:t + horizon_fc])

        X = np.array(X, dtype="float32")
        yy = np.array(yy, dtype="float32")

        mu = float(arr.mean())
        sigma = float(arr.std() + 1e-6)
        X_norm = (X - mu) / sigma

        model = _build_nbeats_model(
            backcast_length=window,
            forecast_length=horizon_fc,
            n_blocks=n_blocks,
            width=width,
            n_hidden=n_hidden,
        )
        cbs = [keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
        model.fit(X_norm, yy, epochs=15, batch_size=32, verbose=0, callbacks=cbs, shuffle=False)

        last_window = arr[-window:].copy()
        preds_30 = []
        for _ in range(30):
            lw_norm = (last_window - mu) / sigma
            fc = model.predict(lw_norm.reshape(1, -1), verbose=0)[0]
            yhat = float(fc[-1])
            if not np.isfinite(yhat):
                yhat = float(last_window[-1])
            preds_30.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat

        return pd.Series(preds_30, index=range(1, 31))

    if best.extra.get("type") == "transformer":
        model, window, _, d_model, n_heads, n_layers, dropout = best.model_obj
        arr = y.values.astype(float)

        if len(arr) <= window + 10:
            last = float(arr[-1])
            logger.warning("TR refit: too few samples; returning flat series")
            return pd.Series([last] * 30, index=range(1, 31))

        horizon_fc = 1
        X, yy = [], []
        max_idx = len(arr) - horizon_fc + 1
        for s in range(window, max_idx):
            X.append(arr[s - window:s])
            yy.append(arr[s:s + horizon_fc])
        X = np.array(X, dtype="float32")[..., None]
        yy = np.array(yy, dtype="float32")

        mu = float(arr.mean())
        sigma = float(arr.std() + 1e-6)
        X_norm = (X - mu) / sigma
        yy_norm = (yy - mu) / sigma

        # rebuild same arch
        tr_preds, tr_obj = _wf_transformer(
            y, val_steps=1, window=window, epochs=8, batch_size=32, horizon=horizon_fc,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        model = tr_obj[0]
        model.fit(X_norm, yy_norm, epochs=8, batch_size=32, verbose=0, shuffle=False)

        last_window = arr[-window:].copy()
        preds_30 = []
        for _ in range(30):
            lw_norm = ((last_window - mu) / sigma).reshape(1, window, 1)
            fc_norm = model.predict(lw_norm, verbose=0)[0]
            yhat = float(fc_norm[-1] * sigma + mu)
            if not np.isfinite(yhat):
                yhat = float(last_window[-1])
            preds_30.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
        return pd.Series(preds_30, index=range(1, 31))

    last = float(y.iloc[-1])
    logger.warning("refit_and_forecast_30d: unknown model type; returning flat series")
    return pd.Series([last] * 30, index=range(1, 31))


# =========================
#           N-BEATS
# =========================

def _build_nbeats_block(
    inp,
    backcast_length: int,
    forecast_length: int,
    width: int = 128,
    n_hidden: int = 4,
    name_prefix: str = "block",
):
    x = inp
    for i in range(n_hidden):
        x = keras.layers.Dense(width, activation="relu", name=f"{name_prefix}_fc{i}")(x)

    theta = keras.layers.Dense(
        backcast_length + forecast_length,
        activation="linear",
        name=f"{name_prefix}_theta"
    )(x)

    backcast = keras.layers.Lambda(
        lambda t: t[..., :backcast_length], name=f"{name_prefix}_backcast"
    )(theta)
    forecast = keras.layers.Lambda(
        lambda t: t[..., backcast_length:], name=f"{name_prefix}_forecast"
    )(theta)
    return backcast, forecast


def _build_nbeats_model(
    backcast_length: int,
    forecast_length: int,
    n_blocks: int = 3,
    width: int = 128,
    n_hidden: int = 4,
) -> keras.Model:
    inp = keras.Input(shape=(backcast_length,), name="input_backcast")
    backcast = inp
    forecast_sum = None

    for b in range(n_blocks):
        b_backcast, b_forecast = _build_nbeats_block(
            backcast,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            width=width,
            n_hidden=n_hidden,
            name_prefix=f"nbeats_b{b}",
        )
        backcast = keras.layers.Subtract(name=f"nbeats_b{b}_backcast_sub")([backcast, b_backcast])
        if forecast_sum is None:
            forecast_sum = b_forecast
        else:
            forecast_sum = keras.layers.Add(name=f"nbeats_b{b}_forecast_add")([forecast_sum, b_forecast])

    model = keras.Model(inputs=inp, outputs=forecast_sum, name="NBEATS")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def _wf_nbeats(
    y: pd.Series,
    val_steps: int,
    window: int = 60,
    horizon: int = 1,
    n_blocks: Optional[int] = None,
    width: Optional[int] = None,
    n_hidden: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Walk-forward N-BEATS:
      - refit раз в WF_REFIT_EVERY шагов
      - forecast_length = horizon
    """
    p = y.astype(float).values
    if len(p) < window + val_steps + 5:
        raise ValueError("Слишком короткий ряд для N-BEATS.")

    n_blocks = n_blocks or NBEATS_BLOCKS
    width = width or NBEATS_WIDTH
    n_hidden = n_hidden or NBEATS_HIDDEN
    epochs = epochs or NBEATS_EPOCHS
    batch_size = batch_size or NBEATS_BATCH

    _log(
        "WF_NBEATS start | window=%d | blocks=%d | width=%d | hidden=%d | val_steps=%d | horizon=%d | refit_every=%d",
        window, n_blocks, width, n_hidden, val_steps, horizon, WF_REFIT_EVERY
    )
    t0 = time.time()

    start = len(p) - val_steps
    n_valid = min(val_steps, max(0, len(p) - (start + horizon - 1)))
    preds: List[float] = []

    model: Optional[keras.Model] = None

    for i in range(n_valid):
        t = start + i
        train = p[:t]

        if len(train) <= window:
            preds.append(float(train[-1]))
            model = None
            continue

        X, yy = [], []
        max_idx = len(train) - horizon
        for s in range(window, max_idx):
            X.append(train[s - window:s])
            yy.append(train[s:s + horizon])

        X = np.array(X, dtype="float32")
        yy = np.array(yy, dtype="float32")
        if X.shape[0] < 8:
            preds.append(float(train[-1]))
            model = None
            continue

        mu = float(train.mean())
        sigma = float(train.std() + 1e-6)
        X_norm = (X - mu) / sigma

        if model is None:
            model = _build_nbeats_model(
                backcast_length=window,
                forecast_length=horizon,
                n_blocks=n_blocks,
                width=width,
                n_hidden=n_hidden,
            )

        if i % WF_REFIT_EVERY == 0:
            base_epochs = epochs if i == 0 else max(2, epochs // 3)
            model.fit(
                X_norm, yy,
                epochs=base_epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
            )
            _log(
                "WF_NBEATS refit | i=%d/%d | train_len=%d | X=%s | epochs=%d",
                i + 1, n_valid, len(train), X_norm.shape, base_epochs
            )

        last_window = train[-window:]
        last_norm = (last_window - mu) / sigma
        fc = model.predict(last_norm.reshape(1, -1), verbose=0)[0]
        yhat = float(fc[-1])
        if not np.isfinite(yhat):
            yhat = float(train[-1])
        preds.append(yhat)

        if MODELS_VERBOSE and ((i + 1) % MODELS_LOG_EVERY == 0 or i == n_valid - 1):
            _log("WF_NBEATS progress | step=%d/%d | last_pred=%.4f", i + 1, n_valid, yhat)

    preds = np.array(preds, dtype=float)
    fill = float(p[start - 1]) if start - 1 >= 0 else float(p[0])
    preds = _pad_to_val_steps(preds, val_steps, fill)

    _log("WF_NBEATS done | elapsed=%.2fs", time.time() - t0)
    nbeats_obj = (window, horizon, n_blocks, width, n_hidden)
    return preds, nbeats_obj
