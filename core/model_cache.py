#model_cache.py
"""Auto-cleanup при запуске можно настроить через env:
- PURGE_MODEL_CACHE_ON_START=1     -> убрать все кэшированные модели
- PURGE_EXPIRED_MODEL_CACHE_ON_START=1 and MODEL_CACHE_TTL_SECONDS=86400 (default)
  -> удалить модели старше Time ti live
"""
from pathlib import Path
import os
import json
import hashlib
import time
import joblib
from typing import Optional, Tuple, Dict, Any

MODEL_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

def _hash_obj(obj) -> str:
    b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(b).hexdigest()

def make_cache_key(ticker: str, model_type: str, params: Dict, data_sig: str) -> str:
    params_hash = _hash_obj(params)
    safe = f"{ticker}__{model_type}__{params_hash}__{data_sig}"
    return _hash_obj(safe)

def _model_dir_for_key(key: str) -> Path:
    d = MODEL_ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_sklearn_model(key: str, model_obj, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)
    joblib.dump(model_obj, d / "model.pkl")
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

def load_sklearn_model(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    mfile = d / "model.pkl"
    mmeta = d / "meta.json"
    if not mfile.exists() or not mmeta.exists():
        return None, None
    model = joblib.load(mfile)
    meta = json.load(open(mmeta, encoding="utf8"))
    return model, meta

def save_statsmodels_result(key: str, res_obj, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)
    joblib.dump(res_obj, d / "sm_res.pkl")
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

def load_statsmodels_result(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    mfile = d / "sm_res.pkl"
    mmeta = d / "meta.json"
    if not mfile.exists() or not mmeta.exists():
        return None, None
    res = joblib.load(mfile)
    meta = json.load(open(mmeta, encoding="utf8"))
    return res, meta

def save_tf_model(key: str, keras_model, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)

    # ✅ путь для новой модели
    model_file = d / "tf_model.keras"

    # удаляем старый SavedModel формата TF1.x, если был
    old_dir = d / "tf_model"
    if old_dir.exists():
        try:
            import shutil
            shutil.rmtree(old_dir)
        except Exception:
            pass

    # ✅ Сохраняем в правильном формате Keras 3
    keras_model.save(str(model_file))

    # ✅ Пишем метаданные
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_tf_model(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    model_file = d / "tf_model.keras"
    mmeta = d / "meta.json"

    if not model_file.exists() or not mmeta.exists():
        return None, None

    from tensorflow import keras
    model = keras.models.load_model(str(model_file))
    meta = json.load(open(mmeta, encoding="utf8"))
    return model, meta


def remove_key(key: str):
    d = MODEL_ROOT / key
    if d.exists():
        try:
            import shutil
            shutil.rmtree(d)
        except Exception:
            pass

def purge_all(reason: str = "") -> int:
    """Remove ALL cached models. Returns number of removed entries."""
    if not MODEL_ROOT.exists():
        return 0
    removed = 0
    for child in MODEL_ROOT.iterdir():
        if child.is_dir():
            try:
                import shutil
                shutil.rmtree(child)
                removed += 1
            except Exception:
                pass
    try:
        print(f"[model_cache] purge_all removed={removed} reason={reason}")
    except Exception:
        pass
    return removed

def purge_expired(ttl_seconds: int = None) -> int:
    if ttl_seconds is None:
        try:
            # читаем единый флаг
            ttl_seconds = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))
        except Exception:
            ttl_seconds = 86400

    now = int(time.time())
    removed = 0
    if not MODEL_ROOT.exists():
        return 0

    for d in MODEL_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        try:
            if not meta_path.exists():
                import shutil
                shutil.rmtree(d)
                removed += 1
                continue
            meta = json.load(open(meta_path, encoding="utf8"))
            trained_at = int(meta.get("trained_at", 0))
            if trained_at <= 0 or (now - trained_at) > ttl_seconds:
                import shutil
                shutil.rmtree(d)
                removed += 1
        except Exception:
            try:
                import shutil
                shutil.rmtree(d)
                removed += 1
            except Exception:
                pass

    try:
        print(f"[model_cache] purge_expired ttl={ttl_seconds}s removed={removed}")
    except Exception:
        pass
    return removed

def get_cache_info() -> Dict[str, Any]:
    """Return brief diagnostic info about cache size and entries."""
    info = {"root": str(MODEL_ROOT), "entries": []}
    if not MODEL_ROOT.exists():
        return info
    for d in MODEL_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta = {}
        try:
            meta_path = d / "meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path, encoding="utf8"))
        except Exception:
            pass
        info["entries"].append({"dir": d.name, "meta": meta})
    return info

def _startup_cleanup():
    """
    Controls:
      PURGE_MODEL_CACHE_ON_START=1          -> purge_all()
      PURGE_EXPIRED_MODEL_CACHE_ON_START=1  -> purge_expired(TTL)
      MODEL_CACHE_TTL_SECONDS               -> TTL value (default 86400)
    """
    try:
        if os.getenv("PURGE_MODEL_CACHE_ON_START", "0") == "1":
            purge_all("env PURGE_MODEL_CACHE_ON_START=1")
        elif os.getenv("PURGE_EXPIRED_MODEL_CACHE_ON_START", "0") == "1":
            ttl = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))
            purge_expired(ttl)
    except Exception:
        pass

# ---------- Forecasts (best / avg_all / avg_top3) ----------
import pandas as pd

_F_BEST   = "forecasts_best.csv"
_F_ALL    = "forecasts_avg_all.csv"
_F_TOP3   = "forecasts_avg_top3.csv"
_F_META   = "forecasts_meta.json"

def make_forecasts_key(ticker: str, data_sig: str) -> str:
    """
    Ключ ТОЛЬКО по тикеру и сигнатуре данных.
    Специально не тащим val_steps/disable_lstm и т.п., чтобы не плодить ключи.
    """
    safe = f"{(ticker or '').upper()}__{data_sig}"
    return _hash_obj(safe)

def save_forecasts(key: str,
                   fcst_best_df: pd.DataFrame,
                   fcst_avg_all_df: pd.DataFrame,
                   fcst_avg_top3_df: pd.DataFrame,
                   meta: Dict[str, Any]) -> None:
    d = _model_dir_for_key(key)
    print(f"[model_cache] save_forecasts -> {d}")
    # Индекс в CSV хранится как ISO8601
    fcst_best_df.to_csv(d / _F_BEST, index=True)
    fcst_avg_all_df.to_csv(d / _F_ALL, index=True)
    fcst_avg_top3_df.to_csv(d / _F_TOP3, index=True)
    with open(d / _F_META, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, default=str)

def load_forecasts(key: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    d = _model_dir_for_key(key)
    print(f"[model_cache] load_forecasts <- {d}")
    p_best = d / _F_BEST
    p_all  = d / _F_ALL
    p_top3 = d / _F_TOP3
    p_meta = d / _F_META
    if not (p_best.exists() and p_all.exists() and p_top3.exists() and p_meta.exists()):
        return None, None, None, None
    def _read(path):
        df = pd.read_csv(path, index_col=0)
        # Попытаемся привести индекс к DatetimeIndex
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return df
    best  = _read(p_best)
    all_  = _read(p_all)
    top3  = _read(p_top3)
    meta  = json.load(open(p_meta, encoding="utf8"))
    return best, all_, top3, meta

def load_latest_forecasts_for_ticker(ticker: str):
    """
    Найти последний (по trained_at) набор forecast'ов для данного тикера.
    Возвращает (fb, fa, ft, meta) или (None, None, None, None), если ничего не найдено.
    """
    ticker = (ticker or "").upper()
    latest_key = None
    latest_meta = None
    latest_ts = 0

    if not MODEL_ROOT.exists():
        return None, None, None, None

    for d in MODEL_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / _F_META
        if not meta_path.exists():
            continue
        try:
            meta = json.load(open(meta_path, encoding="utf8"))
        except Exception:
            continue

        if meta.get("ticker") != ticker:
            continue

        trained_at = int(meta.get("trained_at", 0))
        if trained_at > latest_ts:
            latest_ts = trained_at
            latest_key = d.name
            latest_meta = meta

    if not latest_key:
        return None, None, None, None

    # используем уже готовый load_forecasts
    fb, fa, ft, _ = load_forecasts(latest_key)
    if fb is None or fa is None or ft is None:
        return None, None, None, None

    return fb, fa, ft, latest_meta


_startup_cleanup()