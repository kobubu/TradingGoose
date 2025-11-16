# core/favorites.py
import os
import json
import logging

logger = logging.getLogger(__name__)

FAV_FILE = os.path.join("artifacts", "favorites.json")


def _load_favorites():
    if not os.path.exists(FAV_FILE):
        return {}
    try:
        with open(FAV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load favorites file")
        return {}


def _save_favorites(data):
    try:
        tmp = FAV_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, FAV_FILE)
    except Exception:
        logger.exception("Failed to save favorites file")


def get_favorites(user_id: int):
    data = _load_favorites()
    return data.get(str(user_id), [])


def add_favorite(user_id: int, ticker: str):
    data = _load_favorites()
    key = str(user_id)
    favs = data.get(key, [])
    if ticker not in favs:
        favs.append(ticker)
        data[key] = favs
        _save_favorites(data)
    return favs


def remove_favorite(user_id: int, ticker: str):
    data = _load_favorites()
    key = str(user_id)
    favs = data.get(key, [])
    if ticker in favs:
        favs.remove(ticker)
        data[key] = favs
        _save_favorites(data)
    return favs
