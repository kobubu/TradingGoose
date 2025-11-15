# core/payments_ton.py
import os
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import requests

from core.subs import set_tier, get_status

logger = logging.getLogger("payments")

TONCENTER_API_KEY = os.getenv("TONCENTER_API_KEY", "").strip()
TONCENTER_ENDPOINT = os.getenv("TONCENTER_ENDPOINT", "https://toncenter.com/api/v2").rstrip("/")
TON_RECEIVER = os.getenv("TON_RECEIVER", "").strip()

TON_PRICE_TON = float(os.getenv("TON_PRICE_TON", "1.0"))
PRO_DAYS = int(os.getenv("PRO_DAYS", "31"))

ART_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = ART_DIR / "payments_state.json"


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load payments_state.json")
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save payments_state.json")


def _ton_api_get(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not TONCENTER_API_KEY:
        logger.info("[payments] TONCENTER_API_KEY не задан — пропускаю запрос %s", method)
        raise RuntimeError("TONCENTER_API_KEY not set")

    url = f"{TONCENTER_ENDPOINT}/{method}"
    p = dict(api_key=TONCENTER_API_KEY)
    p.update(params)

    logger.debug("TON API GET %s params=%s", url, p)
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok", True) and "error" in data:
        raise RuntimeError(f"TON API error: {data.get('error')}")
    return data


def _normalize_address(addr: str) -> str:
    if not addr:
        return ""
    return addr.strip()


def _extract_comment(tx: Dict[str, Any]) -> Optional[str]:
    """
    Достаём comment (message) из in_msg, если есть.
    У toncenter обычно это in_msg['message'].
    """
    in_msg = tx.get("in_msg") or {}
    msg = in_msg.get("message")
    if not msg:
        return None
    if isinstance(msg, str):
        return msg.strip()
    return str(msg).strip()


def _extract_value_ton(tx: Dict[str, Any]) -> float:
    """
    value в nanotons -> в TON
    """
    in_msg = tx.get("in_msg") or {}
    val = in_msg.get("value")
    try:
        v = int(val)
        return v / 1e9
    except Exception:
        return 0.0


def _extract_to_address(tx: Dict[str, Any]) -> str:
    in_msg = tx.get("in_msg") or {}
    return _normalize_address(in_msg.get("destination", ""))


def _extract_tx_hash(tx: Dict[str, Any]) -> str:
    tid = tx.get("transaction_id") or {}
    return (tid.get("hash") or "").strip()


def _extract_lt(tx: Dict[str, Any]) -> int:
    tid = tx.get("transaction_id") or {}
    try:
        return int(tid.get("lt") or 0)
    except Exception:
        return 0


def _parse_user_id_from_comment(comment: Optional[str]) -> Optional[int]:
    if not comment:
        return None
    c = comment.strip()
    # Самый простой вариант: всё сообщение — просто число user_id
    try:
        return int(c)
    except Exception:
        # Можно усложнить, искать первую цифру и т.п.
        return None


def _fetch_last_transactions(address: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Берём последние транзакции по адресу через toncenter getTransactions.
    """
    data = _ton_api_get("getTransactions", {
        "address": address,
        "limit": limit,
        "archival": "true",
    })
    res = data.get("result") or []
    logger.info("Fetched %d transactions for address=%s", len(res), address)
    return res


def verify_ton_payment(
    tx_hash: str,
    to_address: str,
    min_amount_ton: float,
    user_id: Optional[int] = None,
) -> Tuple[bool, str, Optional[float]]:
    """
    Возвращает:
      ok:      True/False
      err_msg: текст ошибки или ""
      amount:  фактическая сумма в TON (если ok=True), иначе None
    """

    """
    Проверка конкретного платежа по хэшу (насколько это позволяет toncenter).
    Логика:
      - ищем транзакцию с таким hash среди последних входящих на to_address
      - проверяем, что value >= min_amount_ton
      - если задан user_id, пытаемся вытащить его из комментария и сравнить
    """
    if not TONCENTER_API_KEY:
        logger.info("[payments] TONCENTER_API_KEY не задан — verify_ton_payment пропускаю")
        return False, "TON API не настроен", None

    tx_hash = tx_hash.strip()
    to_address = _normalize_address(to_address)

    try:
        txs = _fetch_last_transactions(to_address, limit=50)
    except Exception as e:
        logger.exception("verify_ton_payment: TON API error for tx_hash=%s", tx_hash)
        return False, f"Ошибка TON API: {e}", None

    matched: Optional[Dict[str, Any]] = None
    for tx in txs:
        h = _extract_tx_hash(tx)
        if not h:
            continue
        if h.lower() == tx_hash.lower():
            matched = tx
            break

    if matched is None:
        logger.warning("verify_ton_payment: tx_hash %s not found for address=%s", tx_hash, to_address)
        return False, "Транзакция не найдена среди последних входящих.", None

    actual_to = _extract_to_address(matched)
    if actual_to and actual_to != to_address:
        logger.warning(
            "verify_ton_payment: tx_hash=%s destination mismatch: %s != %s",
            tx_hash, actual_to, to_address
        )
        return False, "Платёж пришёл на другой адрес.", None

    amount_ton = _extract_value_ton(matched)
    if amount_ton < min_amount_ton:
        logger.warning(
            "verify_ton_payment: tx_hash=%s amount %.6f TON < required %.6f TON",
            tx_hash, amount_ton, min_amount_ton
        )
        return False, f"Недостаточная сумма: {amount_ton:.4f} TON", None

    comment = _extract_comment(matched)
    parsed_uid = _parse_user_id_from_comment(comment)
    if user_id is not None:
        if parsed_uid is None:
            logger.warning(
                "verify_ton_payment: tx_hash=%s no user_id in comment='%s', expected %s",
                tx_hash, comment, user_id
            )
            # Не фейлим, но предупреждаем
        elif parsed_uid != user_id:
            logger.warning(
                "verify_ton_payment: tx_hash=%s comment_user_id=%s != expected %s",
                tx_hash, parsed_uid, user_id
            )
            return False, "ID в комментарии не совпадает с вашим Telegram ID.", None

    logger.info(
        "verify_ton_payment OK: tx_hash=%s to=%s amount=%.6fTON user_id=%s comment=%r",
        tx_hash, to_address, amount_ton, user_id, comment
    )
    return True, "", amount_ton


def _activate_pro_for_user(user_id: int, amount_ton: float, tx_hash: str, comment: Optional[str]) -> None:
    """
    Продлевает подписку пропорционально сумме.
    Пример:
      TON_PRICE_TON = 1.0, PRO_DAYS = 31
      amount_ton = 1.0  -> +31 день
      amount_ton = 2.0  -> +62 дня
      amount_ton = 10.0 -> +310 дней
    """
    now = int(time.time())
    st = get_status(user_id)
    base = max(now, int(st.get("sub_until") or 0))

    # коэффициент: сколько "единичных платежей" пришло
    factor = amount_ton / float(TON_PRICE_TON or 1.0)
    # кол-во дней, пропорциональное сумме (можно округлить как хочешь)
    extra_days = int(PRO_DAYS * factor)

    # на всякий случай минимум 1 день, если вдруг округление дало 0
    if extra_days < 1:
        extra_days = 1

    until = base + extra_days * 86400
    set_tier(user_id, "pro", until)

    logger.info(
        "Pro activated via TON payment: user_id=%s amount=%.6fTON factor=%.3f extra_days=%d until=%s tx=%s comment=%r",
        user_id,
        amount_ton,
        factor,
        extra_days,
        time.strftime("%Y-%m-%d", time.gmtime(until)),
        tx_hash,
        comment,
    )



def scan_and_redeem_incoming(bot) -> None:
    """
    Фоновый сканер:
      - смотрит последние транзакции на TON_RECEIVER
      - ищет новые (по lt) входящие >= TON_PRICE_TON
      - достаёт user_id из комментария
      - активирует / продлевает Pro, шлёт уведомление пользователю
    """
    if not TONCENTER_API_KEY:
        logger.info("[payments] TONCENTER_API_KEY не задан — пропускаю scan_and_redeem_incoming")
        return
    if not TON_RECEIVER:
        logger.warning("[payments] TON_RECEIVER не задан — пропускаю scan_and_redeem_incoming")
        return

    address = TON_RECEIVER
    state = _load_state()
    last_lt = int(state.get("last_lt") or 0)

    logger.info("scan_and_redeem_incoming: start for address=%s last_lt=%s", address, last_lt or "0")

    try:
        txs = _fetch_last_transactions(address, limit=50)
    except Exception as e:
        logger.exception("scan_and_redeem_incoming: TON API error: %s", e)
        return

    if not txs:
        logger.info("scan_and_redeem_incoming: no transactions")
        return

    # сортируем от старых к новым по lt
    txs_sorted = sorted(txs, key=_extract_lt)
    new_last_lt = last_lt
    processed = 0

    for tx in txs_sorted:
        lt = _extract_lt(tx)
        if last_lt and lt <= last_lt:
            continue  # уже видели

        to_addr = _extract_to_address(tx)
        if to_addr and _normalize_address(to_addr) != _normalize_address(address):
            continue  # не наш адрес

        amount_ton = _extract_value_ton(tx)
        if amount_ton < TON_PRICE_TON:
            logger.debug(
                "scan_and_redeem_incoming: skip tx lt=%s amount=%.6fTON < price=%.6fTON",
                lt, amount_ton, TON_PRICE_TON
            )
            new_last_lt = max(new_last_lt, lt)
            continue

        comment = _extract_comment(tx)
        user_id = _parse_user_id_from_comment(comment)
        tx_hash = _extract_tx_hash(tx)

        if user_id is None:
            logger.warning(
                "scan_and_redeem_incoming: incoming tx lt=%s tx_hash=%s amount=%.6fTON has no valid user_id comment=%r",
                lt, tx_hash, amount_ton, comment
            )
            new_last_lt = max(new_last_lt, lt)
            continue

        try:
            _activate_pro_for_user(user_id, amount_ton, tx_hash, comment)
            processed += 1
            # уведомление пользователю
            try:
                until_ts = get_status(user_id).get("sub_until") or 0
                until_str = time.strftime("%Y-%m-%d", time.gmtime(until_ts)) if until_ts else "—"
                text = (
                    f"✅ Оплата {amount_ton:.4f} TON получена.\n"
                    f"Pro-подписка активирована/продлена до *{until_str}*.\n\n"
                    f"Tx: `{tx_hash}`"
                )
                bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
            except Exception:
                logger.exception("Failed to send Telegram notification to user_id=%s", user_id)
        except Exception:
            logger.exception(
                "scan_and_redeem_incoming: failed to activate PRO for user_id=%s tx_hash=%s",
                user_id, tx_hash
            )

        new_last_lt = max(new_last_lt, lt)

    if new_last_lt != last_lt:
        state["last_lt"] = new_last_lt
        _save_state(state)

    logger.info(
        "scan_and_redeem_incoming: finished processed=%d last_lt=%s",
        processed, new_last_lt
    )

def get_payments_state() -> Dict[str, Any]:
    """
    Возвращает raw-состояние сканера платежей (last_lt и т.д.).
    Удобно дергать из /debug_payments.
    """
    return _load_state()


def reset_payments_state() -> None:
    """
    Сбрасывает состояние сканера (last_lt=0).
    Полезно, если хочешь пересканировать историю.
    """
    state = _load_state()
    state["last_lt"] = 0
    _save_state(state)
    logger.warning("payments_state reset by admin: last_lt -> 0")

