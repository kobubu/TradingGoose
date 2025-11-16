import os
import time
import asyncio
import json
import logging
from datetime import datetime, timedelta as _td
from zoneinfo import ZoneInfo
from core import warmup

from telegram import Update
from telegram.ext import ContextTypes

from core.data import load_ticker_history, resolve_user_ticker, MAIN_CRYPTO, MAIN_FOREX
from core.forecast import train_select_and_forecast, make_plot_image
from core.recommend import generate_recommendations
from core.subs import (
    get_status, set_signal, is_pro, get_limits,
    set_tier, pro_users_for_signal,
    set_signal_cats, get_signal_cats, get_signal_list
)
from core.reminders import count_active, due_for_day, mark_sent
from core.payments_ton import (
    scan_and_redeem_incoming,
    verify_ton_payment,
    get_payments_state,
    reset_payments_state,
)
from core import model_cache
from ui import category_keyboard

logger = logging.getLogger(__name__)

# --- env / constants specific to Pro / payments / signals ---

TON_RECEIVER = os.getenv("TON_RECEIVER", "<YOUR_TON_ADDRESS>")
TON_PRICE_TON = float(os.getenv("TON_PRICE_TON", "1.0"))
PRO_DAYS = int(os.getenv("PRO_DAYS", "31"))
SIG_CAPITAL = float(os.getenv("SIGNAL_CAPITAL_USD", "1000"))
BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0") or "0")

# сумма по умолчанию, используется и в напоминаниях, и в основном боте
DEFAULT_AMOUNT = 1000.0

# --- supported tickers (общий конфиг) ---

SUPPORTED_TICKERS = [
    # Big Tech / IT
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "AMD", "INTC", "TSM", "ADBE", "CSCO", "IBM", "ORCL",

    # Consumer / Media
    "TSLA", "NFLX", "DIS", "NKE", "MCD", "KO", "PEP",

    # Finance
    "JPM", "BAC", "GS", "V", "MA",

    # Energy / Industrial / China / Pharma
    "XOM", "BA", "BABA", "PFE",
]
SUPPORTED_STOCKS = SUPPORTED_TICKERS
SUPPORTED_CRYPTO = MAIN_CRYPTO
SUPPORTED_FOREX = MAIN_FOREX


def _fmt_until(ts: int):
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


# --------------- Signal mode command handlers ---------------

async def signal_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("signal_all by user_id=%s", u.id if u else None)
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "all")
    await update.effective_message.reply_text("Signal Mode: все категории (акции+крипта+форекс) ✅")


async def signal_stocks_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("signal_stocks_only by user_id=%s", u.id if u else None)
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "stocks")
    await update.effective_message.reply_text("Signal Mode: только акции ✅")


async def signal_crypto_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("signal_crypto_only by user_id=%s", u.id if u else None)
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "crypto")
    await update.effective_message.reply_text("Signal Mode: только крипта ✅")


async def signal_forex_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("signal_forex_only by user_id=%s", u.id if u else None)
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "forex")
    await update.effective_message.reply_text("Signal Mode: только форекс ✅")


async def signal_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /signal_custom AAPL,MSFT,BTC,EURUSD
    """
    u = update.effective_user
    logger.info("signal_custom by user_id=%s args=%s", u.id if u else None, context.args)
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    args = " ".join(context.args).strip()
    if not args:
        await update.effective_message.reply_text("Использование: /signal_custom <тикеры через запятую>")
        return
    set_signal_cats(u.id, "custom")
    set_signal_list(u.id, args)
    await update.effective_message.reply_text(f"Signal Mode: выбранные тикеры ✅\nСписок: {args}")


# --------------- Status / Pro info ---------------

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/status from user_id=%s", u.id if u else None)
    msg = update.effective_message
    st = get_status(u.id)

    try:
        active_rmd = count_active(u.id)
    except Exception:
        logger.exception("count_active failed for user_id=%s", u.id)
        active_rmd = 0
    rmd_limit = 100 if st.get("tier") == "pro" else 1

    mode = get_signal_cats(u.id)
    lst = get_signal_list(u.id)
    mode_h = {
        "all": "все категории",
        "stocks": "только акции",
        "crypto": "только крипта",
        "forex": "только форекс",
        "custom": "выбранные тикеры",
    }.get(mode, mode)

    extra = f"\nSignal режим: {mode_h}"
    if mode == "custom":
        extra += f" ({', '.join(lst) if lst else 'не задано'})"

    cap = (
        f"Статус: {('PRO' if st['tier']=='pro' else 'FREE')}\n"
        f"Лимит/день: {get_limits(u.id)}\n"
        f"Израсходовано сегодня: {st['daily_count']}\n"
        f"Подписка до: {_fmt_until(st['sub_until'])}\n"
        f"Signal Mode: {'ON' if st['signal_enabled'] else 'OFF'}{extra}\n"
        f"Активных напоминаний: {active_rmd} / {rmd_limit}"
    )
    await msg.reply_text(cap, reply_markup=category_keyboard())


async def pro_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/pro from user_id=%s", u.id if u else None)
    msg = update.effective_message
    txt = (
        "💎 *Pro-подписка*\n"
        "Стоимость: 1 TON / месяц\n\n"
        "Преимущества:\n"
        "• до *10 прогнозов в день* (вместо 3)\n"
        "• *Signal Mode* — бот сам присылает лучшие прогнозы в 09:00 МСК\n"
        "• поддержка напоминаний и расширенных функций\n\n"
        "📡 Режимы Signal Mode:\n"
        "• /signal_all — все категории (акции, крипта, форекс)\n"
        "• /signal_stocks_only — только акции\n"
        "• /signal_crypto_only — только крипта\n"
        "• /signal_forex_only — только форекс\n"
        "• /signal_custom AAPL,MSFT,BTC,EURUSD — свои тикеры\n\n"
        "⚙️ Управление:\n"
        "• /signal_on — включить рассылку\n"
        "• /signal_off — выключить\n\n"
        "Для оплаты используйте команду /buy\n"
        "После активации — включите сигналы: /signal_on"
    )
    await msg.reply_text(txt, parse_mode="Markdown", reply_markup=category_keyboard())


async def signal_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/signal_on from user_id=%s", u.id if u else None)
    msg = update.effective_message
    if not is_pro(u.id):
        await msg.reply_text("Сигналы доступны только Pro. Купите подписку: /buy")
        return
    set_signal(u.id, True)
    await msg.reply_text("Signal Mode: включён ✅")


async def signal_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/signal_off from user_id=%s", u.id if u else None)
    msg = update.effective_message
    set_signal(u.id, False)
    await msg.reply_text("Signal Mode: выключен ❌")


# --------------- Billing / Payments ---------------

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/buy from user_id=%s", u.id if u else None)
    msg = update.effective_message

    text = (
        "💎 Оплата Pro-подписки\n\n"
        f"1️⃣ Отправьте {TON_PRICE_TON} TON на адрес:\n"
        f"`{TON_RECEIVER}`\n\n"
        f"2️⃣ В комментарий к переводу укажите ваш ID: `{u.id}` (обязательно)\n"
        "3️⃣ После перевода пришлите боту хеш транзакции командой:\n"
        "`/redeem <tx_hash>`\n\n"
        "Бот автоматически проверит платёж в сети TON и активирует/продлит подписку. 🚀"
    )
    await msg.reply_text(text, parse_mode="Markdown")


async def redeem_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    u = update.effective_user
    args = context.args
    if not args:
        await msg.reply_text("Использование: /redeem <tx_hash>")
        return

    tx_hash = args[0].strip()

    ok, err_msg, amount = verify_ton_payment(
        tx_hash=tx_hash,
        to_address=TON_RECEIVER,
        min_amount_ton=TON_PRICE_TON,
        user_id=u.id,
    )
    if not ok:
        await msg.reply_text(f"Не удалось подтвердить платёж: {err_msg}")
        return

    if amount is None:
        amount = TON_PRICE_TON

    now = int(time.time())
    st = get_status(u.id)
    base = max(now, int(st.get("sub_until") or 0))

    factor = amount / float(TON_PRICE_TON or 1.0)
    extra_days = int(PRO_DAYS * factor)
    if extra_days < 1:
        extra_days = 1

    until = base + extra_days * 86400
    set_tier(u.id, "pro", until)

    logger.info(
        "redeem_cmd: user_id=%s tx_hash=%s amount=%.6fTON factor=%.3f extra_days=%d until=%s",
        u.id,
        tx_hash,
        amount,
        factor,
        extra_days,
        _fmt_until(until),
    )

    await msg.reply_text(
        f"✅ Платёж подтверждён.\n"
        f"Сумма: {amount:.4f} TON\n"
        f"Подписка продлена на {extra_days} дн.\n"
        f"Pro активирован до {_fmt_until(until)}"
    )


# --------------- Daily signals logic ---------------

async def _best_of_category(tickers, label, app):
    logger.info("Compute best_of_category label=%s tickers=%s", label, tickers)
    best = None
    for t in tickers:
        try:
            resolved = resolve_user_ticker(t)
            df = load_ticker_history(resolved)
            if df is None or df.empty:
                logger.warning("No data for ticker=%s in best_of_category(%s)", resolved, label)
                continue
            best_m, metrics, fb, fa, ft = train_select_and_forecast(df, ticker=resolved)
            rec_txt, profit, _ = generate_recommendations(
                fb, SIG_CAPITAL, model_rmse=metrics.get('rmse') if metrics else None
            )
            logger.debug("Candidate %s profit=%.2f rmse=%.4f", resolved, profit, metrics.get("rmse") if metrics else -1)
            if best is None or profit > best["profit"]:
                best = dict(
                    ticker=resolved, profit=profit, fcst=fb, df=df,
                    rec=rec_txt, metrics=metrics, best_name=best_m["name"]
                )
        except Exception:
            logger.exception("Error in _best_of_category for ticker=%s label=%s", t, label)
            continue
    logger.info("Best_of_category label=%s -> %s", label, best["ticker"] if best else None)
    return best


async def daily_signals(app):
    logger.info("daily_signals job start")
    users = pro_users_for_signal()
    if not users:
        logger.info("daily_signals: no pro users with active sub")
        return

    cached_best = {}

    async def best_for_key(key, tickers):
        if key in cached_best:
            return cached_best[key]
        best = await _best_of_category(tickers, key, app)
        cached_best[key] = best
        return best

    for uid in users:
        try:
            st = get_status(uid)
            if not st["signal_enabled"]:
                continue

            mode = get_signal_cats(uid)
            custom_list = get_signal_list(uid) if mode == "custom" else []
            logger.info("daily_signals for user_id=%s mode=%s custom=%s", uid, mode, custom_list)

            intro = "Дневной сигнал (оценка прибыли на $1,000):\n"
            await app.bot.send_message(chat_id=uid, text=intro)

            async def send_item(best, label):
                if not best or best["profit"] <= 0:
                    await app.bot.send_message(chat_id=uid, text=f"{label}: сильных сигналов нет.")
                    return
                img = make_plot_image(best["df"], best["fcst"], best["ticker"], title_suffix=f"(Сигнал {label})")
                metrics = best.get("metrics") or {}
                rmse_str = f"{metrics.get('rmse'):.2f}" if metrics.get('rmse') is not None else "—"
                cap = (f"{label}: {best['ticker']}\n"
                       f"Модель: {best['best_name']} (RMSE={rmse_str})\n"
                       f"Оценка прибыли: ~ {best['profit']:.2f} USD\n\n"
                       f"{best['rec']}\n\n"
                       "⚠️ Не является инвестсоветом.")
                await app.bot.send_photo(chat_id=uid, photo=img, caption=cap[:1024])

            if mode == "all":
                await send_item(await best_for_key("stocks", SUPPORTED_STOCKS), "Акции")
                await send_item(await best_for_key("crypto", SUPPORTED_CRYPTO), "Крипта")
                await send_item(await best_for_key("forex",  SUPPORTED_FOREX),  "Форекс")
            elif mode == "stocks":
                await send_item(await best_for_key("stocks", SUPPORTED_STOCKS), "Акции")
            elif mode == "crypto":
                await send_item(await best_for_key("crypto", SUPPORTED_CRYPTO), "Крипта")
            elif mode == "forex":
                await send_item(await best_for_key("forex",  SUPPORTED_FOREX),  "Форекс")
            elif mode == "custom":
                key = "custom:" + ",".join(custom_list)
                await send_item(await best_for_key(key, custom_list), "Выбранные тикеры")
            else:
                await send_item(await best_for_key("stocks", SUPPORTED_STOCKS), "Акции")
                await send_item(await best_for_key("crypto", SUPPORTED_CRYPTO), "Крипта")
                await send_item(await best_for_key("forex",  SUPPORTED_FOREX),  "Форекс")

        except Exception:
            logger.exception("daily_signals failed for user_id=%s", uid)
            continue

    logger.info("daily_signals job finished")


async def daily_signals_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("JobQueue: daily_signals_job triggered")
    app = context.application
    await daily_signals(app)


# --------------- Reminders sending ---------------

async def _send_single_variant(app, user_id: int, ticker: str, variant: str):
    """Пересчитывает прогноз по тикеру и отправляет ОДИН вариант: best/top3/all."""
    logger.info("Reminder send_single_variant user_id=%s ticker=%s variant=%s", user_id, ticker, variant)
    resolved = resolve_user_ticker(ticker)
    df = load_ticker_history(resolved)
    if df is None or df.empty:
        await app.bot.send_message(chat_id=user_id, text=f"Напоминание по {resolved}: не удалось загрузить данные.")
        return

    best, metrics, fb, fa, ft = train_select_and_forecast(df, ticker=resolved)

    if variant == "best":
        rec_txt, profit, markers = generate_recommendations(
            fb, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None
        )
        img = make_plot_image(df, fb, resolved, markers=markers, title_suffix="(Напоминание • Лучшая модель)")
        delta = (fb['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"Прогноз пересчитан на текущих данных — модель могла изменить оценку.\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )

    elif variant == "top3":
        rec_txt, profit, markers = generate_recommendations(
            ft, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None
        )
        img = make_plot_image(df, ft, resolved, markers=markers, title_suffix="(Напоминание • Ансамбль топ-3)")
        delta = (ft['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Ансамбль: среднее по топ-3 моделям\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"Прогноз пересчитан на текущих данных — модель могла изменить оценку.\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )
    else:
        rec_txt, profit, markers = generate_recommendations(
            fa, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None
        )
        img = make_plot_image(df, fa, resolved, markers=markers, title_suffix="(Напоминание • Ансамбль всех)")
        delta = (fa['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Ансамбль: среднее по всем моделям\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"Прогноз пересчитан на текущих данных — модель могла изменить оценку.\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )

    await app.bot.send_photo(chat_id=user_id, photo=img, caption=cap[:1024])


async def reminders_job(context: ContextTypes.DEFAULT_TYPE):
    """Отправляем напоминания, запланированные на сегодня 09:00 МСК."""
    logger.info("JobQueue: reminders_job triggered")
    app = context.application

    now_msk = datetime.now(ZoneInfo("Europe/Moscow"))
    day_start = now_msk.replace(hour=0, minute=0, second=0, microsecond=0)
    send_start = day_start.replace(hour=9)
    send_end = send_start + _td(hours=1)

    day_start_ts = int(send_start.timestamp())
    day_end_ts = int(send_end.timestamp())

    due = due_for_day(day_start_ts, day_end_ts)
    logger.info("reminders_job: found %d due reminders", len(due) if due else 0)
    if not due:
        return

    for rem_id, user_id, ticker, variant, when_ts in due:
        try:
            await _send_single_variant(app, user_id, ticker, variant)
            mark_sent(rem_id)
            logger.info("Reminder sent rem_id=%s user_id=%s ticker=%s variant=%s", rem_id, user_id, ticker, variant)
        except Exception:
            logger.exception("Failed to send reminder rem_id=%s user_id=%s", rem_id, user_id)
            continue


# --------------- Payments redeem background job ---------------

async def payments_redeem_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Фоновый job: раз в N минут проверяет новые платежи и активирует Pro.
    """
    logger.info("JobQueue: payments_redeem_job triggered")
    bot = context.application.bot
    try:
        await asyncio.to_thread(scan_and_redeem_incoming, bot)
        logger.info("payments_redeem_job finished scan_and_redeem_incoming")
    except Exception:
        logger.exception("payments_redeem_job failed")


# --------------- Owner-only debug commands ---------------

def _is_owner(user_id: int) -> bool:
    return BOT_OWNER_ID != 0 and user_id == BOT_OWNER_ID


async def debug_payments_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    state = get_payments_state()

    text = "📟 payments_state.json:\n"
    pretty = json.dumps(state, ensure_ascii=False, indent=2, default=str)
    if len(pretty) > 3800:
        pretty = pretty[:3800] + "\n... (truncated)"

    await msg.reply_text(f"{text}```json\n{pretty}\n```", parse_mode="Markdown")


async def debug_payments_reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    reset_payments_state()
    await msg.reply_text("payments_state сброшен (last_lt=0). Следующий проход заново просканирует историю.")


async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/profile from user_id=%s", u.id if u else None)
    msg = update.effective_message

    st = get_status(u.id)
    try:
        active_rmd = count_active(u.id)
    except Exception:
        logger.exception("count_active failed for user_id=%s", u.id)
        active_rmd = 0

    mode = get_signal_cats(u.id)
    lst = get_signal_list(u.id)

    mode_h = {
        "all": "все категории",
        "stocks": "только акции",
        "crypto": "только крипта",
        "forex": "только форекс",
        "custom": "выбранные тикеры",
        None: "по умолчанию (акции+крипта+форекс)",
    }.get(mode, str(mode))

    if mode == "custom":
        mode_h += f" ({', '.join(lst) if lst else 'не задано'})"

    rmd_limit = 100 if st.get("tier") == "pro" else 1

    text = (
        f"👤 Профиль пользователя\n"
        f"ID: `{u.id}`\n\n"
        f"Тариф: *{'PRO' if st['tier'] == 'pro' else 'FREE'}*\n"
        f"Подписка до: {_fmt_until(st['sub_until'])}\n\n"
        f"🔢 Прогнозы сегодня: {st['daily_count']} / {get_limits(u.id)}\n"
        f"🔔 Напоминаний активных: {active_rmd} / {rmd_limit}\n\n"
        f"📡 Signal Mode: {'ON ✅' if st['signal_enabled'] else 'OFF ❌'}\n"
        f"Режим сигналов: {mode_h}\n\n"
        f"Полезные команды:\n"
        f"/status — краткий статус\n"
        f"/pro — про подписку\n"
        f"/buy — оплата\n"
        f"/signal_on / /signal_off — сигналы\n"
    )

    await msg.reply_text(text, parse_mode="Markdown", reply_markup=category_keyboard())


async def debug_models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    info = model_cache.get_cache_info()
    root = info.get("root")
    entries = info.get("entries", [])

    lines = [f"📂 Модельный кэш: {root}", f"Всего моделей: {len(entries)}"]

    MAX_SHOW = 10
    for i, e in enumerate(entries[:MAX_SHOW], start=1):
        meta = e.get("meta") or {}
        dir_name = e.get("dir")
        winner = meta.get("winner") or meta.get("model") or "?"
        trained_at = meta.get("trained_at") or meta.get("trained_time") or "?"
        lines.append(f"{i}. {dir_name} — {winner}, trained_at={trained_at}")

    if len(entries) > MAX_SHOW:
        lines.append(f"... и ещё {len(entries) - MAX_SHOW} записей")

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:4000] + "\n... (truncated)"

    await msg.reply_text(f"```text\n{text}\n```", parse_mode="Markdown")

# тестирование сигналов и напоминаний владельцем бота

async def debug_signal_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /debug_signal_now — принудительный запуск daily_signals() сейчас.
    Только для владельца бота.
    """
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    await msg.reply_text("🚀 Запускаю daily_signals() прямо сейчас…")
    try:
        await daily_signals(context.application)
        await msg.reply_text("✅ daily_signals() завершился. Смотри свои сообщения и логи.")
    except Exception:
        logger.exception("debug_signal_now_cmd failed")
        await msg.reply_text("❌ Ошибка при выполнении daily_signals(). Подробности в логах.")


async def debug_remind_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /debug_remind_now <TICKER> [best|top3|all]
    Отправляет ОДНО напоминание текущему пользователю, минуя таблицу напоминаний.
    Удобно проверить, что _send_single_variant и отправка фото работают.
    """
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    if not context.args:
        await msg.reply_text("Использование: /debug_remind_now <TICKER> [best|top3|all]")
        return

    ticker = context.args[0].upper().strip()
    variant = (context.args[1] if len(context.args) > 1 else "best").lower().strip()
    if variant not in ("best", "top3", "all"):
        variant = "best"

    await msg.reply_text(f"🔔 Тестовое напоминание: {ticker} ({variant}) — отправляю…")

    try:
        await _send_single_variant(context.application, u.id, ticker, variant)
        await msg.reply_text("✅ Тестовое напоминание отправлено (смотри фото выше).")
    except Exception:
        logger.exception("debug_remind_now_cmd failed")
        await msg.reply_text("❌ Ошибка при отправке тестового напоминания. Смотри логи.")


async def debug_warmup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /debug_warmup — показать состояние idle-прогрева моделей (warmup).
    Только для владельца бота.
    """
    u = update.effective_user
    msg = update.effective_message

    if not u or not _is_owner(u.id):
        await msg.reply_text("Эта команда доступна только владельцу бота.")
        return

    info = warmup.get_debug_info()

    lines = [
        "🔥 Warmup debug",
        "",
        f"Idle threshold (sec): {info.get('idle_sec_for_warmup')}",
        f"Job interval (sec):  {info.get('interval_sec')}",
        "",
        f"Last user activity ts:  {info.get('last_user_activity_ts')}",
        f"Last user activity iso: {info.get('last_user_activity_iso')}",
        "",
        f"Current ticker:   {info.get('current_ticker') or '—'}",
        f"WARMUP_INDEX:     {info.get('index')}",
        f"Total tickers:    {info.get('total_tickers')}",
        "",
        "Preview очереди (первые):",
    ]

    preview = info.get("tickers_preview") or []
    if preview:
        # разбиваем по строкам по несколько тикеров
        row = []
        for i, t in enumerate(preview, start=1):
            row.append(t)
            if i % 8 == 0:  # по 8 в строке
                lines.append("  " + ", ".join(row))
                row = []
        if row:
            lines.append("  " + ", ".join(row))
    else:
        lines.append("  <empty>")

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:4000] + "\n... (truncated)"

    await msg.reply_text(f"```text\n{text}\n```", parse_mode="Markdown")
