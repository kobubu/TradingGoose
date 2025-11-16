# bot.py
# bot.py
import os
import time
import asyncio
from datetime import time as dtime, timedelta
from zoneinfo import ZoneInfo
import logging
import json
from telegram import BotCommand

from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import InlineQueryHandler
import uuid


from dotenv import load_dotenv

# ---------- ENV ----------
load_dotenv()

# ---------- LOGGING ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", os.path.join("artifacts", "bot.log"))
FAV_FILE = os.path.join("artifacts", "favorites.json")


os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

PAYMENTS_LOG = os.path.join("artifacts", "payments.log")
MODELS_LOG = os.path.join("artifacts", "models.log")
os.makedirs("artifacts", exist_ok=True)

payments_logger = logging.getLogger("payments")
payments_logger.setLevel(logging.INFO)
ph = logging.FileHandler(PAYMENTS_LOG, encoding="utf-8")
ph.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
payments_logger.addHandler(ph)
payments_logger.propagate = True  # или False, если не хочешь дублирования в общий лог

models_logger = logging.getLogger("models")
models_logger.setLevel(logging.INFO)
mh = logging.FileHandler(MODELS_LOG, encoding="utf-8")
mh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
models_logger.addHandler(mh)
models_logger.propagate = True

logger = logging.getLogger(__name__)

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import Forbidden
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# --- core imports ---
from core.data import load_ticker_history, resolve_user_ticker, MAIN_CRYPTO, MAIN_FOREX
from core.forecast import export_plot_pdf, make_plot_image, train_select_and_forecast
from core.logging_utils import log_request
from core.recommend import generate_recommendations
from core.subs import (
    init_db, get_status, set_signal, is_pro, get_limits, can_consume, consume_one,
    set_tier, pro_users_for_signal,
    set_signal_cats, get_signal_cats, set_signal_list, get_signal_list
)
from core.reminders import init_reminders, add_reminder, count_active, due_for_day, mark_sent
from core import model_cache
from core.payments_ton import (
    scan_and_redeem_incoming,
    verify_ton_payment,
    get_payments_state,
    reset_payments_state,
)

# ↓ тише лог TF (делай это до импортов tensorflow — но здесь мы TF не импортируем)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- env ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TON_RECEIVER = os.getenv("TON_RECEIVER", "<YOUR_TON_ADDRESS>")
TON_PRICE_TON = float(os.getenv("TON_PRICE_TON", "1.0"))
PRO_DAYS = int(os.getenv("PRO_DAYS", "31"))
SIG_CAPITAL = float(os.getenv("SIGNAL_CAPITAL_USD", "1000"))
BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0") or "0")

# --- constants ---
DEFAULT_AMOUNT = 1000.0
CAPTION_MAX = 1024
TEXT_MAX = 4096

# --- stocks list (можешь менять) ---
SUPPORTED_TICKERS = [
    # Big Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Autos / Consumer
    "TSLA", "NFLX", "DIS", "NKE", "MCD",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Industrials / Energy
    "BA", "XOM",
]
SUPPORTED_STOCKS = SUPPORTED_TICKERS
SUPPORTED_CRYPTO = MAIN_CRYPTO
SUPPORTED_FOREX = MAIN_FOREX

HELP_TEXT = (
    "Привет! Я бот прогноза акций, криптовалют и форекса.\n\n"
    "Обучаю ML-модели, которые строят предсказания\n\n"
    "Команды:\n"
    "/forecast <TICKER> — пример: /forecast AAPL или /forecast BTC\n"
    "/stocks — быстрый список акций\n"
    "/crypto — топ-10 криптовалют\n"
    "/forex — основные валютные пары\n"
    "/status — ваш тариф и лимиты\n"
    "/pro — про подписку, /buy — оплата, /signal_on, signal_off — включить, выключить сигналы\n\n"
    "Бесплатно: 3 прогноза/день.\n"
    "Pro (1 TON/мес): 10 прогнозов/день + ежедневный «Signal Mode».\n\n"
    "⚠️ Не является инвестсоветом."
)


# ---------------- UI helpers ----------------

def _main_menu_keyboard() -> InlineKeyboardMarkup:
    """Главное меню — категории + навигация."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 Акции", callback_data="menu:stocks"),
            InlineKeyboardButton("₿ Крипта", callback_data="menu:crypto"),
            InlineKeyboardButton("💱 Форекс", callback_data="menu:forex"),
        ],
        [
            InlineKeyboardButton("💎 Pro", callback_data="menu:pro"),
            InlineKeyboardButton("💳 Купить", callback_data="menu:buy"),
            InlineKeyboardButton("ℹ️ Статус", callback_data="menu:status"),
        ],
        [
            InlineKeyboardButton("❓ Все команды", callback_data="menu:help"),
            InlineKeyboardButton("⭐ Избранное", callback_data="menu:fav")
        ]
    ])


def _category_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📈 Акции", callback_data="menu:stocks"),
                InlineKeyboardButton("₿ Крипта", callback_data="menu:crypto"),
                InlineKeyboardButton("💱 Форекс", callback_data="menu:forex"),
            ],
            [
                InlineKeyboardButton("⭐ Избранное", callback_data="menu:fav"),
            ],
        ]
    )


def _pro_cta_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("💎 Pro", callback_data="menu:pro"),
            InlineKeyboardButton("💳 Купить", callback_data="menu:buy"),
            InlineKeyboardButton("ℹ️ Статус", callback_data="menu:status"),
        ]]
    )

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

def _build_list_rows(items, per_row=3):
    rows, row = [], []
    for it in items:
        row.append(InlineKeyboardButton(it, callback_data=f"forecast:{it}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows

def _fmt_until(ts: int):
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d", time.gmtime(ts))

# --------------- Forecast pipeline ---------------

async def _run_forecast_for(ticker: str, amount: float, reply_text_fn, reply_photo_fn, user_id=None):
    """
    Строит 3 прогноза (best/top3/all), отправляет 3 картинки + тексты.
    """
    logger.info("Forecast start: user_id=%s ticker=%s amount=%s", user_id, ticker, amount)
    try:
        # 1) резолвим тикер и грузим историю
        resolved = resolve_user_ticker(ticker)
        await reply_text_fn(f"Загружаю данные для {resolved} и считаю прогноз. Может занять несколько минут…")

        df = load_ticker_history(resolved)
        if df is None or df.empty:
            logger.warning("No data for ticker=%s resolved=%s", ticker, resolved)
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=_category_keyboard())
            return

        logger.debug("History loaded: ticker=%s len=%d last_dt=%s", resolved, len(df), df.index[-1])

        # 2) три прогноза
        best, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df = train_select_and_forecast(df, ticker=resolved)
        logger.info(
            "Models trained/loaded: ticker=%s best=%s rmse=%.4f",
            resolved, best["name"], metrics.get("rmse") if metrics else -1.0
        )

        # 3) рекомендации
        rec_best,  profit_best,  markers_best  = generate_recommendations(
            fcst_best_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )
        rec_all,   profit_all,   markers_all   = generate_recommendations(
            fcst_avg_all_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )
        rec_top3,  profit_top3,  markers_top3  = generate_recommendations(
            fcst_avg_top3_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )

        logger.debug(
            "Recs: ticker=%s profit_best=%.2f profit_top3=%.2f profit_all=%.2f",
            resolved, profit_best, profit_top3, profit_all
        )

        # 4) картинки
        img_best = make_plot_image(df, fcst_best_df,     resolved, markers=markers_best,  title_suffix="(Лучшая модель)")
        img_t3   = make_plot_image(df, fcst_avg_top3_df, resolved, markers=markers_top3, title_suffix="(Ансамбль топ-3)")
        img_all  = make_plot_image(df, fcst_avg_all_df,  resolved, markers=markers_all,  title_suffix="(Ансамбль всех)")

        # 5) (опционально) PDF
        try:
            from datetime import datetime as _dt
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            ts = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
            export_plot_pdf(df, fcst_best_df,     resolved, os.path.join(art_dir, f"{resolved}_best_{ts}.pdf"))
            export_plot_pdf(df, fcst_avg_top3_df, resolved, os.path.join(art_dir, f"{resolved}_avg-top3_{ts}.pdf"))
            export_plot_pdf(df, fcst_avg_all_df,  resolved, os.path.join(art_dir, f"{resolved}_avg-all_{ts}.pdf"))
            logger.debug("PDF exported: ticker=%s ts=%s", resolved, ts)
        except Exception as e:
            logger.warning("PDF export failed for ticker=%s: %s", resolved, e)

        # 6) дельты
        last_close = float(df['Close'].iloc[-1])
        delta_best = ((fcst_best_df['forecast'].iloc[-1]     - last_close) / last_close) * 100.0
        delta_t3   = ((fcst_avg_top3_df['forecast'].iloc[-1] - last_close) / last_close) * 100.0
        delta_all  = ((fcst_avg_all_df['forecast'].iloc[-1]  - last_close) / last_close) * 100.0

        # 7) подписи
        cap_best = (
            f"Тикер: {resolved}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены (30д): {delta_best:+.2f}%\n\n"
            f"{rec_best}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_best:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )
        cap_t3 = (
            f"Тикер: {resolved}\n"
            f"Ансамбль: среднее по топ-3 моделям (минимальный RMSE)\n"
            f"Изменение цены (30д): {delta_t3:+.2f}%\n\n"
            f"{rec_top3}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_top3:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )
        cap_all = (
            f"Тикер: {resolved}\n"
            f"Ансамбль: среднее по всем моделям-кандидатам\n"
            f"Изменение цены (30д): {delta_all:+.2f}%\n\n"
            f"{rec_all}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_all:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )

        # 8) даты для «Напомнить…»
        date_best = _pick_reminder_date(markers_best,  fcst_best_df)
        date_t3   = _pick_reminder_date(markers_top3, fcst_avg_top3_df)
        date_all  = _pick_reminder_date(markers_all,  fcst_avg_all_df)
        logger.debug("Reminder dates: best=%s top3=%s all=%s", date_best, date_t3, date_all)

        # Клавиатуры «Напомнить»
        kb_best = _reminders_keyboard_from_markers(resolved, "best", markers_best)
        kb_t3   = _reminders_keyboard_from_markers(resolved, "top3", markers_top3)
        kb_all  = _reminders_keyboard_from_markers(resolved, "all",  markers_all)

        # 1/3 best
        if len(cap_best) <= CAPTION_MAX:
            await (reply_photo_fn(photo=img_best, caption=cap_best, reply_markup=kb_best) if kb_best
                   else reply_photo_fn(photo=img_best, caption=cap_best))
        else:
            await (reply_photo_fn(photo=img_best, reply_markup=kb_best) if kb_best
                   else reply_photo_fn(photo=img_best))
            for i in range(0, len(cap_best), TEXT_MAX):
                await reply_text_fn(cap_best[i:i + TEXT_MAX])

        # 2/3 top3
        if len(cap_t3) <= CAPTION_MAX:
            await (reply_photo_fn(photo=img_t3, caption=cap_t3, reply_markup=kb_t3) if kb_t3
                   else reply_photo_fn(photo=img_t3, caption=cap_t3))
        else:
            await (reply_photo_fn(photo=img_t3, reply_markup=kb_t3) if kb_t3
                   else reply_photo_fn(photo=img_t3))
            for i in range(0, len(cap_t3), TEXT_MAX):
                await reply_text_fn(cap_t3[i:i + TEXT_MAX])

        # 3/3 all
        if len(cap_all) <= CAPTION_MAX:
            await (reply_photo_fn(photo=img_all, caption=cap_all, reply_markup=kb_all) if kb_all
                   else reply_photo_fn(photo=img_all, caption=cap_all))
        else:
            await (reply_photo_fn(photo=img_all, reply_markup=kb_all) if kb_all
                   else reply_photo_fn(photo=img_all))
            for i in range(0, len(cap_all), TEXT_MAX):
                await reply_text_fn(cap_all[i:i + TEXT_MAX])

        # 10) меню
        await reply_text_fn("Быстрый выбор категории:", reply_markup=_category_keyboard())

        # 11) лог (по лучшей модели)
        try:
            log_request(
                user_id=user_id,
                ticker=resolved,
                amount=amount,
                best_model=best['name'],
                metric_name='RMSE',
                metric_value=metrics['rmse'],
                est_profit=profit_best,
            )
        except Exception:
            logger.exception("log_request failed for user_id=%s ticker=%s", user_id, resolved)

        # 12) мягкий upsell (если не Pro)
        try:
            if user_id:
                st = get_status(user_id)
                remaining = max(0, get_limits(user_id) - st["daily_count"])
                if st.get("tier") != "pro":
                    tip = (
                        f"Сегодня осталось прогнозов: {remaining}. "
                        f"Проапгрейд до Pro (1 TON/мес) — 10/день + ежедневные сигналы. "
                        f"Команды: /pro • /buy • /signal_on"
                    )
                    await reply_text_fn(tip, reply_markup=_pro_cta_keyboard())
        except Exception:
            logger.exception("Upsell section failed for user_id=%s ticker=%s", user_id, resolved)

        logger.info("Forecast finished: user_id=%s ticker=%s", user_id, resolved)

    except Exception:
        logger.exception("Error in _run_forecast_for: ticker=%s user_id=%s", ticker, user_id)
        await reply_text_fn("Ошибка при построении прогноза.", reply_markup=_category_keyboard())


async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/menu from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text("📋 Главное меню:", reply_markup=_main_menu_keyboard())

# --------------- Command handlers ---------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/start from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text(HELP_TEXT, reply_markup=_category_keyboard())
    await msg.reply_text(
        "Полезное:\n"
        "💎 /pro — про подписку и Signal Mode\n"
        "💳 /buy — как оплатить\n"
        "📡 /signal_on — включить сигналы (Pro)\n"
        "🛰 /signal_all — все категории\n"
        "📈 /signal_stocks_only — только акции\n"
        "₿ /signal_crypto_only — только крипта\n"
        "💱 /signal_forex_only — только форекс\n"
        "🎯 /signal_custom <тикеры> — свои тикеры\n\n"
        "💬 /status — ваш тариф, лимиты и напоминания",
        reply_markup=_pro_cta_keyboard()
    )


def _reminder_keyboard(ticker: str, variant: str, schedule_date) -> InlineKeyboardMarkup:
    date_iso = schedule_date.strftime("%Y-%m-%d")
    return InlineKeyboardMarkup([[  # legacy не используется, оставляем
        InlineKeyboardButton(f"🔔 Напомнить {date_iso} в 09:00 МСК",
                             callback_data=f"remind:{ticker}:{variant}:{date_iso}")
    ]])


def _pick_reminder_date(markers, fcst_df):
    try:
        if markers and markers[0].get('buy'):
            return markers[0]['buy'].to_pydatetime().date()
    except Exception:
        pass
    return None


def _reminders_keyboard_from_markers(ticker: str, variant: str, markers, max_buttons: int = 6):
    rows = []
    cnt = 0
    for m in (markers or []):
        try:
            d = m.get("buy")
            if not d:
                continue
            d_iso = d.to_pydatetime().date().strftime("%Y-%m-%d")
            rows.append([InlineKeyboardButton(
                f"🔔 Напомнить {d_iso} в 09:00 МСК",
                callback_data=f"rmd:{ticker}:{variant}:{d_iso}"
            )])
            cnt += 1
            if cnt >= max_buttons:
                break
        except Exception:
            continue
    return InlineKeyboardMarkup(rows) if rows else None


async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    u = update.effective_user
    logger.info("/forecast from user_id=%s args=%s", u.id if u else None, context.args)
    try:
        user_id = u.id if u else None
        if user_id is None:
            await msg.reply_text("Не удалось определить пользователя.")
            return

        if len(context.args) < 1:
            await msg.reply_text("Использование: /forecast <TICKER>", reply_markup=_category_keyboard())
            return

        if not can_consume(user_id):
            lim = get_limits(user_id)
            logger.info("User %s hit daily limit=%s", user_id, lim)
            await msg.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 10 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=_pro_cta_keyboard()
            )
            return

        user_ticker = context.args[0].upper().strip()
        consume_one(user_id)

        await _run_forecast_for(
            ticker=user_ticker,
            amount=DEFAULT_AMOUNT,
            reply_text_fn=msg.reply_text,
            reply_photo_fn=msg.reply_photo,
            user_id=user_id
        )
    except Exception:
        logger.exception("Error in /forecast handler for user_id=%s", u.id if u else None)
        await msg.reply_text("Ошибка при обработке команды /forecast.", reply_markup=_category_keyboard())

async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Inline-режим: @YourBot AAPL -> подсказываем тикеры.
    При выборе варианта в чат отправится текст вида "/forecast AAPL".
    """
    query = update.inline_query
    if not query:
        return

    q = (query.query or "").strip().upper()

    # Если пользователь ничего не ввёл — покажем несколько популярных тикеров
    if not q:
        candidates = SUPPORTED_TICKERS[:6]  # первые 6 акций
    else:
        # Ищем по всем спискам тикеров
        all_tickers = list(dict.fromkeys(
            list(SUPPORTED_TICKERS) + list(SUPPORTED_CRYPTO) + list(SUPPORTED_FOREX)
        ))
        candidates = [t for t in all_tickers if q in t][:10]  # максимум 10 совпадений

    results = []
    for t in candidates:
        # Текст, который реально отправится в чат при выборе
        msg_text = f"/forecast {t}"

        results.append(
            InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title=f"{t} — построить прогноз",
                description=f"Отправить команду: {msg_text}",
                input_message_content=InputTextMessageContent(msg_text),
            )
        )

    await query.answer(results, cache_time=60, is_personal=True)


async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/stocks from user_id=%s", u.id if u else None)
    rows = _build_list_rows(SUPPORTED_TICKERS, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите акцию:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())


async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/crypto from user_id=%s", u.id if u else None)
    rows = _build_list_rows(SUPPORTED_CRYPTO, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())


async def forex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/forex from user_id=%s", u.id if u else None)
    rows = _build_list_rows(SUPPORTED_FOREX, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())


async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/tickers from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text(
        "Списки обновлены. Используйте /stocks (акции), /crypto (криптовалюты) и /forex (валютные пары).",
        reply_markup=_category_keyboard(),
    )


async def error_handler(update, context):
    err = context.error
    if isinstance(err, Forbidden):
        return
    logger.exception("Unhandled error in application: %s", err)


# --------------- Callback handler ---------------

async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").strip()
    user_id = query.from_user.id if query.from_user else None
    logger.info("Callback from user_id=%s data=%s", user_id, data)

    if data.startswith("forecast:"):
        ticker = data.split(":", 1)[1].strip().upper()
        amount = DEFAULT_AMOUNT

        async def reply_text(text, **kwargs):
            await query.message.reply_text(text, **kwargs)

        async def reply_photo(photo, caption=None, **kwargs):
            await query.message.reply_photo(photo=photo, caption=caption, **kwargs)

        if user_id is not None and not can_consume(user_id):
            lim = get_limits(user_id)
            logger.info("User %s hit daily limit on inline forecast; limit=%s", user_id, lim)
            await query.message.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 10 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=_pro_cta_keyboard()
            )
            return
        if user_id is not None:
            consume_one(user_id)

        await _run_forecast_for(
            ticker=ticker,
            amount=amount,
            reply_text_fn=reply_text,
            reply_photo_fn=reply_photo,
            user_id=user_id
        )
        return

    if data.startswith("menu:"):
        kind = data.split(":", 1)[1]
        logger.debug("Menu callback kind=%s user_id=%s", kind, user_id)
        if kind == "root":
            await query.message.reply_text("Выберите категорию:", reply_markup=_category_keyboard())
            return
        if kind == "stocks":
            await stocks(update, context)
            return
        if kind == "crypto":
            await crypto(update, context)
            return
        if kind == "forex":
            await forex(update, context)
            return
        if kind == "pro":
            await query.message.reply_text(
                "Pro-подписка: 1 TON/мес. 10 прогнозов/день + ежедневные сигналы.\nКоманды: /buy, /signal_on",
                reply_markup=_pro_cta_keyboard()
            )
            return
        if kind == "buy":
            await buy_cmd(update, context)
            return
        if kind == "status":
            await status_cmd(update, context)
            return
        if kind == "help":
            await query.message.reply_text(HELP_TEXT, reply_markup=_main_menu_keyboard())
            return
        if kind == "fav":
            await fav_list_cmd(update, context)
            return

    if data.startswith("rmd:") or data.startswith("remind:"):
        parts = data.split(":")
        if len(parts) != 4:
            await query.message.reply_text("Неверный формат напоминания.")
            return
        _, ticker, variant, date_iso = parts
        logger.info("Reminder callback user_id=%s ticker=%s variant=%s date=%s", user_id, ticker, variant, date_iso)

        if not user_id:
            await query.message.reply_text("Не удалось определить пользователя.")
            return

        st = get_status(user_id)
        active = count_active(user_id)
        limit = 100 if st.get("tier") == "pro" else 1
        if active >= limit:
            await query.message.reply_text(
                f"Достигнут лимит активных напоминаний ({active}/{limit}). "
                f"Очистите старые (они автоматически исчезают после отправки) или оформите Pro.",
                reply_markup=_pro_cta_keyboard()
            )
            return

        from datetime import datetime
        try:
            dt_local = datetime.strptime(date_iso, "%Y-%m-%d").replace(hour=9, minute=0, second=0, microsecond=0)
            dt_msk = dt_local.replace(tzinfo=ZoneInfo("Europe/Moscow"))
            when_ts = int(dt_msk.timestamp())
        except Exception:
            logger.exception("Failed to parse reminder date: %s", date_iso)
            await query.message.reply_text("Не удалось распознать дату для напоминания.")
            return

        add_reminder(user_id, ticker, variant, when_ts)
        await query.message.reply_text(
            f"Готово! Напомню про {ticker} ({'Лучшая' if variant=='best' else 'Топ-3' if variant=='top3' else 'Все'}) "
            f"{date_iso} в 09:00 (МСК)."
        )
        return

# --------------- Pro / Billing / Signals ---------------

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
    lst  = get_signal_list(u.id)
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
    await msg.reply_text(cap, reply_markup=_category_keyboard())


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
    await msg.reply_text(txt, parse_mode="Markdown", reply_markup=_category_keyboard())


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
        # на всякий случай, но по факту сюда не попадём
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

    # лог в основной логгер
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
        fcst_df = fb
        rec_txt, profit, markers = generate_recommendations(fb, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None)
        img = make_plot_image(df, fb, resolved, markers=markers, title_suffix="(Напоминание • Лучшая модель)")
        delta = (fb['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )
    elif variant == "top3":
        fcst_df = ft
        rec_txt, profit, markers = generate_recommendations(ft, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None)
        img = make_plot_image(df, ft, resolved, markers=markers, title_suffix="(Напоминание • Ансамбль топ-3)")
        delta = (ft['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Ансамбль: среднее по топ-3 моделям\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )
    else:
        fcst_df = fa
        rec_txt, profit, markers = generate_recommendations(fa, DEFAULT_AMOUNT, model_rmse=metrics.get('rmse') if metrics else None)
        img = make_plot_image(df, fa, resolved, markers=markers, title_suffix="(Напоминание • Ансамбль всех)")
        delta = (fa['forecast'].iloc[-1] - float(df['Close'].iloc[-1])) / float(df['Close'].iloc[-1]) * 100.0
        cap = (
            f"🔔 Напоминание\nТикер: {resolved}\n"
            f"Ансамбль: среднее по всем моделям\n"
            f"Изменение цены (30д): {delta:+.2f}%\n\n"
            f"{rec_txt}\n\n"
            "⚠️ Не является инвестсоветом."
        )

    await app.bot.send_photo(chat_id=user_id, photo=img, caption=cap[:1024])


async def daily_signals_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("JobQueue: daily_signals_job triggered")
    app = context.application
    await daily_signals(app)


async def reminders_job(context: ContextTypes.DEFAULT_TYPE):
    """Отправляем напоминания, запланированные на сегодня 09:00 МСК."""
    logger.info("JobQueue: reminders_job triggered")
    app = context.application
    from datetime import datetime, timedelta as _td

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
    # телега максимум 4096 символов – подрежем на всякий
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

    # Signal-режим
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

    await msg.reply_text(text, parse_mode="Markdown", reply_markup=_category_keyboard())


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

    # чуть-чуть подробностей по первым N моделям
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

# ---------------- Favorites storage ----------------

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

async def fav_add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message
    logger.info("/fav_add from user_id=%s args=%s", u.id if u else None, context.args)

    if not u:
        await msg.reply_text("Не удалось определить пользователя.")
        return

    if not context.args:
        await msg.reply_text("Использование: /fav_add <TICKER>")
        return

    user_ticker = context.args[0].upper().strip()
    # Можно сразу прогнать через resolve_user_ticker, чтобы нормализовать:
    try:
        resolved = resolve_user_ticker(user_ticker)
    except Exception:
        resolved = user_ticker

    favs = add_favorite(u.id, resolved)
    await msg.reply_text(
        f"Тикер {resolved} добавлен в избранное.\n"
        f"Текущее избранное: {', '.join(favs)}"
    )


async def fav_remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message
    logger.info("/fav_remove from user_id=%s args=%s", u.id if u else None, context.args)

    if not u:
        await msg.reply_text("Не удалось определить пользователя.")
        return

    if not context.args:
        await msg.reply_text("Использование: /fav_remove <TICKER>")
        return

    user_ticker = context.args[0].upper().strip()
    try:
        resolved = resolve_user_ticker(user_ticker)
    except Exception:
        resolved = user_ticker

    favs = remove_favorite(u.id, resolved)
    await msg.reply_text(
        f"Тикер {resolved} удалён из избранного.\n"
        f"Текущее избранное: {', '.join(favs) if favs else 'пусто'}"
    )


async def fav_list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    msg = update.effective_message
    logger.info("/fav from user_id=%s", u.id if u else None)

    if not u:
        await msg.reply_text("Не удалось определить пользователя.")
        return

    favs = get_favorites(u.id)
    if not favs:
        await msg.reply_text(
            "У вас пока нет избранных тикеров.\n"
            "Добавьте через /fav_add <TICKER>.\n\n"
            "Например: /fav_add AAPL",
            reply_markup=_category_keyboard()
        )
        return

    rows = _build_list_rows(favs, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])

    await msg.reply_text(
        "⭐ Ваши избранные тикеры:",
        reply_markup=InlineKeyboardMarkup(rows)
    )



async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("buy", "Оплата Pro-подписки"),
        BotCommand("pro", "Pro-подписка и Signal Mode"),
        BotCommand("status", "Ваш тариф и лимиты"),
        BotCommand("help", "Помощь по боту"),
    ])

# --------------- Entrypoint ---------------

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")

    logger.info("Initializing DB and reminders…")
    init_db()
    init_reminders()

    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    logger.info("Telegram application built")

    # хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("stocks", stocks))
    app.add_handler(CommandHandler("crypto", crypto))
    app.add_handler(CommandHandler("forex", forex))
    app.add_handler(CommandHandler("tickers", tickers))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("pro", pro_cmd))
    app.add_handler(CommandHandler("signal_on", signal_on))
    app.add_handler(CommandHandler("signal_off", signal_off))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("redeem", redeem_cmd))
    app.add_handler(CallbackQueryHandler(_on_callback))
    app.add_handler(CommandHandler("menu", menu_cmd))
    app.add_handler(CommandHandler("signal_all", signal_all))
    app.add_handler(CommandHandler("signal_stocks_only", signal_stocks_only))
    app.add_handler(CommandHandler("signal_crypto_only", signal_crypto_only))
    app.add_handler(CommandHandler("signal_forex_only", signal_forex_only))
    app.add_handler(CommandHandler("signal_custom", signal_custom))
    app.add_handler(CommandHandler("debug_payments", debug_payments_cmd))
    app.add_handler(CommandHandler("debug_payments_reset", debug_payments_reset_cmd))
    app.add_handler(CommandHandler("debug_models", debug_models_cmd))
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("fav_add", fav_add_cmd))
    app.add_handler(CommandHandler("fav_remove", fav_remove_cmd))
    app.add_handler(CommandHandler("fav", fav_list_cmd))
    app.add_handler(CommandHandler("favorites", fav_list_cmd))

    app.add_error_handler(error_handler)
    

    # ежедневные «сигналы» через JobQueue (09:00 по МСК)
    app.job_queue.run_daily(
        daily_signals_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="daily_signals",
    )
    # ежедневные «напоминания»
    app.job_queue.run_daily(
        reminders_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="reminders",
    )

    # фоновый redeem job — каждые N минут (по умолчанию 2 мин)
    INTERVAL_MIN = int(os.getenv("TON_REDEEM_INTERVAL_MIN", "2"))
    app.job_queue.run_repeating(
        payments_redeem_job,
        interval=timedelta(minutes=INTERVAL_MIN),
        first=10,
        name="payments_redeem",
    )

    logger.info("Bot is starting polling…")
    print("Bot is running…")
    app.run_polling()


if __name__ == '__main__':
    main()
