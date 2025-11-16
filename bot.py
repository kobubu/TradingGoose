# bot.py
import os
import time
import asyncio
from datetime import time as dtime, timedelta
from zoneinfo import ZoneInfo
import logging
import json
import uuid
import numpy as np

from dotenv import load_dotenv

from telegram import (
    BotCommand,
    InlineQueryResultArticle,
    InputTextMessageContent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import Forbidden
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
)



# ---------- ENV ----------
load_dotenv()

# ---------- LOGGING ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.path.join("artifacts", "bot.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

# приглушаем болтливый httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

PAYMENTS_LOG = os.path.join("artifacts", "payments.log")
MODELS_LOG = os.path.join("artifacts", "models.log")


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

# ↓ тише лог TF (делай это до импортов tensorflow — но здесь мы TF не импортируем)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- env for bot token ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- constants used in this module ---
CAPTION_MAX = 1024
TEXT_MAX = 4096

# --- core imports ---
from core.data import load_ticker_history, resolve_user_ticker
from core.forecast import export_plot_pdf, make_plot_image, train_select_and_forecast
from core.logging_utils import log_request
from core.recommend import generate_recommendations
from core.subs import (
    init_db, get_status, set_signal, is_pro, get_limits, can_consume, consume_one,
    set_tier, pro_users_for_signal,
    set_signal_cats, get_signal_cats, set_signal_list, get_signal_list
)
from core.forecast import (
    export_plot_pdf,
    make_plot_image,
    train_select_and_forecast,
    _make_data_signature,   # ← добавили
)

from core.reminders import init_reminders, add_reminder, count_active, due_for_day, mark_sent
from core import model_cache
from core.favorites import get_favorites, add_favorite, remove_favorite

from ui import (
    HELP_TEXT,
    main_menu_keyboard,
    category_keyboard,
    pro_cta_keyboard,
    build_list_rows,
)

from handlers_pro import (
    DEFAULT_AMOUNT,
    SUPPORTED_TICKERS,
    SUPPORTED_STOCKS,
    SUPPORTED_CRYPTO,
    SUPPORTED_FOREX,
    status_cmd,
    pro_cmd,
    signal_on,
    signal_off,
    signal_all,
    signal_stocks_only,
    signal_crypto_only,
    signal_forex_only,
    signal_custom,
    buy_cmd,
    redeem_cmd,
    debug_payments_cmd,
    debug_payments_reset_cmd,
    debug_models_cmd,
    profile_cmd,
    daily_signals_job,
    reminders_job,
    payments_redeem_job,
    debug_signal_now_cmd,      # ← добавили
    debug_remind_now_cmd,      # ← добавили
)



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
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=category_keyboard())
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
        await reply_text_fn("Быстрый выбор категории:", reply_markup=category_keyboard())

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
                    await reply_text_fn(tip, reply_markup=pro_cta_keyboard())
        except Exception:
            logger.exception("Upsell section failed for user_id=%s ticker=%s", user_id, resolved)

        logger.info("Forecast finished: user_id=%s ticker=%s", user_id, resolved)

    except Exception:
        logger.exception("Error in _run_forecast_for: ticker=%s user_id=%s", ticker, user_id)
        await reply_text_fn("Ошибка при построении прогноза.", reply_markup=category_keyboard())


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


# --------------- Command handlers ---------------

async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/menu from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text("📋 Главное меню:", reply_markup=main_menu_keyboard())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/start from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text(HELP_TEXT, reply_markup=category_keyboard())
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
        reply_markup=pro_cta_keyboard()
    )


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
            await msg.reply_text("Использование: /forecast <TICKER>", reply_markup=category_keyboard())
            return

        if not can_consume(user_id):
            lim = get_limits(user_id)
            logger.info("User %s hit daily limit=%s", user_id, lim)
            await msg.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 10 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=pro_cta_keyboard()
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
        await msg.reply_text("Ошибка при обработке команды /forecast.", reply_markup=category_keyboard())

async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /history <TICKER> — показать последний сохранённый прогноз по тикеру из кэша,
    БЕЗ переобучения моделей и БЕЗ привязки к текущей длине истории.
    """
    msg = update.effective_message
    u = update.effective_user
    logger.info("/history from user_id=%s args=%s", u.id if u else None, context.args)

    if not context.args:
        await msg.reply_text("Использование: /history <TICKER>", reply_markup=category_keyboard())
        return

    user_ticker = context.args[0].upper().strip()

    try:
        # нормализуем тикер, как в /forecast
        try:
            resolved = resolve_user_ticker(user_ticker)
        except Exception:
            resolved = user_ticker

        # грузим историю — для графика (ось X + история)
        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await msg.reply_text("Не удалось загрузить данные по тикеру.", reply_markup=category_keyboard())
            return

        # берём последний forecast для этого тикера из кэша
        fb, fa, ft, meta = model_cache.load_latest_forecasts_for_ticker(resolved)
        if fb is None or meta is None:
            await msg.reply_text(
                "Для этого тикера ещё нет сохранённого прогноза.\n"
                "Сначала сделайте /forecast, чтобы построить и сохранить прогноз.",
                reply_markup=category_keyboard()
            )
            return

        last_close = float(df["Close"].iloc[-1])
        last_fc = float(fb["forecast"].iloc[-1])
        delta = (last_fc - last_close) / last_close * 100.0

        best_name = meta.get("best_name", "cached_best")
        trained_at = meta.get("trained_at")
        if trained_at:
            try:
                trained_at_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(trained_at)))
            except Exception:
                trained_at_str = str(trained_at)
        else:
            trained_at_str = "неизвестно"

        img = make_plot_image(
            df,
            fb,
            resolved,
            markers=None,
            title_suffix="(последний сохранённый прогноз)"
        )

        cap_lines = [
            "📜 Последний сохранённый прогноз",
            f"Тикер: {resolved}",
            f"Лучшая модель: {best_name}",
            f"Дата расчёта: {trained_at_str} (UTC)",
            "",
            f"Оценка изменения цены к концу горизонта: {delta:+.2f}%",
            "",
            "⚠️ Это сохранённый прогноз на момент обучения модели.",
            "Он не пересчитывается заново и не является инвестсоветом.",
        ]
        caption = "\n".join(cap_lines)

        await msg.reply_photo(photo=img, caption=caption[:1024])

    except Exception:
        logger.exception("Error in /history handler for user_id=%s", u.id if u else None)
        await msg.reply_text("Ошибка при выполнении /history.", reply_markup=category_keyboard())


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
    rows = build_list_rows(SUPPORTED_TICKERS, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите акцию:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=pro_cta_keyboard())


async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/crypto from user_id=%s", u.id if u else None)
    rows = build_list_rows(SUPPORTED_CRYPTO, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=pro_cta_keyboard())


async def forex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/forex from user_id=%s", u.id if u else None)
    rows = build_list_rows(SUPPORTED_FOREX, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=pro_cta_keyboard())


async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    logger.info("/tickers from user_id=%s", u.id if u else None)
    msg = update.effective_message
    await msg.reply_text(
        "Списки обновлены. Используйте /stocks (акции), /crypto (криптовалюты) и /forex (валютные пары).",
        reply_markup=category_keyboard(),
    )


# ---------------- Favorites command handlers ----------------

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
            reply_markup=category_keyboard()
        )
        return

    rows = build_list_rows(favs, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])

    await msg.reply_text(
        "⭐ Ваши избранные тикеры:",
        reply_markup=InlineKeyboardMarkup(rows)
    )


# --------------- Error handler ---------------

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
                reply_markup=pro_cta_keyboard()
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
            await query.message.reply_text("Выберите категорию:", reply_markup=category_keyboard())
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
                reply_markup=pro_cta_keyboard()
            )
            return
        if kind == "buy":
            await buy_cmd(update, context)
            return
        if kind == "status":
            await status_cmd(update, context)
            return
        if kind == "help":
            await query.message.reply_text(HELP_TEXT, reply_markup=main_menu_keyboard())
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
                reply_markup=pro_cta_keyboard()
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
            f"Готово! {date_iso} в 09:00 (МСК) я пересчитаю прогноз по {ticker} "
            f"({ 'лучшая модель' if variant=='best' else 'ансамбль топ-3' if variant=='top3' else 'ансамбль всех моделей' }) "
            f"на текущих данных и пришлю обновлённую рекомендацию."
        )
        return


# --------------- Post-init (set commands) ---------------

async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("buy", "Оплата Pro-подписки"),
        BotCommand("pro", "Pro-подписка и Signal Mode"),
        BotCommand("status", "Ваш тариф и лимиты"),
        BotCommand("help", "Помощь по боту"),
    ])


# --------------- Entrypoint ---------------

# --------------- Entrypoint ---------------

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")

    logger.info("Initializing DB and reminders…")
    init_db()
    init_reminders()

    # важно: добавили post_init(post_init)
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    logger.info("Telegram application built")

    # хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))  # ← добавили /help
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("history", history_cmd))
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
    app.add_error_handler(error_handler)

    # джобы
    app.job_queue.run_daily(
        daily_signals_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="daily_signals",
    )
    app.job_queue.run_daily(
        reminders_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="reminders",
    )

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
