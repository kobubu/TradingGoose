# bot.py
import io
import os
import time
import asyncio
from datetime import time as dtime, timedelta
from zoneinfo import ZoneInfo
import logging
import uuid
import sys
from datetime import date as _date
from concurrent.futures import ProcessPoolExecutor

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
    JobQueue,
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
payments_logger.propagate = True

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
BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0") or "0")

# --- constants used in this module ---
CAPTION_MAX = 1024
TEXT_MAX = 4096

# --- core imports ---
from core.data import load_ticker_history, resolve_user_ticker
from core.logging_utils import log_request
from core.recommend import generate_recommendations
from core.subs import (
    init_db, get_status, get_limits, can_consume, consume_one,
    pro_users_for_signal,
    set_signal_cats, get_signal_cats, set_signal_list, get_signal_list
)
from core.forecast import (
    train_select_and_forecast,
    _make_data_signature,
    load_cached_forecasts_if_fresh,
    make_fc_key_and_sig,
    load_cached_plot_if_fresh,
)
from core.plot_utils import export_plot_pdf, make_plot_image
from core.reminders import init_reminders, add_reminder, count_active
from core import model_cache
from core.favorites import get_favorites, add_favorite, remove_favorite
from core import warmup

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
    debug_signal_now_cmd,
    debug_remind_now_cmd,
    debug_warmup_cmd,
)

# Глобальный реестр "идущих" прогнозов: signature -> asyncio.Future
INFLIGHT_FORECASTS: dict[str, asyncio.Future] = {}
INFLIGHT_LOCK = asyncio.Lock()

def _no_inflight() -> bool:
    return not INFLIGHT_FORECASTS

warmup.set_inflight_checker(_no_inflight)

FORECAST_WORKERS = int(os.getenv("FORECAST_WORKERS", "2"))
FORECAST_EXECUTOR = ProcessPoolExecutor(max_workers=FORECAST_WORKERS)
logger.info("Using FORECAST_WORKERS=%s", FORECAST_WORKERS)

# --------------- Forecast pipeline ---------------

async def _run_forecast_for(ticker: str, amount: float, reply_text_fn, reply_photo_fn, user_id=None):
    """
    Строит прогноз по лучшей модели, отправляет 1 картинку + текст.
    """
    logger.info("Forecast start: user_id=%s ticker=%s amount=%s", user_id, ticker, amount)
    try:
        # 1) резолвим тикер и грузим историю
        resolved = resolve_user_ticker(ticker)

        df = load_ticker_history(resolved)
        if df is None or df.empty:
            logger.warning("No data for ticker=%s resolved=%s", ticker, resolved)
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=category_keyboard())
            return

        logger.debug("History loaded: ticker=%s len=%d last_dt=%s", resolved, len(df), df.index[-1])

        last_close = float(df["Close"].iloc[-1])

        # 2) общий расчёт для всех конкурентных запросов к этому df/ticker
        best, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df = await _get_shared_forecast(df, resolved)

        logger.info(
            "Models trained/loaded: ticker=%s best=%s rmse=%.4f",
            resolved, best["name"], float(metrics.get("rmse") or -1.0)
        )

        # 3) рекомендации — только по лучшей модели
        rec_best, profit_best, markers_best = generate_recommendations(
            fcst_best_df,
            amount,
            model_rmse=metrics.get("rmse") if metrics else None,
            baseline_price=last_close,
        )

        # 4) картинка только по лучшей модели
        # Сначала пробуем взять готовый PNG из кэша forecasts
        img_best = None
        try:
            png_cached = load_cached_plot_if_fresh(df, resolved)
            if png_cached:
                buf = io.BytesIO(png_cached)
                buf.name = f"{resolved}_forecast.png"
                buf.seek(0)
                img_best = buf
                logger.info("Plot cache HIT for %s", resolved)
        except Exception:
            logger.exception("Plot cache check failed for %s", resolved)

        # Если кэша нет — рисуем и сохраняем PNG в кэш
        if img_best is None:
            img_best = make_plot_image(
                df,
                fcst_best_df,
                resolved,
                markers=markers_best,
                title_suffix="(Лучшая модель)"
            )
            try:
                fc_key, _ = make_fc_key_and_sig(df, resolved)
                model_cache.save_plot(fc_key, img_best.getvalue())
                logger.info("Plot saved to cache for %s", resolved)
            except Exception:
                logger.exception("Failed to save plot cache for %s", resolved)

        # 5) (опционально) PDF
        try:
            from datetime import datetime as _dt
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
            export_plot_pdf(df, fcst_best_df, resolved, os.path.join(art_dir, f"{resolved}_best_{ts}.pdf"))
            logger.debug("PDF exported: ticker=%s ts=%s", resolved, ts)
        except Exception as e:
            logger.warning("PDF export failed for ticker=%s: %s", resolved, e)

        # 6) дельта по лучшему прогнозу
        delta_best = ((fcst_best_df["forecast"].iloc[-1] - last_close) / last_close) * 100.0

        # 7) подпись
        cap_best = (
            f"Тикер: {resolved}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены (30д): {delta_best:+.2f}%\n\n"
            f"{rec_best}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_best:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )

        # 8) даты для «Напомнить…»
        kb_best = _reminders_keyboard_from_markers(resolved, "best", markers_best)

        # Отправляем ОДНУ картинку
        if len(cap_best) <= CAPTION_MAX:
            await (reply_photo_fn(photo=img_best, caption=cap_best, reply_markup=kb_best) if kb_best
                   else reply_photo_fn(photo=img_best, caption=cap_best))
        else:
            await (reply_photo_fn(photo=img_best, reply_markup=kb_best) if kb_best
                   else reply_photo_fn(photo=img_best))
            for i in range(0, len(cap_best), TEXT_MAX):
                await reply_text_fn(cap_best[i:i + TEXT_MAX])

        # 9) меню
        await reply_text_fn("Быстрый выбор категории:", reply_markup=category_keyboard())

        # 10) лог
        try:
            log_request(
                user_id=user_id,
                ticker=resolved,
                amount=amount,
                best_model=best["name"],
                metric_name="RMSE",
                metric_value=metrics["rmse"],
                est_profit=profit_best,
            )
        except Exception:
            logger.exception("log_request failed for user_id=%s ticker=%s", user_id, resolved)

        # 11) upsell
        try:
            if user_id:
                st = get_status(user_id)
                remaining = max(0, get_limits(user_id) - st["daily_count"])
                if st.get("tier") != "pro":
                    tip = (
                        f"Сегодня осталось прогнозов: {remaining}. "
                        f"Проапгрейд до Pro (1 TON/мес) — 20/день + ежедневные сигналы. "
                        f"Команды: /pro • /buy • /signal_on"
                    )
                    await reply_text_fn(tip, reply_markup=pro_cta_keyboard())
        except Exception:
            logger.exception("Upsell section failed for user_id=%s", user_id)

        logger.info("Forecast finished: user_id=%s ticker=%s", user_id, resolved)

    except Exception:
        logger.exception("Error in _run_forecast_for: ticker=%s user_id=%s", ticker, user_id)
        await reply_text_fn("Ошибка при построении прогноза.", reply_markup=category_keyboard())


async def _get_shared_forecast(df, resolved_ticker: str):
    """
    Гарантирует, что для одного и того же df/ticker одновременно
    считается только ОДИН train_select_and_forecast.

    Если прогноз уже есть в кэше и свежий — отдаём сразу.
    """
    # ---------- 0. Быстрая проверка кэша прогнозов ----------
    try:
        cached = load_cached_forecasts_if_fresh(df, resolved_ticker)
        if cached is not None:
            return cached
    except Exception:
        logger.exception("Fast cache check failed for ticker=%s", resolved_ticker)

    # ---------- 1. Если в кэше нет — синхронизация по сигнатуре ----------
    sig = _make_data_signature(df)

    async with INFLIGHT_LOCK:
        fut = INFLIGHT_FORECASTS.get(sig)
        if fut is None:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            INFLIGHT_FORECASTS[sig] = fut
            owner = True
        else:
            owner = False

    if owner:
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                FORECAST_EXECUTOR,
                train_select_and_forecast,
                df,
                resolved_ticker,
            )
            fut.set_result(res)
            return res
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            async with INFLIGHT_LOCK:
                INFLIGHT_FORECASTS.pop(sig, None)
    else:
        return await fut


warmup.set_forecast_fn(_get_shared_forecast)

# ---------------- reminders helpers ----------------

def _pick_reminder_date(markers, fcst_df):
    try:
        if markers and markers[0].get("buy"):
            return markers[0]["buy"].to_pydatetime().date()
    except Exception:
        pass
    return None


def _reminders_keyboard_from_markers(ticker: str, variant: str, markers, max_buttons: int = 6):
    if not markers:
        return None

    entry_dates: list[_date] = []
    for m in (markers or []):
        try:
            if isinstance(m, dict):
                side = m.get("side", "long")
                dt = m.get("sell") if side == "short" else m.get("buy")
            else:
                dt = m[0]

            if dt is None:
                continue

            d = dt.to_pydatetime().date()
            entry_dates.append(d)
        except Exception:
            continue

    if not entry_dates:
        return None

    uniq_dates = sorted(set(entry_dates))[:max_buttons]

    rows = []
    for d in uniq_dates:
        d_iso = d.strftime("%Y-%m-%d")
        rows.append([
            InlineKeyboardButton(
                f"🔔 Напомнить {d_iso} в 09:00 МСК",
                callback_data=f"rmd:{ticker}:{variant}:{d_iso}"
            )
        ])

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
    warmup.mark_user_activity()
    msg = update.effective_message
    u = update.effective_user
    logger.info("/forecast from user_id=%s args=%s", u.id if u else None, context.args)

    try:
        user_id = u.id if u else None
        if user_id is None:
            await msg.reply_text("Не удалось определить пользователя.")
            return

        if len(context.args) < 1:
            await msg.reply_text("Быстрый выбор категории:", reply_markup=category_keyboard())
            return

        if not can_consume(user_id):
            lim = get_limits(user_id)
            logger.info("User %s hit daily limit=%s", user_id, lim)
            await msg.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 20 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=pro_cta_keyboard()
            )
            return

        user_ticker = context.args[0].upper().strip()
        consume_one(user_id)

        await msg.reply_text(
            f"✅ Запрос принят. Сейчас загружу данные по {user_ticker} и посчитаю прогноз.\n"
            f"Это может занять несколько минут, я пришлю результат, когда буду готов.",
        )

        async def _job():
            try:
                await _run_forecast_for(
                    ticker=user_ticker,
                    amount=DEFAULT_AMOUNT,
                    reply_text_fn=msg.reply_text,
                    reply_photo_fn=msg.reply_photo,
                    user_id=user_id
                )
            except Exception:
                logger.exception("Error in forecast background task user_id=%s", user_id)
                await msg.reply_text("Ошибка при построении прогноза.", reply_markup=category_keyboard())

        context.application.create_task(_job())

    except Exception:
        logger.exception("Error in /forecast handler for user_id=%s", u.id if u else None)
        await msg.reply_text("Ошибка при обработке команды /forecast.", reply_markup=category_keyboard())


async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /history <TICKER> — показать последний сохранённый прогноз по тикеру из кэша.
    """
    msg = update.effective_message
    u = update.effective_user
    logger.info("/history from user_id=%s args=%s", u.id if u else None, context.args)

    if not context.args:
        await msg.reply_text("Использование: /history <TICKER>", reply_markup=category_keyboard())
        return

    user_ticker = context.args[0].upper().strip()

    try:
        try:
            resolved = resolve_user_ticker(user_ticker)
        except Exception:
            resolved = user_ticker

        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await msg.reply_text("Не удалось загрузить данные по тикеру.", reply_markup=category_keyboard())
            return

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

        # ---------- пытаемся загрузить сохранённый PNG ----------
        img = None
        try:
            fc_key = (meta or {}).get("fc_key")
            if not fc_key:
                # на старых метах могло не быть fc_key — попробуем посчитать
                fc_key, _ = make_fc_key_and_sig(df, resolved)

            png_bytes = model_cache.load_plot(fc_key)
            if png_bytes:
                buf = io.BytesIO(png_bytes)
                buf.name = f"{resolved}_history.png"
                buf.seek(0)
                img = buf
                logger.info("History plot cache HIT for %s", resolved)
        except Exception:
            logger.exception("Failed to load cached plot for /history %s", resolved)

        # ---------- если PNG нет — рисуем заново ----------
        if img is None:
            img = make_plot_image(
                df,
                fb,
                resolved,
                markers=None,
                title_suffix="(последний сохранённый прогноз)"
            )

        caption = "\n".join([
            "📜 Последний сохранённый прогноз",
            f"Тикер: {resolved}",
            f"Лучшая модель: {best_name}",
            f"Дата расчёта: {trained_at_str} (UTC)",
            "",
            f"Оценка изменения цены к концу горизонта: {delta:+.2f}%",
            "",
            "⚠️ Это сохранённый прогноз на момент обучения модели.",
            "Он не пересчитывается заново и не является инвестсоветом.",
        ])

        await msg.reply_photo(photo=img, caption=caption[:CAPTION_MAX])

    except Exception:
        logger.exception("Error in /history handler for user_id=%s", u.id if u else None)
        await msg.reply_text("Ошибка при выполнении /history.", reply_markup=category_keyboard())


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Inline-режим: @YourBot AAPL -> подсказываем тикеры.
    """
    query = update.inline_query
    if not query:
        return

    q = (query.query or "").strip().upper()

    if not q:
        candidates = SUPPORTED_TICKERS[:6]
    else:
        all_tickers = list(dict.fromkeys(
            list(SUPPORTED_TICKERS) + list(SUPPORTED_CRYPTO) + list(SUPPORTED_FOREX)
        ))
        candidates = [t for t in all_tickers if q in t][:10]

    results = []
    for t in candidates:
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


# --------------- Shutdown and restart handlers ---------------

async def shutdown_cmd(update, context):
    u = update.effective_user
    msg = update.effective_message

    if not u or u.id != BOT_OWNER_ID:
        await msg.reply_text("Эта команда только для владельца бота.")
        return

    logger.info("Shutdown requested by owner user_id=%s", u.id)
    await msg.reply_text("🛑 Останавливаю бота…")

    await asyncio.sleep(0.3)
    await context.application.stop()
    logger.info("Exiting process with os._exit(0) after /shutdown")
    os._exit(0)


async def restart_cmd(update, context):
    u = update.effective_user
    msg = update.effective_message

    if not u or u.id != BOT_OWNER_ID:
        await msg.reply_text("Эта команда только для владельца бота.")
        return

    await msg.reply_text("🔁 Перезапускаю бота…")
    await asyncio.sleep(0.3)

    await context.application.stop()
    import os as _os, sys as _sys
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)


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
        warmup.mark_user_activity()
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
                "💎 Pro-подписка: 1 TON/мес — 20 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=pro_cta_keyboard()
            )
            return

        if user_id is not None:
            consume_one(user_id)

        await query.message.reply_text(
            f"✅ Запрос по {ticker} принят.\n"
            f"Считаю прогноз, это может занять некоторое время.\n"
            f"Результат пришлю сюда, как только будет готов.",
        )

        async def _job():
            try:
                await _run_forecast_for(
                    ticker=ticker,
                    amount=amount,
                    reply_text_fn=reply_text,
                    reply_photo_fn=reply_photo,
                    user_id=user_id
                )
            except Exception:
                logger.exception("Error in inline forecast background task user_id=%s", user_id)
                try:
                    await query.message.reply_text(
                        "Ошибка при построении прогноза.",
                        reply_markup=category_keyboard()
                    )
                except Exception:
                    pass

        context.application.create_task(_job())
        return

    if data.startswith("menu:"):
        kind = data.split(":", 1)[1]
        logger.debug("Menu callback kind=%s user_id=%s", kind, user_id)
        if kind == "root":
            await query.message.reply_text("Выберите категорию:", reply_markup=category_keyboard())
            return
        if kind == "stocks":
            await stocks(update, context); return
        if kind == "crypto":
            await crypto(update, context); return
        if kind == "forex":
            await forex(update, context); return
        if kind == "pro":
            await pro_cmd(update, context); return
        if kind == "buy":
            await buy_cmd(update, context); return
        if kind == "status":
            await status_cmd(update, context); return
        if kind == "help":
            await query.message.reply_text(HELP_TEXT, reply_markup=main_menu_keyboard()); return
        if kind == "fav":
            await fav_list_cmd(update, context); return

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
            f"({'лучшая модель' if variant == 'best' else 'ансамбль топ-3' if variant == 'top3' else 'ансамбль всех моделей'}) "
            f"на текущих данных и пришлю обновлённую рекомендацию."
        )
        return


# --------------- Post-init (set commands) ---------------

async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("forecast", "Прогноз по тикеру / категории"),
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

    jq = JobQueue()

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .job_queue(jq)
        .build()
    )

    logger.info("Telegram application built")

    # хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
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
    app.add_handler(CommandHandler("debug_warmup", debug_warmup_cmd))
    app.add_handler(CommandHandler("debug_signal_now", debug_signal_now_cmd))
    app.add_handler(CommandHandler("debug_remind_now", debug_remind_now_cmd))
    app.add_handler(CommandHandler("fav_add", fav_add_cmd))
    app.add_handler(CommandHandler("fav_remove", fav_remove_cmd))
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_handler(CommandHandler("shutdown", shutdown_cmd))
    app.add_handler(CommandHandler("restart", restart_cmd))
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

    WARMUP_INTERVAL_SEC = int(os.getenv("WARMUP_INTERVAL_SEC", "30"))
    app.job_queue.run_repeating(
        warmup.warmup_job,
        interval=timedelta(seconds=WARMUP_INTERVAL_SEC),
        first=60,
        name="warmup_models",
    )

    logger.info("Bot is starting polling…")
    print("Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C)")
        os._exit(0)
