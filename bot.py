# bot.py
import os
import time
from datetime import time as dtime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
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


# ↓ опционально: тише лог TF (делай это до импортов tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- env ---
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TON_RECEIVER = os.getenv("TON_RECEIVER", "<YOUR_TON_ADDRESS>")
TON_PRICE_TON = float(os.getenv("TON_PRICE_TON", "1.0"))
PRO_DAYS = int(os.getenv("PRO_DAYS", "31"))
SIG_CAPITAL = float(os.getenv("SIGNAL_CAPITAL_USD", "1000"))

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
            InlineKeyboardButton("❓ Все команды", callback_data="menu:help")
        ]
    ])


def _category_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("📈 Акции", callback_data="menu:stocks"),
            InlineKeyboardButton("₿ Крипта", callback_data="menu:crypto"),
            InlineKeyboardButton("💱 Форекс", callback_data="menu:forex"),
        ]]
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
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "all")
    await update.effective_message.reply_text("Signal Mode: все категории (акции+крипта+форекс) ✅")

async def signal_stocks_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "stocks")
    await update.effective_message.reply_text("Signal Mode: только акции ✅")

async def signal_crypto_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    if not is_pro(u.id):
        await update.effective_message.reply_text("Опция доступна только Pro. /pro")
        return
    set_signal_cats(u.id, "crypto")
    await update.effective_message.reply_text("Signal Mode: только крипта ✅")

async def signal_forex_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
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
    Кнопка «🔔 Напомнить … 09:00 МСК» показывается только если есть чёткие сигналы
    (т.е. _pick_reminder_date вернула дату; иначе — кнопки нет).
    """
    try:
        # 1) резолвим тикер и грузим историю
        resolved = resolve_user_ticker(ticker)
        await reply_text_fn(f"Загружаю данные для {resolved} и считаю прогноз. Может занять несколько минут…")

        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=_category_keyboard())
            return

        # 2) три прогноза
        best, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df = train_select_and_forecast(df, ticker=resolved)

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
        except Exception:
            pass

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

        # 8) даты для «Напомнить…» — только если есть маркеры (иначе None)
        date_best = _pick_reminder_date(markers_best,  fcst_best_df)
        date_t3   = _pick_reminder_date(markers_top3, fcst_avg_top3_df)
        date_all  = _pick_reminder_date(markers_all,  fcst_avg_all_df)

        # Клавиатуры «Напомнить» — только если есть реальные маркеры
        kb_best = _reminders_keyboard_from_markers(resolved, "best", markers_best)
        kb_t3   = _reminders_keyboard_from_markers(resolved, "top3", markers_top3)
        kb_all  = _reminders_keyboard_from_markers(resolved, "all",  markers_all)

        # 1/3 best
        if len(cap_best) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_best, caption=cap_best, reply_markup=kb_best) if kb_best \
                else await reply_photo_fn(photo=img_best, caption=cap_best)
        else:
            await reply_photo_fn(photo=img_best, reply_markup=kb_best) if kb_best \
                else await reply_photo_fn(photo=img_best)
            for i in range(0, len(cap_best), TEXT_MAX):
                await reply_text_fn(cap_best[i:i + TEXT_MAX])

        # 2/3 top3
        if len(cap_t3) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_t3, caption=cap_t3, reply_markup=kb_t3) if kb_t3 \
                else await reply_photo_fn(photo=img_t3, caption=cap_t3)
        else:
            await reply_photo_fn(photo=img_t3, reply_markup=kb_t3) if kb_t3 \
                else await reply_photo_fn(photo=img_t3)
            for i in range(0, len(cap_t3), TEXT_MAX):
                await reply_text_fn(cap_t3[i:i + TEXT_MAX])

        # 3/3 all
        if len(cap_all) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_all, caption=cap_all, reply_markup=kb_all) if kb_all \
                else await reply_photo_fn(photo=img_all, caption=cap_all)
        else:
            await reply_photo_fn(photo=img_all, reply_markup=kb_all) if kb_all \
                else await reply_photo_fn(photo=img_all)
            for i in range(0, len(cap_all), TEXT_MAX):
                await reply_text_fn(cap_all[i:i + TEXT_MAX])


        # 10) меню
        await reply_text_fn("Быстрый выбор категории:", reply_markup=_category_keyboard())

        # 11) лог (по лучшей модели)
        log_request(
            user_id=user_id,
            ticker=resolved,
            amount=amount,
            best_model=best['name'],
            metric_name='RMSE',
            metric_value=metrics['rmse'],
            est_profit=profit_best,
        )

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
            pass

    except Exception as e:
        await reply_text_fn(f"Ошибка: {e}", reply_markup=_category_keyboard())



async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    await msg.reply_text("📋 Главное меню:", reply_markup=_main_menu_keyboard())

# --------------- Command handlers ---------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    # schedule_date — это date/datetime
    date_iso = schedule_date.strftime("%Y-%m-%d")
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(f"🔔 Напомнить {date_iso} в 09:00 МСК",
                             callback_data=f"remind:{ticker}:{variant}:{date_iso}")
    ]])

def _pick_reminder_date(markers, fcst_df):
    # markers: [{'buy': pd.Timestamp, ...}, ...]
    try:
        if markers and markers[0].get('buy'):
            return markers[0]['buy'].to_pydatetime().date()
    except Exception:
        pass
    # иначе первая дата прогноза
    return None

def _reminders_keyboard_from_markers(ticker: str, variant: str, markers, max_buttons: int = 6):
    """
    Делает по кнопке на каждую рекомендацию (по buy-дате).
    Формат callback: rmd:<ticker>:<variant>:<YYYY-MM-DD>
    Показываем не больше max_buttons (чтобы не раздувать сообщение).
    """
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
    try:
        user_id = update.effective_user.id if update.effective_user else None
        if user_id is None:
            await msg.reply_text("Не удалось определить пользователя.")
            return

        if len(context.args) < 1:
            await msg.reply_text("Использование: /forecast <TICKER>", reply_markup=_category_keyboard())
            return

        if not can_consume(user_id):
            lim = get_limits(user_id)
            # ✨ CTA при исчерпании
            await msg.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 10 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=_pro_cta_keyboard()
            )
            return

        user_ticker = context.args[0].upper().strip()

        await _run_forecast_for(
            ticker=user_ticker,
            amount=DEFAULT_AMOUNT,
            reply_text_fn=msg.reply_text,
            reply_photo_fn=msg.reply_photo,
            user_id=user_id
        )
    except Exception as e:
        await msg.reply_text(f"Ошибка: {e}", reply_markup=_category_keyboard())

async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_TICKERS, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите акцию:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())

async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_CRYPTO, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())

async def forex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_FOREX, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(rows))
    await msg.reply_text("Хотите больше прогнозов и сигналы? → /pro", reply_markup=_pro_cta_keyboard())

async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    await msg.reply_text(
        "Списки обновлены. Используйте /stocks (акции), /crypto (криптовалюты) и /forex (валютные пары).",
        reply_markup=_category_keyboard(),
    )

async def error_handler(update, context):
    err = context.error
    if isinstance(err, Forbidden):
        return
    try:
        print(f"[ERROR] {err}")
    except Exception:
        pass

# --------------- Callback handler ---------------
async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").strip()

    if data.startswith("forecast:"):
        ticker = data.split(":", 1)[1].strip().upper()
        amount = DEFAULT_AMOUNT

        async def reply_text(text, **kwargs):
            await query.message.reply_text(text, **kwargs)

        async def reply_photo(photo, caption=None, **kwargs):
            await query.message.reply_photo(photo=photo, caption=caption, **kwargs)

        user_id = query.from_user.id if query.from_user else None
        
        if user_id is not None and not can_consume(user_id):
            lim = get_limits(user_id)
            await query.message.reply_text(
                f"Лимит исчерпан. Ваш дневной лимит: {lim}.\n\n"
                "💎 Pro-подписка: 1 TON/мес — 10 прогнозов в день + ежедневные сигналы.\n"
                "Нажмите кнопку ниже, чтобы узнать больше 👇",
                reply_markup=_pro_cta_keyboard()
            )
            return
        
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
    
    if data.startswith("rmd:") or data.startswith("remind:"):
        # форматы:
        # rmd:<ticker>:<variant>:<YYYY-MM-DD>
        # remind:<ticker>:<variant>:<YYYY-MM-DD>  (legacy)
        parts = data.split(":")
        if len(parts) != 4:
            await query.message.reply_text("Неверный формат напоминания.")
            return
        _, ticker, variant, date_iso = parts

        user_id = query.from_user.id if query.from_user else None
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
        from zoneinfo import ZoneInfo
        try:
            dt_local = datetime.strptime(date_iso, "%Y-%m-%d").replace(hour=9, minute=0, second=0, microsecond=0)
            dt_msk = dt_local.replace(tzinfo=ZoneInfo("Europe/Moscow"))
            when_ts = int(dt_msk.timestamp())
        except Exception:
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
    msg = update.effective_message
    u = update.effective_user
    st = get_status(u.id)

    # напоминания: считаем активные и лимит по тарифу
    try:
        active_rmd = count_active(u.id)
    except Exception:
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
    msg = update.effective_message
    u = update.effective_user
    if not is_pro(u.id):
        await msg.reply_text("Сигналы доступны только Pro. Купите подписку: /buy")
        return
    set_signal(u.id, True)
    await msg.reply_text("Signal Mode: включён ✅")

async def signal_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    u = update.effective_user
    set_signal(u.id, False)
    await msg.reply_text("Signal Mode: выключен ❌")

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    text = (f"Оплата Pro: {TON_PRICE_TON} TON на адрес:\n{TON_RECEIVER}\n\n"
            f"После оплаты пришлите хеш транзакции командой:\n/redeem <tx_hash>\n\n"
            "На старте это обрабатывается вручную. Спасибо!")
    await msg.reply_text(text)

async def redeem_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    u = update.effective_user
    args = context.args
    if not args:
        await msg.reply_text("Использование: /redeem <tx_hash>")
        return
    # tx_hash = args[0]  # пока не используем
    now = int(time.time())
    until = now + PRO_DAYS * 86400
    set_tier(u.id, "pro", until)
    await msg.reply_text(f"Pro активирован до {_fmt_until(until)} ✅")

async def _best_of_category(tickers, label, app):
    best = None
    for t in tickers:
        try:
            resolved = resolve_user_ticker(t)
            df = load_ticker_history(resolved)
            if df is None or df.empty:
                continue
            best_m, metrics, fb, fa, ft = train_select_and_forecast(df, ticker=resolved)
            rec_txt, profit, _ = generate_recommendations(
                fb, SIG_CAPITAL, model_rmse=metrics.get('rmse') if metrics else None
            )
            if best is None or profit > best["profit"]:
                best = dict(
                    ticker=resolved, profit=profit, fcst=fb, df=df,
                    rec=rec_txt, metrics=metrics, best_name=best_m["name"]
                )
        except Exception:
            continue
    return best

async def daily_signals(app):
    users = pro_users_for_signal()
    if not users:
        return

    # Чтобы не считать по 100 раз одно и то же, сделаем кэш результатов по категориям/спискам:
    cached_best = {}  # ключ -> dict(...)

    async def best_for_key(key, tickers):
        # key: str ('stocks'|'crypto'|'forex'|'custom:<csv>')
        if key in cached_best:
            return cached_best[key]
        # считаем лучший из набора tickers
        best = await _best_of_category(tickers, key, app)
        cached_best[key] = best
        return best

    for uid in users:
        try:
            st = get_status(uid)
            if not st["signal_enabled"]:
                continue

            mode = get_signal_cats(uid)  # 'all'|'stocks'|'crypto'|'forex'|'custom'
            custom_list = get_signal_list(uid) if mode == "custom" else []

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
                # сохраним ключ для кэша, чтобы у разных пользователей с одним списком не дублировать
                key = "custom:" + ",".join(custom_list)
                await send_item(await best_for_key(key, custom_list), "Выбранные тикеры")
            else:
                # fallback: all
                await send_item(await best_for_key("stocks", SUPPORTED_STOCKS), "Акции")
                await send_item(await best_for_key("crypto", SUPPORTED_CRYPTO), "Крипта")
                await send_item(await best_for_key("forex",  SUPPORTED_FOREX),  "Форекс")

        except Exception:
            continue


async def _send_single_variant(app, user_id: int, ticker: str, variant: str):
    """Пересчитывает прогноз по тикеру и отправляет ОДИН вариант: best/top3/all."""
    resolved = resolve_user_ticker(ticker)
    df = load_ticker_history(resolved)
    if df is None or df.empty:
        await app.bot.send_message(chat_id=user_id, text=f"Напоминание по {resolved}: не удалось загрузить данные.")
        return

    best, metrics, fb, fa, ft = train_select_and_forecast(df, ticker=resolved)

    # выбираем набор
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
    app = context.application
    await daily_signals(app)
    
async def reminders_job(context: ContextTypes.DEFAULT_TYPE):
    """Отправляем напоминания, запланированные на сегодня 09:00 МСК."""
    app = context.application
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    now_msk = datetime.now(ZoneInfo("Europe/Moscow"))
    day_start = now_msk.replace(hour=0, minute=0, second=0, microsecond=0)
    send_start = day_start.replace(hour=9)          # 09:00 МСК
    send_end = send_start + timedelta(hours=1)      # окно 1 час на всякий случай

    day_start_ts = int(send_start.timestamp())
    day_end_ts = int(send_end.timestamp())

    due = due_for_day(day_start_ts, day_end_ts)
    if not due:
        return

    for rem_id, user_id, ticker, variant, when_ts in due:
        try:
            # отправляем ОДИН выбранный вариант прогноза (без списания лимитов)
            await _send_single_variant(app, user_id, ticker, variant)
            mark_sent(rem_id)
        except Exception:
            # не падаем из-за одного пользователя
            continue

# --------------- Entrypoint ---------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")

    init_db()  # БД подписок
    init_reminders()  # БД напоминалок
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # хендлеры
    app.add_handler(CommandHandler("start", start))
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
    app.add_error_handler(error_handler)


    # ежедневные «сигналы» через JobQueue (09:00 по МСК)
    app.job_queue.run_daily(
        daily_signals_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="daily_signals",
    )
    # ежедневные «напоминания» через JobQueue (09:00 по МСК)
    app.job_queue.run_daily(
    reminders_job,
    time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
    name="reminders",
)

    print("Bot is running…")
    app.run_polling()

if __name__ == '__main__':
    main()
