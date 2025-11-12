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
    init_db,
    get_status,
    set_signal,
    is_pro,
    get_limits,
    can_consume,
    consume_one,
    set_tier,
    pro_users_for_signal,
)

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
    "/pro — про подписку, /buy — оплата, /signal_on — включить сигналы\n\n"
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
    try:
        resolved = resolve_user_ticker(ticker)
        await reply_text_fn(f"Загружаю данные для {resolved} и считаю прогноз. Может занять несколько минут…")
        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=_category_keyboard())
            return
        if user_id is not None:
            try:
                consume_one(user_id)
            except Exception:
                pass        

        # три прогноза
        best, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df = train_select_and_forecast(df, ticker=resolved)

        # рекомендации
        rec_best,  profit_best,  markers_best  = generate_recommendations(
            fcst_best_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )
        rec_all,   profit_all,   markers_all   = generate_recommendations(
            fcst_avg_all_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )
        rec_top3,  profit_top3,  markers_top3  = generate_recommendations(
            fcst_avg_top3_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )

        # 3 картинки
        img_best = make_plot_image(df, fcst_best_df,     resolved, markers=markers_best,  title_suffix="(Лучшая модель)")
        img_t3   = make_plot_image(df, fcst_avg_top3_df, resolved, markers=markers_top3, title_suffix="(Ансамбль топ-3)")
        img_all  = make_plot_image(df, fcst_avg_all_df,  resolved, markers=markers_all,  title_suffix="(Ансамбль всех)")

        # 3 PDF-артефакта (опционально)
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

        # дельты к последнему Close
        last_close = float(df['Close'].iloc[-1])
        delta_best = ((fcst_best_df['forecast'].iloc[-1]     - last_close) / last_close) * 100.0
        delta_t3   = ((fcst_avg_top3_df['forecast'].iloc[-1] - last_close) / last_close) * 100.0
        delta_all  = ((fcst_avg_all_df['forecast'].iloc[-1]  - last_close) / last_close) * 100.0

        # подписи
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

        # отправка 3 изображений
        if len(cap_best) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_best, caption=cap_best)
        else:
            await reply_photo_fn(photo=img_best)
            for i in range(0, len(cap_best), TEXT_MAX):
                await reply_text_fn(cap_best[i:i + TEXT_MAX])

        if len(cap_t3) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_t3, caption=cap_t3)
        else:
            await reply_photo_fn(photo=img_t3)
            for i in range(0, len(cap_t3), TEXT_MAX):
                await reply_text_fn(cap_t3[i:i + TEXT_MAX])

        if len(cap_all) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_all, caption=cap_all)
        else:
            await reply_photo_fn(photo=img_all)
            for i in range(0, len(cap_all), TEXT_MAX):
                await reply_text_fn(cap_all[i:i + TEXT_MAX])

        # меню
        await reply_text_fn("📋 Главное меню:", reply_markup=_main_menu_keyboard())


        # лог (по лучшей модели)
        log_request(
            user_id=user_id,
            ticker=resolved,
            amount=amount,
            best_model=best['name'],
            metric_name='RMSE',
            metric_value=metrics['rmse'],
            est_profit=profit_best,
        )

        # мягкий upsell (если юзер не Pro)
        try:
            if user_id:
                st = get_status(user_id)
                remaining = max(0, get_limits(user_id) - st["daily_count"])
                if st["tier"] != "pro":
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
    await msg.reply_text("Полезное:", reply_markup=_pro_cta_keyboard())

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

        async def reply_photo(photo, caption=None):
            await query.message.reply_photo(photo=photo, caption=caption)

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

# --------------- Pro / Billing / Signals ---------------
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    u = update.effective_user
    st = get_status(u.id)
    cap = (
        f"Статус: {('PRO' if st['tier']=='pro' else 'FREE')}\n"
        f"Лимит/день: {get_limits(u.id)}\n"
        f"Израсходовано сегодня: {st['daily_count']}\n"
        f"Подписка до: {_fmt_until(st['sub_until'])}\n"
        f"Signal Mode: {'ON' if st['signal_enabled'] else 'OFF'}"
    )
    await msg.reply_text(cap, reply_markup=_category_keyboard())

async def pro_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    txt = (
        "💎 *Pro-подписка*\n"
        "Стоимость: 1 TON / месяц\n\n"
        "Преимущества:\n"
        "• до 10 прогнозов в день (вместо 3)\n"
        "• автоматический *Signal Mode* — бот присылает лучший прогноз дня (акции / крипта / форекс)\n\n"
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

    # считаем один раз на всех
    best_stocks = await _best_of_category(SUPPORTED_STOCKS, "stocks", app)
    best_crypto = await _best_of_category(SUPPORTED_CRYPTO, "crypto", app)
    best_fx     = await _best_of_category(SUPPORTED_FOREX, "forex", app)

    for uid in users:
        try:
            st = get_status(uid)
            if not st["signal_enabled"]:
                continue
            intro = "Дневной сигнал: лучшие возможности по категориям\n(оценка по потенциальной прибыли на $1,000)\n\n"
            await app.bot.send_message(chat_id=uid, text=intro)

            async def send_best(item, cat_name):
                if not item or item["profit"] <= 0:
                    await app.bot.send_message(chat_id=uid, text=f"{cat_name}: на сегодня сильных сигналов нет.")
                    return
                img = make_plot_image(item["df"], item["fcst"], item["ticker"], title_suffix=f"(Сигнал {cat_name})")
                cap = (f"{cat_name}: {item['ticker']}\n"
                       f"Модель: {item['best_name']} (RMSE={item['metrics'].get('rmse') if item['metrics'] else '—'})\n"
                       f"Оценка прибыли (на $1,000): ~ {item['profit']:.2f} USD\n\n"
                       f"{item['rec']}\n\n"
                       "⚠️ Не является инвестсоветом.")
                await app.bot.send_photo(chat_id=uid, photo=img, caption=cap[:1024])

            await send_best(best_stocks, "Акции")
            await send_best(best_crypto, "Крипта")
            await send_best(best_fx,     "Форекс")
        except Exception:
            continue

async def daily_signals_job(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    await daily_signals(app)

# --------------- Entrypoint ---------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")

    init_db()  # БД подписок
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
    app.add_error_handler(error_handler)


    # ежедневные «сигналы» через JobQueue (09:00 по Хельсинки)
    app.job_queue.run_daily(
        daily_signals_job,
        time=dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Moscow")),
        name="daily_signals",
    )

    print("Bot is running…")
    app.run_polling()

if __name__ == '__main__':
    main()
