# bot.py
import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes

from core.data import load_ticker_history, resolve_user_ticker, MAIN_CRYPTO, MAIN_FOREX
from core.forecast import export_plot_pdf, make_plot_image, train_select_and_forecast
from core.logging_utils import log_request
from core.recommend import generate_recommendations

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

DEFAULT_AMOUNT = 1000.0

CAPTION_MAX = 1024
TEXT_MAX = 4096

# БОЛЬШЕ АКЦИЙ (≥10). Можно менять на свои:
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

# Отдельные списки для крипты/форекса
SUPPORTED_CRYPTO = MAIN_CRYPTO   # ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TRX","AVAX","LTC"]
SUPPORTED_FOREX  = MAIN_FOREX    # ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY"]

HELP_TEXT = (
    "Привет! Я бот прогноза акций, криптовалют и форекса.\n\n"
    "Команды:\n"
    "/forecast <TICKER> — пример: /forecast AAPL или /forecast BTC\n"
    "/stocks — быстрые кнопки с популярными акциями\n"
    "/crypto — быстрые кнопки с топ-10 криптовалют\n"
    "/forex — быстрые кнопки с основными валютными парами\n\n"
    "Я загружу котировки за ~2 года, обучу несколько моделей и пришлю прогноз на 30 дней,\n"
    "плюс 3 варианта: Лучшая модель, Ансамбль топ-3, Ансамбль всех. Также высылаю 3 набора рекомендаций.\n\n"
    "⚠️ Учебный проект, не является инвестсоветом."
)

# ---------------- UI helpers ----------------
def _category_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 Акции", callback_data="menu:stocks"),
            InlineKeyboardButton("₿ Крипта", callback_data="menu:crypto"),
            InlineKeyboardButton("💱 Форекс", callback_data="menu:forex"),
        ]
    ])

def _build_list_rows(items, per_row=3):
    """Возвращает СПИСОК строк (а не InlineKeyboardMarkup), чтобы можно было добавить 'Назад'."""
    rows, row = [], []
    for it in items:
        row.append(InlineKeyboardButton(it, callback_data=f"forecast:{it}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows

# --------------- Forecast pipeline ---------------
async def _run_forecast_for(ticker: str, amount: float, reply_text_fn, reply_photo_fn, user_id=None):
    """Общий пайплайн прогноза для команд и callback-кнопок."""
    try:
        # резолвим тикер (например 'BTC' -> 'BTC-USD')
        resolved = resolve_user_ticker(ticker)
        await reply_text_fn(f"Загружаю данные для {resolved} и считаю прогноз…")
        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.", reply_markup=_category_keyboard())
            return

        # получаем 3 прогноза: лучшая, среднее по всем, среднее по топ-3
        best, metrics, fcst_best_df, fcst_avg_all_df, fcst_avg_top3_df = train_select_and_forecast(
            df, ticker=resolved
        )

        # рекомендации (RMSE ансамблей отдельно не считаем — передаём RMSE лучшей)
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
            from datetime import datetime
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
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
        # 1/3: лучшая модель
        if len(cap_best) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_best, caption=cap_best)
        else:
            await reply_photo_fn(photo=img_best)
            for i in range(0, len(cap_best), TEXT_MAX):
                await reply_text_fn(cap_best[i:i + TEXT_MAX])

        # 2/3: ансамбль топ-3
        if len(cap_t3) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_t3, caption=cap_t3)
        else:
            await reply_photo_fn(photo=img_t3)
            for i in range(0, len(cap_t3), TEXT_MAX):
                await reply_text_fn(cap_t3[i:i + TEXT_MAX])

        # 3/3: ансамбль всех
        if len(cap_all) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_all, caption=cap_all)
        else:
            await reply_photo_fn(photo=img_all)
            for i in range(0, len(cap_all), TEXT_MAX):
                await reply_text_fn(cap_all[i:i + TEXT_MAX])

        # показать меню внизу
        await reply_text_fn("Быстрый выбор категории:", reply_markup=_category_keyboard())

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
    except Exception as e:
        await reply_text_fn(f"Ошибка: {e}", reply_markup=_category_keyboard())

# --------------- Command handlers ---------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # единая точка входа — сразу показываем меню
    await update.message.reply_text(HELP_TEXT, reply_markup=_category_keyboard())

async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text(
                "Использование: /forecast <TICKER>\nНапример: /forecast AAPL или /forecast BTC",
                reply_markup=_category_keyboard(),
            )
            return

        try:
            print("DEBUG: received message_text=", update.message.text if update.message else None)
            print("DEBUG: context.args=", context.args)
        except Exception:
            pass

        user_ticker = context.args[0].upper().strip()
        amount = DEFAULT_AMOUNT

        await _run_forecast_for(
            ticker=user_ticker,
            amount=amount,
            reply_text_fn=update.message.reply_text,
            reply_photo_fn=update.message.reply_photo,
            user_id=update.effective_user.id if update.effective_user else None
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}", reply_markup=_category_keyboard())

# — списки через callback или команду — используем effective_message
async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_TICKERS, per_row=3)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите акцию:", reply_markup=InlineKeyboardMarkup(rows))

async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_CRYPTO, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(rows))

async def forex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = _build_list_rows(SUPPORTED_FOREX, per_row=4)
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="menu:root")])
    msg = update.effective_message
    await msg.reply_text("Выберите валютную пару:", reply_markup=InlineKeyboardMarkup(rows))

async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Списки обновлены. Используйте /stocks (акции), /crypto (криптовалюты) и /forex (валютные пары).",
        reply_markup=_category_keyboard(),
    )

# --------------- Callback handler ---------------
async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").strip()

    # запускаем прогноз
    if data.startswith("forecast:"):
        ticker = data.split(":", 1)[1].strip().upper()

        # Разрешаем и акции, и крипту, и форекс (плюс ручные тикеры)
        if (SUPPORTED_TICKERS and ticker not in SUPPORTED_TICKERS) and \
           (SUPPORTED_CRYPTO and ticker not in SUPPORTED_CRYPTO) and \
           (SUPPORTED_FOREX and ticker not in SUPPORTED_FOREX):
            pass

        amount = DEFAULT_AMOUNT

        async def reply_text(text, **kwargs):
            await query.message.reply_text(text, **kwargs)

        async def reply_photo(photo, caption=None):
            await query.message.reply_photo(photo=photo, caption=caption)

        user_id = query.from_user.id if query.from_user else None
        await _run_forecast_for(
            ticker=ticker,
            amount=amount,
            reply_text_fn=reply_text,
            reply_photo_fn=reply_photo,
            user_id=user_id
        )
        return

    # открываем меню списков
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

# --------------- Entrypoint ---------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("stocks", stocks))
    app.add_handler(CommandHandler("crypto", crypto))
    app.add_handler(CommandHandler("forex", forex))
    app.add_handler(CommandHandler("tickers", tickers))  # legacy
    app.add_handler(CallbackQueryHandler(_on_callback))
    print("Bot is running…")
    app.run_polling()

if __name__ == '__main__':
    main()
