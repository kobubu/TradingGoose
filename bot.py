#bot.
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

# NEW: отдельный список для крипты (символы как у пользователя)
SUPPORTED_CRYPTO = MAIN_CRYPTO  # ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TRX","AVAX","LTC"]

# NEW: отдельный список для форекса (символы как у пользователя)
SUPPORTED_FOREX = MAIN_FOREX  # ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY"]

async def _run_forecast_for(ticker: str, amount: float, reply_text_fn, reply_photo_fn, user_id=None):
    """Shared forecast workflow used by command and callback handlers"""
    try:
        # NEW: резолвим тикер (например 'BTC' -> 'BTC-USD')
        resolved = resolve_user_ticker(ticker)
        await reply_text_fn(f"Загружаю данные для {resolved} и считаю прогноз…")
        df = load_ticker_history(resolved)
        if df is None or df.empty:
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.")
            return

        best, metrics, fcst_df = train_select_and_forecast(df, ticker=resolved)
        rec_summary, profit_est, markers = generate_recommendations(
            fcst_df, amount, model_rmse=metrics.get('rmse') if metrics else None
        )
        img_buf = make_plot_image(df, fcst_df, resolved, markers=markers)

        try:
            from datetime import datetime
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            pdf_path = os.path.join(art_dir, f"{resolved}_forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
            export_plot_pdf(df, fcst_df, resolved, pdf_path)
        except Exception:
            pass

        delta_pct = ((fcst_df['forecast'].iloc[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100.0
        msg = (
            f"Тикер: {resolved}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены к последнему дню: {delta_pct:+.2f}%\n\n"
            f"{rec_summary}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_est:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )

        if len(msg) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_buf, caption=msg)
        else:
            await reply_photo_fn(photo=img_buf)
            for i in range(0, len(msg), TEXT_MAX):
                await reply_text_fn(msg[i:i+TEXT_MAX])

        log_request(
            user_id=user_id,
            ticker=resolved,
            amount=amount,
            best_model=best['name'],
            metric_name='RMSE',
            metric_value=metrics['rmse'],
            est_profit=profit_est,
        )
    except Exception as e:
        await reply_text_fn(f"Ошибка: {e}")

HELP_TEXT = (
    "Привет! Я бот прогноза акций и криптовалют.\n\n"
    "Команды:\n"
    "/forecast <TICKER> — пример: /forecast AAPL или /forecast BTC\n"
    "/stocks — быстрые кнопки с популярными акциями\n"
    "/crypto — быстрые кнопки с топ-10 криптовалют\n\n"
    "Я загружу котировки за ~2 года, обучу несколько моделей и пришлю прогноз на 30 дней,\n"
    "рекомендации по дням покупки/продажи и оценку условной прибыли.\n\n"
    "⚠️ Учебный проект, не является инвестсоветом."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(HELP_TEXT)

async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text("Использование: /forecast <TICKER>\nНапример: /forecast AAPL или /forecast BTC")
            return

        try:
            print("DEBUG: received message_text=", update.message.text if update.message else None)
            print("DEBUG: context.args=", context.args)
        except Exception:
            pass

        user_ticker = context.args[0].upper().strip()
        amount = DEFAULT_AMOUNT

        # резолвим здесь, но текст прогресса сформирует _run_forecast_for
        await _run_forecast_for(
            ticker=user_ticker,
            amount=amount,
            reply_text_fn=update.message.reply_text,
            reply_photo_fn=update.message.reply_photo,
            user_id=update.effective_user.id if update.effective_user else None
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


# --- NEW: /stocks и /crypto с инлайн-кнопками ---
async def stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons, row = [], []
    for t in SUPPORTED_TICKERS:
        row.append(InlineKeyboardButton(t, callback_data=f"forecast:{t}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    await update.message.reply_text(
        "Выберите акцию (нажмите кнопку):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )

async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons, row = [], []
    for c in SUPPORTED_CRYPTO:
        row.append(InlineKeyboardButton(c, callback_data=f"forecast:{c}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    await update.message.reply_text(
        "Выберите криптовалюту (нажмите кнопку):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )

async def forex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons, row = [], []
    for f in SUPPORTED_FOREX:
        row.append(InlineKeyboardButton(f, callback_data=f"forecast:{f}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    await update.message.reply_text(
        "Выберите валютную пару (нажмите кнопку):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
# -------------------------------------------------

async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сохраняем оригинальную команду, но теперь лучше использовать /stocks и /crypto"""
    await update.message.reply_text("Списки обновлены. Используйте /stocks (акции) и /crypto (криптовалюты).")

async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").strip()

    if data.startswith("forecast:"):
        ticker = data.split(":", 1)[1].strip().upper()

        # Разрешаем и акции, и крипту
        if (SUPPORTED_TICKERS and ticker not in SUPPORTED_TICKERS) and (SUPPORTED_CRYPTO and ticker not in SUPPORTED_CRYPTO) and (SUPPORTED_FOREX and ticker not in SUPPORTED_FOREX):
            # всё равно позволим вручную: пользователь мог ввести другой валидный тикер
            pass

        amount = DEFAULT_AMOUNT

        async def reply_text(text):
            await query.message.reply_text(text)

        async def reply_photo(photo, caption=None):
            await query.message.reply_photo(photo=photo, caption=caption)

        user_id = query.from_user.id if query.from_user else None
        await _run_forecast_for(ticker=ticker, amount=amount,
                                reply_text_fn=reply_text, reply_photo_fn=reply_photo,
                                user_id=user_id)

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("stocks", stocks))
    app.add_handler(CommandHandler("forex", forex))    # NEW
    app.add_handler(CommandHandler("crypto", crypto))   # NEW
    app.add_handler(CommandHandler("tickers", tickers)) # legacy-совместимость
    app.add_handler(CallbackQueryHandler(_on_callback))
    print("Bot is running…")
    app.run_polling()

if __name__ == '__main__':
    main()