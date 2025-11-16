# ui.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

HELP_TEXT = (
    "Привет! Я бот прогноза акций, криптовалют и форекса.\n\n"
    "Обучаю ML-модели, которые строят предсказания\n\n"
    "Команды:\n"
    "/forecast <TICKER> — пример: /forecast AAPL или /forecast BTC\n"
    "/history <TICKER> — последний сохранённый прогноз из кэша\n"
    "/stocks — быстрый список акций\n"
    "/crypto — топ-10 криптовалют\n"
    "/forex — основные валютные пары\n"
    "/status — ваш тариф и лимиты\n"
    "/pro — про подписку, /buy — оплата, /signal_on, signal_off — включить, выключить сигналы\n\n"
    "Бесплатно: 3 прогноза/день.\n"
    "Pro (1 TON/мес): 10 прогнозов/день + ежедневный «Signal Mode».\n\n"
    "⚠️ Не является инвестсоветом."
)



def main_menu_keyboard() -> InlineKeyboardMarkup:
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


def category_keyboard() -> InlineKeyboardMarkup:
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


def pro_cta_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("💎 Pro", callback_data="menu:pro"),
            InlineKeyboardButton("💳 Купить", callback_data="menu:buy"),
            InlineKeyboardButton("ℹ️ Статус", callback_data="menu:status"),
        ]]
    )


def build_list_rows(items, per_row=3):
    rows, row = [], []
    for it in items:
        row.append(InlineKeyboardButton(it, callback_data=f"forecast:{it}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows
