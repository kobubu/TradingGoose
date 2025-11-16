# core/reminder_runner.py
import time
from typing import Callable, Tuple

from .reminders import due_for_day, mark_sent
from .recommend import generate_recommendations, UP_EMOJI, DOWN_EMOJI

# forecast_loader: Callable[[str], Tuple[DataFrame, float | None]]
# ожидание: forecast_loader(ticker) -> (fcst_df, model_rmse)


def _select_by_variant(markers, variant: str):
    """
    markers – список сделок из generate_recommendations (лонг + шорт).
    variant – 'best' | 'top3' | 'all'
    Возвращает подсписок markers, отсортированный по pnl.
    """
    if not markers:
        return []

    # сортируем по ожидаемой прибыли
    markers_sorted = sorted(markers, key=lambda m: m.get('pnl', 0.0), reverse=True)

    v = (variant or "").lower()
    if v == "best":
        return markers_sorted[:1]
    elif v == "top3":
        return markers_sorted[:3]
    elif v == "all":
        return markers_sorted
    else:
        # дефолт — как 'best'
        return markers_sorted[:1]


def _format_markers_message(ticker: str, markers):
    """
    Формирует текст сообщения для пользователя по списку markers.
    Ожидается, что marker имеет:
      - side: 'long' или 'short' (если нет – считаем 'long')
      - buy, sell: pandas.Timestamp
      - buy_price, sell_price, pnl: float
    """
    if not markers:
        return f"По тикеру {ticker} сейчас нет достаточно сильных сигналов по прогнозу."

    lines = [f"Напоминание по тикеру {ticker}:"]
    for m in markers:
        side = m.get("side", "long")
        buy = m["buy"]
        sell = m["sell"]
        buy_p = m["buy_price"]
        sell_p = m["sell_price"]
        pnl = m["pnl"]

        if side == "long":
            # покупаем на минимуме, продаём на максимуме
            line = (
                f"Лонг — покупать {DOWN_EMOJI}: {buy.date()} @ {buy_p:.2f} → "
                f"продавать {UP_EMOJI}: {sell.date()} @ {sell_p:.2f} "
                f"(ожидаемый доход ~ {pnl:.2f} USD)"
            )
        else:  # short
            # продаём на максимуме, выкупаем на минимуме
            line = (
                f"Шорт — продавать {UP_EMOJI}: {sell.date()} @ {sell_p:.2f} → "
                f"покупать обратно {DOWN_EMOJI}: {buy.date()} @ {buy_p:.2f} "
                f"(ожидаемый доход ~ {pnl:.2f} USD)"
            )

        lines.append(line)

    return "\n".join(lines)


def process_due_reminders(
    bot,
    capital_usd: float,
    forecast_loader: Callable[[str], Tuple[object, float | None]],
    window_sec: int = 300,
):
    """
    Обрабатывает напоминания, которые "созрели" во временном окне [now-window_sec, now+window_sec).

    bot             – объект Telegram-бота (у которого есть send_message(chat_id=..., text=...))
    capital_usd     – размер капитала, который передаётся в generate_recommendations
    forecast_loader – функция: ticker -> (fcst_df, model_rmse)
    window_sec      – ширина окна вокруг "сейчас"
    """
    now = int(time.time())
    start_ts = now - window_sec
    end_ts = now + window_sec

    # rows: [(id, user_id, ticker, variant, when_ts)]
    rows = due_for_day(start_ts, end_ts)

    for rem_id, user_id, ticker, variant, when_ts in rows:
        try:
            # 1) получаем прогноз по тикеру
            fcst_df, model_rmse = forecast_loader(ticker)

            # 2) получаем рекомендации (лонг + шорт внутри)
            summary, profit_est, markers = generate_recommendations(
                fcst_df, capital_usd, model_rmse
            )

            # 3) выбираем часть сигналов согласно variant (best/top3/all)
            selected_markers = _select_by_variant(markers, variant)

            # 4) формируем текст именно под напоминание
            text = _format_markers_message(ticker, selected_markers)

            # 5) отправляем пользователю
            bot.send_message(chat_id=user_id, text=text)

            # 6) помечаем напоминание как отправленное
            mark_sent(rem_id)

        except Exception as e:
            # на проде лучше логировать
            print(f"[reminder] error for rem_id={rem_id}, ticker={ticker}: {e}")
            # напоминание можно не помечать как sent, чтобы попробовать позже
