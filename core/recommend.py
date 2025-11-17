"""recommend.py Core module for the Telegram stock forecast bot."""
import os

MIN_PROFIT_PCT = float(os.getenv('MIN_PROFIT_PCT', '0.015'))
MIN_PROFIT_USD = float(os.getenv('MIN_PROFIT_USD', '3.0'))
RMSE_MULTIPLIER = float(os.getenv('RMSE_MULTIPLIER', '0.7'))

UP_EMOJI = "🔴📉"    # использовано как "продажа"
DOWN_EMOJI = "🟢📈"  # использовано как "покупка"


def _local_extrema(series):
    """Находит локальные минимумы и максимумы в временном ряду"""
    idx = series.index
    vals = series.values
    mins, maxs = [], []
    for i in range(1, len(vals)-1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            mins.append(idx[i])
        if vals[i] > vals[i-1] and vals[i] > vals[i+1]:
            maxs.append(idx[i])
    return mins, maxs


def _build_long_trades(s, mins, maxs, capital_usd, model_rmse):
    """Сценарий: покупка на локальных минимумах, продажа на следующих максимумах"""
    trades = []
    local_maxs = list(maxs)  # копия, чтобы не портить исходный список
    i = 0
    while i < len(mins):
        buy_day = mins[i]
        sell_candidates = [m for m in local_maxs if m > buy_day]
        if not sell_candidates:
            break
        sell_day = sell_candidates[0]
        trades.append((buy_day, sell_day))
        # удаляем все максимумы до выбранного
        local_maxs = [m for m in local_maxs if m > sell_day]
        i += 1

    profit = 0.0
    lines = []
    markers = []

    # RMSE-порог
    rmse_req = 0.0
    try:
        if model_rmse is not None:
            rmse_req = float(model_rmse) * float(RMSE_MULTIPLIER)
    except Exception:
        rmse_req = 0.0

    min_required = max(MIN_PROFIT_USD, capital_usd * MIN_PROFIT_PCT, rmse_req)

    for buy, sell in trades:
        buy_price = float(s.loc[buy])
        sell_price = float(s.loc[sell])
        if sell_price <= buy_price:
            continue

        shares = capital_usd / buy_price
        pnl = shares * (sell_price - buy_price)
        if pnl < min_required:
            continue

        profit += pnl
        lines.append(
            f"Лонг — покупать {DOWN_EMOJI}: {buy.date()} @ {buy_price:.2f} → "
            f"продавать {UP_EMOJI}: {sell.date()} @ {sell_price:.2f} "
            f"(доход ~ {pnl:.2f} USD)"
        )
        markers.append({
            'side': 'long',
            'buy': buy,
            'sell': sell,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'pnl': pnl,
        })

    return profit, lines, markers


def _build_short_trades(s, mins, maxs, capital_usd, model_rmse):
    """Сценарий: шорт — продажа на локальных максимумах, закрытие на следующих минимумах"""
    trades = []
    local_mins = list(mins)
    i = 0
    while i < len(maxs):
        sell_day = maxs[i]  # открываем шорт на максимуме
        cover_candidates = [m for m in local_mins if m > sell_day]
        if not cover_candidates:
            break
        cover_day = cover_candidates[0]  # закрываем шорт на ближайшем минимуме
        trades.append((sell_day, cover_day))
        local_mins = [m for m in local_mins if m > cover_day]
        i += 1

    profit = 0.0
    lines = []
    markers = []

    # RMSE-порог
    rmse_req = 0.0
    try:
        if model_rmse is not None:
            rmse_req = float(model_rmse) * float(RMSE_MULTIPLIER)
    except Exception:
        rmse_req = 0.0

    min_required = max(MIN_PROFIT_USD, capital_usd * MIN_PROFIT_PCT, rmse_req)

    for sell, cover in trades:
        sell_price = float(s.loc[sell])
        cover_price = float(s.loc[cover])
        if cover_price >= sell_price:
            continue  # шорт не имеет смысла, если цена не падает

        # размер позиции: сколько акций можно "продать" на данный капитал
        shares = capital_usd / sell_price
        pnl = shares * (sell_price - cover_price)
        if pnl < min_required:
            continue

        profit += pnl
        lines.append(
            f"Шорт — продавать {UP_EMOJI}: {sell.date()} @ {sell_price:.2f} → "
            f"покупать обратно {DOWN_EMOJI}: {cover.date()} @ {cover_price:.2f} "
            f"(доход ~ {pnl:.2f} USD)"
        )
        markers.append({
            'side': 'short',
            'sell': sell,
            'buy': cover,  # покупка для закрытия шорта
            'sell_price': sell_price,
            'buy_price': cover_price,
            'pnl': pnl,
        })

    return profit, lines, markers


def generate_recommendations(fcst_df, capital_usd, model_rmse=None):
    """
    Генерирует торговые рекомендации на основе прогноза цен.
    """
    s = fcst_df['forecast']
    mins, maxs = _local_extrema(s)

    # Сценарий 1: только лонг
    long_profit, long_lines, long_markers = _build_long_trades(
        s, mins, maxs, capital_usd, model_rmse
    )

    # Сценарий 2: только шорт
    short_profit, short_lines, short_markers = _build_short_trades(
        s, mins, maxs, capital_usd, model_rmse
    )

    all_markers = long_markers + short_markers

    # Общий "наклон" прогноза — от первой до последней точки
    try:
        first_price = float(s.iloc[0])
        last_price = float(s.iloc[-1])
        if first_price > 0:
            delta_pct = (last_price - first_price) / first_price * 100.0
        else:
            delta_pct = 0.0
    except Exception:
        delta_pct = 0.0

    if not long_lines and not short_lines:
        # Нет сделок, но тренд может быть сильным
        if abs(delta_pct) >= TREND_ALERT_PCT:
            direction = "рост" if delta_pct > 0 else "снижение"
            summary = (
                f"Модель ожидает выраженный {direction} цены (~{delta_pct:+.2f}% за период прогноза), "
                "но локальные точки входа/выхода по стратегии (минимумы/максимумы) не проходят "
                "фильтры по ожидаемой прибыли и надёжности (RMSE).\n\n"
                "В такой ситуации разумно наблюдать за инструментом и оценивать риски вручную, "
                "а не входить по формальным сигналам."
            )
        else:
            summary = (
                "По прогнозу нет достаточно сильных локальных сигналов ни для лонга, "
                "ни для шорта (мелкие сигналы были отфильтрованы по порогу прибыли/RMSE). "
                "Рекомендуется наблюдать за динамикой и рисками."
            )
        est_profit = 0.0
    else:
        parts = []
        if long_lines:
            parts.append(
                "Сценарий 1 — только лонг (покупка на минимумах, продажа на максимумах):\n"
                + "\n".join(long_lines)
                + f"\nИтого ожидаемый доход (лонг): ~{long_profit:.2f} USD"
            )
        if short_lines:
            parts.append(
                "Сценарий 2 — только шорт (продажа на максимумах, закрытие на минимумах):\n"
                + "\n".join(short_lines)
                + f"\nИтого ожидаемый доход (шорт): ~{short_profit:.2f} USD"
            )

        summary = "\n\n".join(parts)
        est_profit = float(max(long_profit, short_profit))

    return summary, est_profit, all_markers