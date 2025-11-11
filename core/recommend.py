"""recommend.py Core module for the Telegram stock forecast bot."""
import os

MIN_PROFIT_USD = float(os.getenv('MIN_PROFIT_USD', '0.5'))
MIN_PROFIT_PCT = float(os.getenv('MIN_PROFIT_PCT', '0.001'))
RMSE_MULTIPLIER = float(os.getenv('RMSE_MULTIPLIER', '0.5'))

UP_EMOJI = "🔴📉"
DOWN_EMOJI = "🟢📈"

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

def generate_recommendations(fcst_df, capital_usd, model_rmse=None):
    """
    Генерирует торговые рекомендации на основе прогноза цен
    
    Return (summary_text, profit_est_usd, markers)

    markers: list of dicts with keys 'buy','sell','buy_price','sell_price','pnl'
    """
    s = fcst_df['forecast']
    mins, maxs = _local_extrema(s)

    trades = []
    i = 0
    while i < len(mins):
        buy_day = mins[i]
        sell_candidates = [m for m in maxs if m > buy_day]
        if not sell_candidates:
            break
        sell_day = sell_candidates[0]
        trades.append((buy_day, sell_day))
        maxs = [m for m in maxs if m > sell_day]
        i += 1

    profit = 0.0
    lines = []
    markers = []
    for buy, sell in trades:
        buy_price = s.loc[buy]
        sell_price = s.loc[sell]
        if sell_price <= buy_price:
            continue
        shares = capital_usd / float(buy_price)
        pnl = shares * (float(sell_price) - float(buy_price))

        rmse_req = 0.0
        try:
            if model_rmse is not None:
                rmse_req = float(model_rmse) * float(RMSE_MULTIPLIER)
        except Exception:
            rmse_req = 0.0

        min_required = max(MIN_PROFIT_USD, capital_usd * MIN_PROFIT_PCT, rmse_req)
        if pnl < min_required:
            continue

        profit += pnl
        lines.append(
            f"Покупать {DOWN_EMOJI}: {buy.date()} @ {buy_price:.2f} → "
            f"Продавать {UP_EMOJI}: {sell.date()} @ {sell_price:.2f} "
            f"(доход ~ {pnl:.2f} USD)"
        )
        markers.append({
            'buy': buy, 'sell': sell,
            'buy_price': float(buy_price), 'sell_price': float(sell_price), 'pnl': float(pnl)
        })

    if not lines:
        summary = "По прогнозу чётких локальных точек для покупки/продажи немного (мелкие сигналы были отфильтрованы). Рекомендуется наблюдать за динамикой и рисками."
    else:
        summary = "Рекомендации (локальные минимумы/максимумы):\n" + "\n".join(lines)

    return summary, float(profit), markers