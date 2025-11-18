# core/plot_utils.py
import io
import matplotlib
matplotlib.use("Agg")  # безопасный backend без GUI
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List


def make_plot_image(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    markers: Optional[List] = None,  # оставлено для совместимости, пока не используем
    title_suffix: str = "",
) -> io.BytesIO:
    """
    Рисует график истории и 30-дневного прогноза и возвращает PNG в виде BytesIO.
    """
    plt.figure(figsize=(10, 5))

    # История и прогноз
    plt.plot(history_df.index, history_df["Close"], label="History")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")

    # Аккуратно соединяем последнюю точку истории с первой прогнозной,
    # чтобы визуально не было "разрыва".
    try:
        if not history_df.empty and not forecast_df.empty:
            plt.plot(
                [history_df.index[-1], forecast_df.index[0]],
                [history_df["Close"].iloc[-1], forecast_df["forecast"].iloc[0]],
                linestyle=":",
                linewidth=1.0,
            )
    except Exception:
        pass

    title = f"{ticker}: History & 30-Day Forecast"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)

    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def export_plot_pdf(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    out_path: str,
) -> None:
    """
    То же самое, но в PDF на диск.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history_df.index, history_df["Close"], label="History")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")

    # Соединяем последнюю точку истории с первой прогнозной
    try:
        if not history_df.empty and not forecast_df.empty:
            plt.plot(
                [history_df.index[-1], forecast_df.index[0]],
                [history_df["Close"].iloc[-1], forecast_df["forecast"].iloc[0]],
                linestyle=":",
                linewidth=1.0,
            )
    except Exception:
        pass

    plt.title(f"{ticker}: History & 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close()
