import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_ticker_history
from core.forecast import train_select_and_forecast
from datetime import datetime, timedelta

def _synthetic_series(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 8*np.pi, n)
    y = 100 + 5*np.sin(x) + rng.normal(0, 0.8, n)
    idx = pd.bdate_range(end=datetime.utcnow().date(), periods=n)
    return pd.DataFrame({"Close": y}, index=idx)

def test_model_selection_and_forecast_runs():
    df = _synthetic_series()
    best, metrics, fcst_df = train_select_and_forecast(df)
    assert "name" in best and isinstance(best["name"], str)
    assert "rmse" in metrics and metrics["rmse"] >= 0
    assert len(fcst_df) == 30
    assert "forecast" in fcst_df.columns