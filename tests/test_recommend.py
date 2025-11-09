import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_ticker_history
from core.recommend import generate_recommendations
from datetime import datetime



def test_recommendations_and_profit():
    idx = pd.bdate_range(start='2024-01-01', periods=30)
    y = 100 + np.sin(np.linspace(0, 6.0, 30))*5
    df = pd.DataFrame({'forecast': y}, index=idx)
    summary, profit, markers = generate_recommendations(df, 1000.0)
    assert isinstance(summary, str)
    assert isinstance(profit, float)