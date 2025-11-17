#helper to check worst models
import pandas as pd
from pathlib import Path

STATS_PATH = Path("artifacts/models_stats.csv")

df = pd.read_csv(STATS_PATH)
# на всякий случай
df = df.dropna(subset=["config_id"])

agg = (
    df.groupby("config_id")
      .agg(
          n_wins=("ticker", "count"),
          mean_rmse=("rmse", "mean"),
          median_rmse=("rmse", "median"),
          models=("model_name", lambda x: ",".join(sorted(set(x)))),
      )
      .sort_values("median_rmse")
)

print(agg.head(20))
print("Worst configs:")
print(agg.sort_values("median_rmse", ascending=False).head(20))