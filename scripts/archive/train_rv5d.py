"""Train DL range model with RV 5d normalization window.

Monkey-patches build_targets to use rolling(5) instead of rolling(10),
then saves output as dl_range_v2_oos_rv5d.parquet and restores the rv10d version.
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import src.models.train_dl_range as train_mod


def build_targets_rv5(gld):
    close, high, low = gld["Close"], gld["High"], gld["Low"]
    max_high_5d = high.shift(-1).rolling(5).max().shift(-4)
    min_low_5d = low.shift(-1).rolling(5).min().shift(-4)
    upper_pct = (max_high_5d / close - 1) * 100
    lower_pct = (min_low_5d / close - 1) * 100
    log_ret = np.log(close / close.shift(1))
    rv_scale = log_ret.rolling(5).std() * np.sqrt(5) * 100
    return upper_pct, lower_pct, rv_scale


train_mod.build_targets = build_targets_rv5

print("=" * 70)
print("  Training with RV 5d normalization (rolling(5))")
print("=" * 70)

train_mod.main()

# Save rv5d output and restore rv10d as default
shutil.copy('data/models/dl_range_v2_oos.parquet',
            'data/models/dl_range_v2_oos_rv5d.parquet')
shutil.copy('data/models/dl_range_v2_oos_rv10d.parquet',
            'data/models/dl_range_v2_oos.parquet')
print(f"\nSaved: data/models/dl_range_v2_oos_rv5d.parquet")
print(f"Restored: dl_range_v2_oos.parquet (RV 10d)")
