"""
GoldDash 数据下载与特征构建

一键完成:
  1. 下载市场数据 (yfinance)
  2. 下载宏观数据 (FRED API)
  3. 下载波动率数据 (CBOE GVZ)
  4. 构建特征矩阵
  5. 训练 DL Range 模型 (可选)

用法:
    # 首次使用 — 需要 FRED API Key (免费申请: https://fred.stlouisfed.org/docs/api/api_key.html)
    python scripts/setup_data.py --fred-key YOUR_API_KEY

    # 日常更新
    python scripts/setup_data.py

    # 跳过模型训练 (仅更新数据)
    python scripts/setup_data.py --no-train

依赖:
    pip install yfinance fredapi requests pandas numpy torch scikit-learn
"""
import os
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

DATA_ROOT = os.path.join(PROJECT_DIR, "data")
RAW_MARKET = os.path.join(DATA_ROOT, "raw", "market")
RAW_MACRO = os.path.join(DATA_ROOT, "raw", "macro")
RAW_VOL = os.path.join(DATA_ROOT, "raw", "volatility")
RAW_COT = os.path.join(DATA_ROOT, "raw", "cot")
PROCESSED = os.path.join(DATA_ROOT, "processed")
MODELS = os.path.join(DATA_ROOT, "models")

START_DATE = "2004-11-18"  # GLD 上市日

# ── Yahoo Finance 下载列表 ──
YAHOO_TICKERS = {
    "gld": "GLD",
    "gold_futures": "GC=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "crude_oil": "CL=F",
    "copper": "HG=F",
    "silver": "SI=F",
    "us10y": "^TNX",
    "us2y": "^IRX",
    "usdcny": "CNY=X",
}

# ── FRED 下载列表 ──
FRED_SERIES = {
    "real_yield_10y": "DFII10",
    "real_yield_5y": "DFII5",
    "breakeven_10y": "T10YIE",
    "fed_funds_rate": "DFF",
    "federal_debt": "GFDEBTN",
    "cpi": "CPIAUCSL",
    "m2": "M2SL",
    "tw_usd": "DTWEXBGS",
}

GVZ_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv"


def ensure_dirs():
    for d in [RAW_MARKET, RAW_MACRO, RAW_VOL, RAW_COT, PROCESSED, MODELS]:
        os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════
# Step 1: 市场数据
# ══════════════════════════════════════════════════════════
def download_market_data():
    """通过 yfinance 下载市场行情."""
    import yfinance as yf
    print("\n[1/4] 下载市场数据 (yfinance)...")

    for name, ticker in YAHOO_TICKERS.items():
        out = os.path.join(RAW_MARKET, f"{name}.csv")
        try:
            df = yf.download(ticker, start=START_DATE, progress=False,
                             auto_adjust=True)
            if len(df) == 0:
                print(f"  {name} ({ticker}): 无数据, 跳过")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]
            df.index.name = "Date"
            df.to_csv(out)
            print(f"  {name}: {len(df)} rows "
                  f"({df.index[0].date()} ~ {df.index[-1].date()})")
        except Exception as e:
            print(f"  {name} ({ticker}): 下载失败 — {e}")


# ══════════════════════════════════════════════════════════
# Step 2: 宏观数据
# ══════════════════════════════════════════════════════════
def download_macro_data(fred_key: str = None):
    """通过 FRED API 下载宏观经济数据."""
    print("\n[2/4] 下载宏观数据 (FRED)...")

    key_file = os.path.join(PROJECT_DIR, ".fred_key")
    if fred_key:
        with open(key_file, "w") as f:
            f.write(fred_key)
        print(f"  FRED API Key 已保存到 {key_file}")
    elif os.path.exists(key_file):
        with open(key_file) as f:
            fred_key = f.read().strip()
    else:
        print("  跳过: 未提供 FRED API Key")
        print("  → 申请 (免费): https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  → 使用: python scripts/setup_data.py --fred-key YOUR_KEY")
        return

    try:
        from fredapi import Fred
    except ImportError:
        print("  跳过: fredapi 未安装 (pip install fredapi)")
        return

    fred = Fred(api_key=fred_key)
    all_series = {}

    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE)
            s.name = name
            s.index = pd.to_datetime(s.index)
            s.to_csv(os.path.join(RAW_MACRO, f"{name}.csv"), header=True)
            all_series[name] = s
            print(f"  {name} ({sid}): {len(s)} rows")
        except Exception as e:
            print(f"  {name} ({sid}): 失败 — {e}")

    # 合并为 macro_panel
    if all_series:
        panel = pd.DataFrame(all_series)
        panel.index.name = "Date"
        panel.to_csv(os.path.join(RAW_MACRO, "macro_panel.csv"))
        print(f"  macro_panel: {panel.shape}")


# ══════════════════════════════════════════════════════════
# Step 3: 波动率数据
# ══════════════════════════════════════════════════════════
def download_vol_data():
    """下载 GVZ (CBOE Gold Volatility Index)."""
    print("\n[3/4] 下载波动率数据 (CBOE GVZ)...")
    import requests

    try:
        resp = requests.get(GVZ_URL, timeout=30)
        resp.raise_for_status()
        out = os.path.join(RAW_VOL, "gvz.csv")
        # CBOE CSV 格式: DATE, OPEN, HIGH, LOW, CLOSE
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # 标准化列名
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if "date" in cl:
                col_map[c] = "Date"
            elif "close" in cl:
                col_map[c] = "Close"
            elif "open" in cl:
                col_map[c] = "Open"
            elif "high" in cl:
                col_map[c] = "High"
            elif "low" in cl:
                col_map[c] = "Low"
        df = df.rename(columns=col_map)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        df.to_csv(out)
        print(f"  GVZ: {len(df)} rows "
              f"({df.index[0].date()} ~ {df.index[-1].date()})")
    except Exception as e:
        print(f"  GVZ: 下载失败 — {e}")


# ══════════════════════════════════════════════════════════
# Step 4: 特征构建
# ══════════════════════════════════════════════════════════
def build_features():
    """从原始数据构建特征矩阵."""
    print("\n[4/4] 构建特征矩阵...")

    # 读取 GLD
    gld_path = os.path.join(RAW_MARKET, "gld.csv")
    if not os.path.exists(gld_path):
        print("  错误: GLD 数据不存在, 请先运行数据下载")
        return None
    gld = pd.read_csv(gld_path, index_col=0, parse_dates=True)
    close = gld["Close"]
    high = gld["High"]
    low = gld["Low"]
    volume = gld["Volume"] if "Volume" in gld.columns else pd.Series(0, index=gld.index)

    features = pd.DataFrame(index=gld.index)

    # ── 收益率 ──
    for w in [1, 2, 3, 5, 10, 20, 60]:
        features[f"ret_{w}d"] = close.pct_change(w)

    # ── 技术指标 ──
    # RSI
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        features[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    features["macd_hist"] = macd_line - macd_signal

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bb_position"] = (close - sma20) / (2 * std20).replace(0, np.nan)
    features["bb_width"] = (4 * std20) / sma20

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    features["stoch_k_14"] = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
    features["stoch_d_14"] = features["stoch_k_14"].rolling(3).mean()

    # SMA 偏离
    for w in [5, 20, 60, 120]:
        sma = close.rolling(w).mean()
        features[f"close_to_sma_{w}"] = (close - sma) / sma

    # SMA 斜率
    features["sma_20_slope"] = sma20.pct_change(5)
    sma60 = close.rolling(60).mean()
    features["sma_60_slope"] = sma60.pct_change(10)

    # MA alignment
    sma5 = close.rolling(5).mean()
    features["ma_alignment"] = ((sma5 > sma20).astype(int) +
                                 (sma20 > sma60).astype(int) - 1)

    # ATR
    tr = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    features["atr_14"] = atr14
    features["atr_14_pct"] = atr14 / close

    # Daily range
    features["daily_range_pct"] = (high - low) / close

    # HV
    log_ret = np.log(close / close.shift(1))
    for w in [5, 60]:
        features[f"hv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
    features["hv_5d_change"] = features["hv_5d"].pct_change(5)

    # RV
    for w in [5, 10, 20]:
        features[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100

    # Volume
    if volume.sum() > 0:
        features["vol_ratio_5d"] = volume / volume.rolling(5).mean()
        features["vol_ratio_20d"] = volume / volume.rolling(20).mean()
        features["vol_change_1d"] = volume.pct_change()

    # Gap
    features["gap_pct"] = (gld["Open"] - close.shift(1)) / close.shift(1)

    # ── 跨市场 ──
    cross_pairs = {
        "gold_futures": ("gc_gld_ratio", close),
        "dxy": ("dxy", None),
        "vix": ("vix", None),
        "copper": ("copper", None),
        "silver": ("silver", None),
        "us10y": ("us10y", None),
        "crude_oil": ("crude", None),
    }
    for fname, (prefix, denom) in cross_pairs.items():
        fpath = os.path.join(RAW_MARKET, f"{fname}.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        s = df["Close"].reindex(gld.index).ffill()

        if fname == "gold_futures":
            ratio = s / close
            features["gc_gld_ratio"] = ratio
            features["gc_gld_ratio_zscore"] = (
                (ratio - ratio.rolling(60).mean()) /
                ratio.rolling(60).std().replace(0, np.nan))
        elif fname == "dxy":
            features["dxy_ret_1d"] = s.pct_change(1)
            features["dxy_ret_5d"] = s.pct_change(5)
        elif fname == "vix":
            features["vix_level"] = s
            features["vix_ret_1d"] = s.pct_change(1)
            features["vix_sma20_dev"] = (
                s - s.rolling(20).mean()) / s.rolling(20).mean()
        elif fname == "copper":
            features["copper_gold_ratio"] = s / close
            features["copper_gold_ratio_change"] = (s / close).pct_change(20)
        elif fname == "silver":
            features["gold_silver_ratio"] = close / s.replace(0, np.nan)
        elif fname == "us10y":
            features["us10y_level"] = s
            features["us10y_change_5d"] = s.diff(5)
        elif fname == "crude_oil":
            features["crude_ret_5d"] = s.pct_change(5)

    # ── 宏观数据 ──
    macro_path = os.path.join(RAW_MACRO, "macro_panel.csv")
    if os.path.exists(macro_path):
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro = macro.reindex(gld.index).ffill()

        if "real_yield_10y" in macro.columns:
            features["real_yield_10y"] = macro["real_yield_10y"]
            features["real_yield_10y_change_20d"] = macro["real_yield_10y"].diff(20)
            ry = macro["real_yield_10y"]
            features["real_yield_10y_zscore"] = (
                (ry - ry.rolling(252).mean()) /
                ry.rolling(252).std().replace(0, np.nan))
        if "breakeven_10y" in macro.columns:
            features["breakeven_10y"] = macro["breakeven_10y"]
        if "fed_funds_rate" in macro.columns:
            features["fed_funds_rate"] = macro["fed_funds_rate"]
            features["fed_funds_rate_change_60d"] = macro["fed_funds_rate"].diff(60)
        if "tw_usd" in macro.columns:
            features["tw_usd"] = macro["tw_usd"]
            features["tw_usd_ret_20d"] = macro["tw_usd"].pct_change(20)
            tw = macro["tw_usd"]
            features["tw_usd_zscore"] = (
                (tw - tw.rolling(252).mean()) /
                tw.rolling(252).std().replace(0, np.nan))
        if "cpi" in macro.columns:
            cpi = macro["cpi"]
            features["cpi_yoy"] = cpi.pct_change(252)  # 近似YoY
        if "m2" in macro.columns:
            m2 = macro["m2"]
            features["m2_yoy"] = m2.pct_change(252)
    else:
        print("  警告: macro_panel.csv 不存在, 宏观特征跳过")

    # ── GVZ ──
    gvz_path = os.path.join(RAW_VOL, "gvz.csv")
    if os.path.exists(gvz_path):
        gvz = pd.read_csv(gvz_path, index_col=0, parse_dates=True)
        gvz_close = gvz["Close"].reindex(gld.index).ffill()
        features["gvz"] = gvz_close
        features["gvz_pctile_252d"] = gvz_close.rolling(
            252, min_periods=60).rank(pct=True)
    else:
        print("  警告: gvz.csv 不存在, GVZ 特征跳过")

    # ── VRP ──
    if "rv_20d" in features.columns and "gvz" in features.columns:
        features["iv_rv_spread"] = features["gvz"] - features["rv_20d"]
        features["vrp_20d"] = features["gvz"] - features["rv_20d"]
        features["vrp_10d"] = features.get("gvz", 0) - features.get("rv_10d", 0)

    # ── COT (如果存在) ──
    cot_path = os.path.join(RAW_COT, "gold_cot.csv")
    if os.path.exists(cot_path):
        cot = pd.read_csv(cot_path, index_col=0, parse_dates=True)
        cot = cot.reindex(gld.index).ffill()
        for col in ["cot_noncomm_net", "cot_noncomm_net_change",
                     "cot_open_interest", "cot_oi_change_pct"]:
            if col in cot.columns:
                features[col] = cot[col]
        if "cot_noncomm_net" in features.columns:
            features["cot_noncomm_net_pctile"] = features[
                "cot_noncomm_net"].rolling(252, min_periods=60).rank(pct=True)

    # ── Central Bank (如果存在) ──
    cb_path = os.path.join(DATA_ROOT, "raw", "central_bank", "cb_features.csv")
    if os.path.exists(cb_path):
        cb = pd.read_csv(cb_path, index_col=0, parse_dates=True)
        cb = cb.reindex(gld.index).ffill()
        if "cb_global_12m_rolling" in cb.columns:
            features["cb_global_12m_rolling"] = cb["cb_global_12m_rolling"]

    # ── 标签 (forward returns) ──
    labels = pd.DataFrame(index=gld.index)
    for w in [5, 10, 20]:
        labels[f"fwd_ret_{w}d"] = close.pct_change(w).shift(-w)

    # 保存
    features = features.sort_index()
    features.to_parquet(os.path.join(PROCESSED, "features_all.parquet"))
    labels.to_parquet(os.path.join(PROCESSED, "labels.parquet"))

    # dataset = features + labels
    dataset = pd.concat([features, labels], axis=1)
    dataset.to_parquet(os.path.join(PROCESSED, "dataset.parquet"))

    print(f"  特征: {features.shape[1]} 列, {len(features)} 行")
    print(f"  保存: {PROCESSED}/features_all.parquet")
    return features


# ══════════════════════════════════════════════════════════
# Step 5: 训练 DL Range 模型
# ══════════════════════════════════════════════════════════
def train_model():
    """训练 DL Range 模型, 生成 OOS 预测."""
    print("\n[5/5] 训练 DL Range 模型...")

    features_path = os.path.join(PROCESSED, "features_all.parquet")
    gld_path = os.path.join(RAW_MARKET, "gld.csv")
    if not os.path.exists(features_path):
        print("  错误: 特征矩阵不存在, 请先运行特征构建")
        return

    try:
        import torch
        from core.dl_range import DLRangePredictor, SELECTED_FEATURES
    except ImportError:
        print("  跳过: PyTorch 未安装 (pip install torch)")
        return

    features = pd.read_parquet(features_path)
    gld = pd.read_csv(gld_path, index_col=0, parse_dates=True)
    common = features.index.intersection(gld.index)
    features, gld = features.loc[common], gld.loc[common]
    close, high, low = gld["Close"], gld["High"], gld["Low"]

    # 选择模型特征
    feat_cols = [f for f in SELECTED_FEATURES if f in features.columns]
    print(f"  可用特征: {len(feat_cols)}/{len(SELECTED_FEATURES)}")

    if len(feat_cols) < 20:
        print("  警告: 可用特征不足 20 个, 模型精度可能降低")

    # 构建目标
    max_high_5d = high.shift(-1).rolling(5).max().shift(-4)
    min_low_5d = low.shift(-1).rolling(5).min().shift(-4)
    upper_pct = (max_high_5d / close - 1) * 100
    lower_pct = (min_low_5d / close - 1) * 100
    log_ret = np.log(close / close.shift(1))
    rv_scale = log_ret.rolling(10).std() * np.sqrt(5) * 100

    valid = features[feat_cols].dropna().index
    valid = valid.intersection(rv_scale.dropna().index)
    valid = valid.intersection(upper_pct.dropna().index)

    print(f"  有效样本: {len(valid)}")

    # Walk-forward 训练
    seq_len = 20
    min_train = 1260
    test_size = 252
    cal_size = 126

    all_preds = []
    n = len(valid)
    fold = 0
    cutoff = min_train

    while cutoff + test_size <= n:
        train_end = cutoff - cal_size
        cal_start = cutoff - cal_size
        test_end = min(cutoff + test_size, n)

        train_dates = valid[:train_end]
        val_dates = valid[max(0, train_end - 252):train_end]
        cal_dates = valid[cal_start:cutoff]
        test_dates = valid[cutoff:test_end]

        fold += 1
        print(f"  Fold {fold}: train {len(train_dates)}, "
              f"test {len(test_dates)} "
              f"({test_dates[0].date()}~{test_dates[-1].date()})")

        X_tr = features.loc[train_dates, feat_cols].values
        u_tr = upper_pct.loc[train_dates].values
        l_tr = lower_pct.loc[train_dates].values
        rv_tr = rv_scale.loc[train_dates].values

        X_val = features.loc[val_dates, feat_cols].values
        u_val = upper_pct.loc[val_dates].values
        l_val = lower_pct.loc[val_dates].values
        rv_val = rv_scale.loc[val_dates].values

        X_cal = features.loc[cal_dates, feat_cols].values
        u_cal = upper_pct.loc[cal_dates].values
        l_cal = lower_pct.loc[cal_dates].values
        rv_cal = rv_scale.loc[cal_dates].values

        predictor = DLRangePredictor(
            seq_len=seq_len, hidden_size=64, num_layers=2,
            dropout=0.2, lr=1e-3, weight_decay=1e-4,
            epochs=150, batch_size=64, patience=20,
            q_upper=0.85, q_lower=0.15,
            n_ensemble=3, cal_target_cov=0.80)

        predictor.fit(X_tr, u_tr, l_tr, rv_tr,
                      X_val, u_val, l_val, rv_val,
                      X_cal, u_cal, l_cal, rv_cal)

        # 预测
        prefix_dates = valid[cutoff - seq_len + 1: cutoff]
        combined = pd.Index(prefix_dates).append(pd.Index(test_dates))
        X_comb = features.loc[combined, feat_cols].values
        rv_comb = rv_scale.loc[combined].values

        pred_u, pred_l = predictor.predict(X_comb, rv_comb)

        fold_df = pd.DataFrame({
            "pred_upper_pct": pred_u,
            "pred_lower_pct": pred_l,
            "actual_upper_pct": upper_pct.loc[test_dates].values,
            "actual_lower_pct": lower_pct.loc[test_dates].values,
            "gld_close": close.loc[test_dates].values,
        }, index=test_dates)
        all_preds.append(fold_df)

        cutoff += test_size

    # 最后不完整 fold
    if cutoff < n:
        remaining = valid[cutoff:]
        if len(remaining) >= 60:
            train_end = cutoff - cal_size
            train_dates = valid[:train_end]
            val_dates = valid[max(0, train_end - 252):train_end]
            cal_dates = valid[cutoff - cal_size:cutoff]

            fold += 1
            print(f"  Fold {fold}: train {len(train_dates)}, "
                  f"test {len(remaining)} "
                  f"({remaining[0].date()}~{remaining[-1].date()})")

            X_tr = features.loc[train_dates, feat_cols].values
            u_tr = upper_pct.loc[train_dates].values
            l_tr = lower_pct.loc[train_dates].values
            rv_tr = rv_scale.loc[train_dates].values

            X_val = features.loc[val_dates, feat_cols].values
            u_val = upper_pct.loc[val_dates].values
            l_val = lower_pct.loc[val_dates].values
            rv_val = rv_scale.loc[val_dates].values

            X_cal = features.loc[cal_dates, feat_cols].values
            u_cal = upper_pct.loc[cal_dates].values
            l_cal = lower_pct.loc[cal_dates].values
            rv_cal = rv_scale.loc[cal_dates].values

            predictor = DLRangePredictor(
                seq_len=seq_len, hidden_size=64, num_layers=2,
                dropout=0.2, lr=1e-3, weight_decay=1e-4,
                epochs=150, batch_size=64, patience=20,
                q_upper=0.85, q_lower=0.15,
                n_ensemble=3, cal_target_cov=0.80)

            predictor.fit(X_tr, u_tr, l_tr, rv_tr,
                          X_val, u_val, l_val, rv_val,
                          X_cal, u_cal, l_cal, rv_cal)

            prefix_dates = valid[cutoff - seq_len + 1: cutoff]
            combined = pd.Index(prefix_dates).append(pd.Index(remaining))
            X_comb = features.loc[combined, feat_cols].values
            rv_comb = rv_scale.loc[combined].values

            pred_u, pred_l = predictor.predict(X_comb, rv_comb)

            fold_df = pd.DataFrame({
                "pred_upper_pct": pred_u[:len(remaining)],
                "pred_lower_pct": pred_l[:len(remaining)],
                "actual_upper_pct": upper_pct.loc[remaining].values,
                "actual_lower_pct": lower_pct.loc[remaining].values,
                "gld_close": close.loc[remaining].values,
            }, index=remaining)
            all_preds.append(fold_df)

    if all_preds:
        oos = pd.concat(all_preds)
        oos = oos[~oos.index.duplicated(keep="last")]
        oos_path = os.path.join(MODELS, "dl_range_v2_oos.parquet")
        oos.to_parquet(oos_path)
        print(f"\n  OOS 预测: {len(oos)} rows "
              f"({oos.index[0].date()} ~ {oos.index[-1].date()})")
        print(f"  保存: {oos_path}")

        # 保存最后一个 fold 的模型 (用于在线 inference)
        model_path = os.path.join(MODELS, "dl_range_v2_model.pkl")
        predictor.save(model_path)
        print(f"  模型权重: {model_path}")
    else:
        print("  警告: 无预测结果")


def main():
    parser = argparse.ArgumentParser(description="GoldDash 数据下载与特征构建")
    parser.add_argument("--fred-key", help="FRED API Key")
    parser.add_argument("--no-train", action="store_true",
                        help="跳过模型训练")
    parser.add_argument("--train-only", action="store_true",
                        help="仅训练模型 (假设数据已下载)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GoldDash 数据设置")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据目录: {DATA_ROOT}")
    print("=" * 60)

    ensure_dirs()

    if not args.train_only:
        download_market_data()
        download_macro_data(args.fred_key)
        download_vol_data()
        build_features()

    if not args.no_train:
        train_model()

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)
    print("\n下一步:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
