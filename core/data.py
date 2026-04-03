"""数据加载模块 — 从 config.yaml 指定的路径读取所有数据."""

import os
import logging
from glob import glob
from datetime import datetime, timezone, timedelta

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# SGT/北京时区 (UTC+8)
_TZ_SGT = timezone(timedelta(hours=8))


def load_config(config_path: str = None) -> dict:
    """加载配置文件, 返回解析后的绝对路径."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["data_root"]
    resolved = {}
    for key, rel in cfg["paths"].items():
        resolved[key] = os.path.join(root, rel)
    cfg["resolved"] = resolved
    return cfg


def load_features(cfg: dict) -> pd.DataFrame:
    """加载特征矩阵."""
    return pd.read_parquet(cfg["resolved"]["features"])


def load_gld(cfg: dict) -> pd.DataFrame:
    """加载 GLD OHLCV."""
    return pd.read_csv(cfg["resolved"]["gld_csv"],
                       index_col=0, parse_dates=True)


def load_oos_predictions(cfg: dict) -> pd.DataFrame:
    """加载 DL Range OOS 预测."""
    return pd.read_parquet(cfg["resolved"]["oos_predictions"])


def load_gold_futures(cfg: dict) -> pd.DataFrame:
    """加载纽约黄金期货 (GC=F) OHLCV."""
    path = cfg["resolved"].get("gold_futures_csv")
    if path and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def load_usdcny(cfg: dict) -> pd.Series:
    """加载 USD/CNY 汇率. 返回 Close Series 或 None."""
    path = cfg["resolved"].get("usdcny_csv")
    if path and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["Close"]
    return None


def fetch_realtime_gold_fx():
    """获取实时金价和汇率 (via yfinance).

    Returns dict: {gc_price, usdcny, shfe_approx, timestamp}
    Returns None if fetch fails.
    """
    try:
        import yfinance as yf
        from datetime import datetime
        tickers = yf.Tickers("GC=F CNY=X")
        gc_info = tickers.tickers["GC=F"].fast_info
        cny_info = tickers.tickers["CNY=X"].fast_info
        gc_price = gc_info.get("lastPrice") or gc_info.get("previousClose")
        cny_rate = cny_info.get("lastPrice") or cny_info.get("previousClose")
        if gc_price and cny_rate:
            return {
                "gc_price": float(gc_price),
                "usdcny": float(cny_rate),
                "shfe_approx": float(gc_price) * float(cny_rate) / 31.1035,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
    except Exception:
        pass
    return None


def get_today_sgt():
    """返回 SGT (UTC+8) 的今日日期."""
    return datetime.now(_TZ_SGT).date()


def auto_refresh_market_data(cfg: dict):
    """检测市场数据是否过期, 自动下载最新数据.

    检查 GLD, 黄金期货, USD/CNY 三个 CSV 的最后日期,
    如果落后于今天 (SGT) 且是交易日, 则用 yfinance 下载增量数据并追加.

    Returns: list of (ticker, status_message)
    """
    today = get_today_sgt()
    results = []

    updates = [
        ("gld_csv", "GLD", "GLD"),
        ("gold_futures_csv", "黄金期货", "GC=F"),
        ("usdcny_csv", "USD/CNY", "CNY=X"),
    ]

    # 1h 数据也刷新 (GLD 含盘前盘后, GC=F)
    market_dir = os.path.dirname(cfg["resolved"].get("gld_csv", ""))
    for fname, label, ticker, prepost in [
        ("gld_1h.csv", "GLD 1h", "GLD", True),
        ("gc_1h.csv", "GC=F 1h", "GC=F", False),
    ]:
        path_1h = os.path.join(market_dir, fname)
        if os.path.exists(path_1h):
            try:
                import yfinance as yf
                existing_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
                last_1h = existing_1h.index[-1]
                # 只在距上次超过 6 小时才刷新
                now_utc = datetime.utcnow()
                hours_since = (now_utc - last_1h.to_pydatetime()).total_seconds() / 3600
                if hours_since > 6:
                    t = yf.Ticker(ticker)
                    new_1h = t.history(period="5d", interval="1h",
                                       prepost=prepost)
                    if new_1h is not None and len(new_1h) > 0:
                        new_1h.index = pd.to_datetime(new_1h.index).tz_localize(None)
                        new_1h.index.name = "Datetime"
                        new_1h = new_1h[["Open", "High", "Low", "Close", "Volume"]]
                        new_1h = new_1h[new_1h.index > existing_1h.index[-1]]
                        if len(new_1h) > 0:
                            combined = pd.concat([existing_1h, new_1h])
                            combined.to_csv(path_1h)
                            results.append((label,
                                f"+{len(new_1h)}根 至 {combined.index[-1].strftime('%m/%d %H:%M')}"))
                        else:
                            results.append((label, f"已是最新"))
                    else:
                        results.append((label, "无新数据"))
                else:
                    results.append((label, f"已是最新 (<6h)"))
            except Exception as e:
                results.append((label, f"刷新失败: {e}"))

    for cfg_key, label, yf_ticker in updates:
        path = cfg["resolved"].get(cfg_key)
        if not path or not os.path.exists(path):
            results.append((label, "文件不存在, 跳过"))
            continue

        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            last_date = existing.index[-1].date()

            # 如果最后日期 >= 上一个交易日, 无需更新
            # 周末: 周六/日 → 周五是最后交易日
            ref = today
            wd = ref.weekday()
            if wd == 5:  # Saturday
                last_bday = ref - timedelta(days=1)
            elif wd == 6:  # Sunday
                last_bday = ref - timedelta(days=2)
            else:
                last_bday = ref

            if last_date >= last_bday:
                results.append((label, f"已是最新 ({last_date})"))
                continue

            # 下载增量数据
            import yfinance as yf
            start = last_date + timedelta(days=1)
            ticker = yf.Ticker(yf_ticker)
            new_data = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"))

            if new_data is None or len(new_data) == 0:
                results.append((label, f"无新数据 (最新 {last_date})"))
                continue

            # 统一列名 (yfinance 返回 Open/High/Low/Close/Volume)
            new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
            new_data.index.name = "Date"
            cols_keep = [c for c in ["Close", "High", "Low", "Open", "Volume"]
                         if c in new_data.columns]
            new_data = new_data[cols_keep]

            # 去重: 只保留比 existing 更新的日期
            new_data = new_data[new_data.index > existing.index[-1]]
            if len(new_data) == 0:
                results.append((label, f"无新数据 (最新 {last_date})"))
                continue

            # 追加并保存
            combined = pd.concat([existing, new_data])
            combined.to_csv(path)
            new_last = combined.index[-1].date()
            results.append(
                (label, f"更新 {last_date} → {new_last} (+{len(new_data)}行)"))
            logger.info("Updated %s: %s → %s (+%d rows)",
                        label, last_date, new_last, len(new_data))

        except Exception as e:
            results.append((label, f"更新失败: {e}"))
            logger.warning("Failed to update %s: %s", label, e)

    return results


_FEATURE_REBUILT_TODAY = False  # 每个进程生命周期只重建一次


def update_features_full(cfg: dict):
    """全量重建特征矩阵: 下载最新市场+宏观数据, 完整重建所有特征.

    每天第一次运行时强制重建, 确保 DXY/利率/VIX/波动率等全部更新.
    同一进程内不重复执行.
    """
    global _FEATURE_REBUILT_TODAY

    if _FEATURE_REBUILT_TODAY:
        return 0, "今日已重建"

    feat_path = cfg["resolved"]["features"]
    feat_old = pd.read_parquet(feat_path)

    try:
        import sys
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        import setup_data

        # 覆盖 setup_data 的路径, 指向实际数据目录
        data_root = cfg.get("data_root", "")
        if os.path.isdir(data_root):
            setup_data.DATA_ROOT = data_root
            setup_data.RAW_MARKET = os.path.join(data_root, "raw", "market")
            setup_data.RAW_MACRO = os.path.join(data_root, "raw", "macro")
            setup_data.RAW_VOL = os.path.join(data_root, "raw", "volatility")
            setup_data.RAW_COT = os.path.join(data_root, "raw", "cot")
            setup_data.PROCESSED = os.path.join(data_root, "processed")
            setup_data.MODELS = os.path.join(data_root, "models")

        # 1. 下载最新市场数据 (GLD/DXY/VIX/原油/白银/铜/美债/GC=F)
        setup_data.download_market_data()

        # 2. 下载宏观数据 (FRED)
        fred_key = None
        gold_cfg_path = os.path.join(os.path.dirname(data_root),
                                      "config", "settings.yaml")
        if os.path.exists(gold_cfg_path):
            with open(gold_cfg_path, "r") as _f:
                import yaml as _yaml
                _gold_cfg = _yaml.safe_load(_f)
                fred_key = _gold_cfg.get("fred_api_key")
        try:
            setup_data.download_macro_data(fred_key)
        except Exception:
            pass  # 失败时用已有宏观数据

        # 3. 下载波动率数据 (GVZ)
        try:
            setup_data.download_vol_data()
        except Exception:
            pass

        # 4. 全量重建特征 (从原始数据计算)
        setup_data.build_features()

        _FEATURE_REBUILT_TODAY = True

        feat_new = pd.read_parquet(feat_path)
        n_new = len(feat_new) - len(feat_old)

        # 检查关键宏观特征是否有变化
        changed = []
        for c in ["dxy_ret_5d", "vix_level", "us10y_level", "crude_ret_5d"]:
            if c in feat_new.columns and c in feat_old.columns:
                if abs(feat_new[c].iloc[-1] - feat_old[c].iloc[-1]) > 1e-6:
                    changed.append(c)

        change_str = f" 宏观更新: {','.join(changed)}" if changed else ""
        return n_new, f"全量重建 → {feat_new.index[-1].date()}{change_str}"

    except Exception as e:
        logger.warning("Full feature rebuild failed: %s", e)
        return 0, f"重建失败: {e}"


def extend_oos_predictions(cfg: dict):
    """用保存的模型对最新数据做 inference, 扩展 OOS 预测到今天.

    流程:
      1. 加载 OOS parquet, 检查最后日期
      2. 加载特征, 找到 OOS 之后的新日期
      3. 加载模型权重, 对新日期做预测
      4. 追加到 OOS parquet
    """
    oos_path = cfg["resolved"]["oos_predictions"]
    model_path = os.path.join(os.path.dirname(oos_path), "dl_range_v2_model.pkl")

    if not os.path.exists(model_path):
        return 0, "模型权重未找到, 请运行 setup_data.py 训练"

    oos = pd.read_parquet(oos_path)
    oos_last = oos.index[-1]

    features = pd.read_parquet(cfg["resolved"]["features"])
    new_dates = features.index[features.index > oos_last]

    if len(new_dates) == 0:
        return 0, f"已是最新 ({oos_last.date()})"

    from core.dl_range import DLRangePredictor, select_features
    import numpy as np

    predictor = DLRangePredictor.load(model_path)
    n_expected = predictor.scaler.n_features_in_
    feat_cols = select_features(features)

    # 精确匹配模型训练时的特征数量
    if len(feat_cols) < n_expected:
        # 补齐缺失特征 (填0)
        from core.dl_range import SELECTED_FEATURES
        for f in SELECTED_FEATURES:
            if f not in features.columns:
                features[f] = 0
        feat_cols = select_features(features)

    # 如果还是不匹配 (多了), 截取到 n_expected 个
    if len(feat_cols) > n_expected:
        feat_cols = feat_cols[:n_expected]

    # 需要 seq_len 根历史 + 新日期
    start_idx = max(0, features.index.get_loc(oos_last) - predictor.seq_len - 5)
    feat_window = features.iloc[start_idx:][feat_cols]

    # RV scale
    rv_col = "rv_10d" if "rv_10d" in features.columns else None
    if rv_col:
        rv_scale = features.iloc[start_idx:][rv_col].values
    else:
        rv_scale = np.ones(len(feat_window))

    pred_u, pred_l = predictor.predict(feat_window.values, rv_scale)

    # 对齐到日期
    pred_dates = feat_window.index[predictor.seq_len - 1:]
    pred_df = pd.DataFrame({
        "pred_upper_pct": pred_u[:len(pred_dates)],
        "pred_lower_pct": pred_l[:len(pred_dates)],
    }, index=pred_dates[:len(pred_u)])

    # 只保留新日期
    new_preds = pred_df[pred_df.index > oos_last]
    if len(new_preds) == 0:
        return 0, f"已是最新 ({oos_last.date()})"

    # 合理性检查: 宽度和中心值都不应偏离历史太远
    hist_width = (oos["pred_upper_pct"] - oos["pred_lower_pct"]).median()
    hist_center = ((oos["pred_upper_pct"] + oos["pred_lower_pct"]) / 2).median()
    max_width = max(hist_width * 3, 15.0)
    max_center_dev = max(hist_width * 0.5, 3.0)  # center 偏差不超过半个宽度

    for idx in new_preds.index:
        u = new_preds.loc[idx, "pred_upper_pct"]
        l = new_preds.loc[idx, "pred_lower_pct"]
        w = u - l
        center = (u + l) / 2

        # clamp center
        if abs(center - hist_center) > max_center_dev:
            center = hist_center + max_center_dev * (1 if center > hist_center else -1)

        # clamp width
        if w > max_width:
            w = max_width

        new_preds.loc[idx, "pred_upper_pct"] = center + w / 2
        new_preds.loc[idx, "pred_lower_pct"] = center - w / 2

    # 追加并保存
    combined = pd.concat([oos, new_preds])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(oos_path)

    return len(new_preds), f"预测扩展 {oos_last.date()} → {combined.index[-1].date()} (+{len(new_preds)}天)"


def fetch_live_options(spot_price, expiry_start=None, expiry_end=None,
                       strike_range=20):
    """从 Moomoo API 获取 GLD 期权实时报价.

    Args:
        spot_price: GLD 当前价格 (用于筛选 ATM 附近)
        expiry_start/end: 到期日范围 (默认最近2周)
        strike_range: ATM 附近的 strike 范围 ($)

    Returns: DataFrame with live quotes, or None
    """
    try:
        import sys
        sys.path.insert(0, "/Users/yhdong/Gold/src")
        from moomoo import OpenQuoteContext, RET_OK

        if expiry_start is None:
            today = datetime.now().date()
            expiry_start = today.strftime("%Y-%m-%d")
            expiry_end = (today + timedelta(days=45)).strftime("%Y-%m-%d")

        import time as _time
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            ret, chain = ctx.get_option_chain(
                "US.GLD", start=expiry_start, end=expiry_end)
            if ret != RET_OK or len(chain) == 0:
                return None

            # 筛选 ATM 附近 strike
            def _extract_strike(code):
                try:
                    s = code.split(".")[-1]
                    return int(s[10:]) / 1000
                except Exception:
                    return 0
            chain["_strike"] = chain["code"].apply(_extract_strike)
            lo = spot_price - strike_range
            hi = spot_price + strike_range
            atm = chain[(chain["_strike"] >= lo) & (chain["_strike"] <= hi)]
            if len(atm) == 0:
                atm = chain.head(50)

            codes = atm["code"].tolist()[:200]
            _time.sleep(0.3)  # 避免频率限制
            ret2, snap = ctx.get_market_snapshot(codes)
            if ret2 != RET_OK:
                return None

            return snap
        finally:
            ctx.close()
    except Exception:
        return None


def load_latest_eod_snapshot(cfg: dict):
    """加载最新 EOD 期权快照. 返回 (df, date_str) 或 (None, None)."""
    snap_dir = cfg["resolved"]["eod_snapshots"]
    if not os.path.isdir(snap_dir):
        return None, None
    snaps = sorted(glob(os.path.join(snap_dir, "202*", "eod_full.parquet")))
    if not snaps:
        return None, None
    latest = snaps[-1]
    snap_date = os.path.basename(os.path.dirname(latest))
    return pd.read_parquet(latest), snap_date


def load_all_eod_snapshots(cfg: dict):
    """加载所有 EOD 快照. 返回 {pd.Timestamp: DataFrame}."""
    snap_dir = cfg["resolved"]["eod_snapshots"]
    if not os.path.isdir(snap_dir):
        return {}
    snaps = sorted(glob(os.path.join(snap_dir, "202*", "eod_full.parquet")))
    result = {}
    for path in snaps:
        date_str = os.path.basename(os.path.dirname(path))
        try:
            ts = pd.Timestamp(date_str)
            result[ts] = pd.read_parquet(path)
        except Exception:
            pass
    return result
