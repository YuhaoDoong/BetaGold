"""持续运行模式 — 服务器后台跑, 监控信号变化, 推送通知 (v3.7.58).

设计目标:
  当前: 用户手动开 dashboard 看信号
  本模块: 后台进程定期 (默认 5 min) 检查信号, 变更时推送 → 用户被动接收
  未来: 进一步接 broker API → 全自动下单

工作流 (循环):
  1. auto_refresh_market_data: 拉最新 yfinance 数据 (含腐败检测)
  2. 计算 sig_df / unified / 切点 / 1h synthesis bp_low
  3. 构造 SignalSnapshot
  4. notifier.notify_signal_change(): 跟上次快照对比, 变化时推送
  5. sleep poll_interval

通知渠道 (按 env 启用, 任一即可):
  TG_BOT_TOKEN + TG_CHAT_ID  → Telegram
  SMTP_HOST + SMTP_USER + SMTP_PASS + SMTP_TO  → Email
  WEBHOOK_URL                → 自定义 webhook (Slack/Discord/IFTTT 等)
  默认: FileChannel 写 ~/GoldDash_signals.log

用法:
  python scripts/continuous_runner.py --asset GLD --poll-min 5
  python scripts/continuous_runner.py --asset GLD,SLV --poll-min 15

部署 (后台):
  nohup python scripts/continuous_runner.py --poll-min 5 > /tmp/runner.log 2>&1 &
  # 或 systemd / launchd 持久化
"""
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from core.data import (load_features, load_config, load_oos_predictions,
                         auto_refresh_market_data)
from core.signals import build_band, compute_rv_pctile
from core.signals_v2 import generate_daily_signals
from core.events import (detect_straddle_signal, detect_short_vol_signal)
from core.regime import RegimeClassifier
from core.strategy_selector import build_unified_signals, dedupe_unified
from core.strategy_config import get_config
from core.notifier import (Notifier, SignalSnapshot,
                              FileChannel, TelegramChannel,
                              EmailChannel, WebhookChannel)


def build_snapshot(asset: str, cfg: dict) -> SignalSnapshot:
    """构建当前信号快照 (跟 dashboard 一致的逻辑)."""
    fname = "gld.csv" if asset == "GLD" else "slv.csv"
    fname_1h = "gld_1h.csv" if asset == "GLD" else "slv_1h.csv"
    df_d = pd.read_csv(f"{cfg['data_root']}/raw/market/{fname}",
                          index_col=0, parse_dates=True)
    df_1h_path = f"{cfg['data_root']}/raw/market/{fname_1h}"
    df_1h = (pd.read_csv(df_1h_path, index_col=0, parse_dates=True)
              if os.path.exists(df_1h_path) else None)
    if df_1h is not None:
        df_1h.index = pd.to_datetime(df_1h.index)
    features = load_features(cfg)
    common = features.index.intersection(df_d.index)
    close_d = df_d["Close"][common]
    high_d = df_d["High"][common]
    low_d = df_d["Low"][common]

    # OOS — 用 SLV 自己的 OOS 文件
    if asset == "SLV":
        oos_path = f"{cfg['data_root']}/models/dl_range_slv_oos.parquet"
        oos = (pd.read_parquet(oos_path) if os.path.exists(oos_path)
                else load_oos_predictions(cfg))
    else:
        oos = load_oos_predictions(cfg)
    upper, lower, _ = build_band(oos, close_d)
    rv_pct = compute_rv_pctile(features.loc[common, "rv_10d"])
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier(min_hold_days=1).classify(
        features[feat_cols])["regime"]  # v3.7.233 explicit no-lookahead

    # 1h synthesis (跟 dashboard 一致)
    _close = close_d.copy(); _high = high_d.copy(); _low = low_d.copy()
    bp_dates = upper.dropna().index.intersection(lower.dropna().index)
    last_d = bp_dates[-1] if len(bp_dates) else close_d.index[-1]
    if df_1h is not None:
        _intra = df_1h[df_1h.index > last_d]
        if len(_intra) > 0:
            _new_d = _intra.index[-1].normalize()
            _close.loc[_new_d] = _intra["Close"].iloc[-1]
            _high.loc[_new_d] = _intra["High"].max()
            _low.loc[_new_d] = _intra["Low"].min()
            _close = _close.sort_index()
            _high = _high.sort_index()
            _low = _low.sort_index()

    sig_df = generate_daily_signals(_close, _high, _low, upper, lower,
                                       regime, rv_pct, asset=asset)
    rv_s = features["rv_10d"]
    straddle_df = detect_straddle_signal(rv_s, sig_df.index,
                                              rv_pctile=rv_pct,
                                              close=_close, high=_high, low=_low,
                                              asset=asset)
    short_vol_df = detect_short_vol_signal(rv_s, rv_pct, sig_df.index,
                                                regime=regime,
                                                close=_close, high=_high, low=_low,
                                                asset=asset)
    unified = build_unified_signals(sig_df, straddle_df, _close, _high, _low,
                                       short_vol_df=short_vol_df)
    last_row = unified.iloc[-1]
    last_date = unified.index[-1]
    chosen = last_row["chosen"] or "—"

    # 当前 RV
    rv_v = float(rv_s.get(last_date, 0))
    rv_p = float(rv_pct.get(last_date, 0))
    bp_low_v = float(sig_df.loc[last_date, "bp_low"]) \
        if last_date in sig_df.index else None

    if "STRADDLE" in chosen:
        score = float(last_row.get("straddle_score", 0))
    elif "SHORT_VOL" in chosen:
        score = float(last_row.get("short_vol_score", 0))
    else:
        score = float(last_row.get("straddle_score", 0))

    # 当前 SGT 时段 — US 期权时段 SGT 21:30 ~ 04:00
    now_h = datetime.now().hour + datetime.now().minute / 60.0
    is_us = (now_h >= 21.5) or (now_h < 4.0)

    return SignalSnapshot(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        asset=asset,
        chosen=chosen,
        score=score,
        rv=rv_v,
        rv_pctile=rv_p,
        bp_low=bp_low_v if bp_low_v is not None else 0.0,
        is_us_session=is_us,
        extras={
            "last_date": str(last_date.date()),
            "close": float(_close.iloc[-1]),
            "config_切点": get_config(asset).rv_filter_high,
        },
    )


def build_notifier() -> Notifier:
    """根据环境变量启用所有可用渠道."""
    channels = [FileChannel()]
    if os.environ.get("TG_BOT_TOKEN") and os.environ.get("TG_CHAT_ID"):
        channels.append(TelegramChannel())
    if (os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER")
            and os.environ.get("SMTP_TO")):
        channels.append(EmailChannel())
    if os.environ.get("WEBHOOK_URL"):
        channels.append(WebhookChannel())
    return Notifier(channels=channels)


def run_once(assets: list, notifier: Notifier, cfg: dict,
              refresh_data: bool = True):
    """单次扫描所有资产, 推送变更."""
    if refresh_data:
        try:
            auto_refresh_market_data(cfg)
        except Exception as e:
            print(f"[refresh] failed: {e}")

    for asset in assets:
        try:
            snap = build_snapshot(asset, cfg)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {asset}: "
                  f"chosen={snap.chosen} score={snap.score} "
                  f"rv={snap.rv:.1f}% rv%={snap.rv_pctile:.0%}")
            results = notifier.notify_signal_change(snap)
            if results:
                print(f"  推送: {results}")
        except Exception:
            print(f"[{asset}] 失败:")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="GLD,SLV",
                         help="逗号分隔, 默认 GLD,SLV")
    parser.add_argument("--poll-min", type=int, default=5,
                         help="轮询间隔分钟, 默认 5")
    parser.add_argument("--once", action="store_true",
                         help="只跑一次, 不循环 (用于 cron)")
    parser.add_argument("--no-refresh", action="store_true",
                         help="跳过 yfinance 数据刷新")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.asset.split(",") if a.strip()]
    cfg = load_config()
    notifier = build_notifier()
    print(f"=== 持续运行模式 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"资产: {assets}, 轮询: {args.poll_min}min, "
          f"渠道: {[c.name for c in notifier.channels]}")
    if args.once:
        run_once(assets, notifier, cfg, refresh_data=not args.no_refresh)
        return

    while True:
        try:
            run_once(assets, notifier, cfg, refresh_data=not args.no_refresh)
        except KeyboardInterrupt:
            print("\n停止")
            break
        except Exception:
            traceback.print_exc()
        time.sleep(args.poll_min * 60)


if __name__ == "__main__":
    main()
