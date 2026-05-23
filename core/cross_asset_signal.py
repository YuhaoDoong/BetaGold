"""v3.7.224: 跨资产信号规则 — SLV tier=S → GLD 同步开仓.

实证依据 (cross_asset_sync.py, 历史 n=24 笔 SLV-S):
  SLV tier=S 触发当日, GLD 同日开 BUY CALL 多头:
    5d:  GLD WR=69.6%  mean=+1.07%
    10d: GLD WR=78.3%  mean=+3.46%
    20d: GLD WR=82.6%  mean=+3.69%
  方向同步率 96% (10d)
  相比 GLD 自家 IV filter 拦截, 这条规则:
    - 历史 n=24 笔, 统计显著
    - 不被 GLD IV filter 拦
    - 利用金银 0.78 corr 的物理同步性

规则:
  当 SLV 当日 buy_signal=True 且 signal_tier='S':
    给 GLD 同日生成虚拟 BUY CALL 入场 (asset='GLD', strategy='BUY CALL', source='SLV-S sync')

注意:
  - 这条规则只在 SLV 真正出 S 时触发, 不主观乱开
  - 入场: GLD 当日 Open 开 BUY CALL (30 DTE)
  - 退出: 跟 GLD 自身 BC 同套规则 (pt 2.5x / sl 0.3x / 30d)
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo


CROSS_ENABLED = True             # 一键开关
# v3.7.226 实证 (all-tier cross_asset_all_tiers.csv):
#   SLV → GLD spot 10d 回报 by tier:
#     S    n=23 WR=78.3% mean=+3.46% sum=79.5 max_loss=-4.4%
#     A    n=32 WR=71.9% mean=+1.51% sum=48.2 max_loss=-3.5%
#     S+A  n=55 WR=74.5% mean=+2.32% sum=127.8 ★ 推荐 (n 翻倍, sum 最高)
#     B    n=49 WR=63.3% mean=+0.71% sum=34.7
#     ALL  n=104 WR=69.2% mean=+1.56% sum=162.4
#   反向 GLD → SLV: GLD-S 反向 (WR 57% mean -0.09%), 不要 cross
# v3.7.225 GLD 三策略 P&L:
#   BUY CALL:  WR=57%, mean=+70%, sum=+492%  ★ alpha 最大
#   SELL PUT:  WR=89%, mean=+27%, sum=+239%   高 WR 稳健
#   FUTURES 5x: WR=40%, sum=-203%             ❌ 灾难
CROSS_TIERS = ("S", "A")         # 触发 tier 集合 (S+A 综合最优)
CROSS_STRATEGY = "BUY CALL"      # "BUY CALL" / "SELL PUT" / 双开


def find_cross_entries(slv_sig_df: pd.DataFrame,
                          window_start: pd.Timestamp,
                          window_end: pd.Timestamp) -> list:
    """返回 SLV 触发指定 tier (CROSS_TIERS) 的日期列表."""
    if not CROSS_ENABLED or slv_sig_df is None or not len(slv_sig_df):
        return []
    bs = slv_sig_df["buy_signal"].fillna(False).astype(bool)
    st = slv_sig_df["signal_tier"].fillna("")
    mask = (bs
             & st.isin(list(CROSS_TIERS))
             & (slv_sig_df.index >= window_start)
             & (slv_sig_df.index <= window_end))
    return slv_sig_df.index[mask].tolist()


def should_add_gld_sync(gld_existing_rows: list, slv_s_date: pd.Timestamp,
                          gld_sig_df: pd.DataFrame) -> bool:
    """决定是否给 GLD 加 SLV-S sync 入场.

    跳过条件:
      1. 当日 GLD 已经自己有 BUY CALL 信号 (避免重复)
      2. 当日 GLD 已有 SLV-sync 行 (避免重复)
      3. GLD sig_df 显示当日 ma_trend_skip=True (下跌趋势, 不接飞刀)
    """
    d_iso = pd.Timestamp(slv_s_date).normalize().isoformat()
    for r in gld_existing_rows:
        if r.get("asset") != "GLD": continue
        if not str(r.get("signal_date", "")).startswith(d_iso[:10]): continue
        strat = r.get("strategy", "")
        # 已有 GLD BC (任何来源) → 跳过
        if "BUY CALL" in strat: return False
    # GLD 自己 ma_trend 严重下行也跳过 (cross 规则不无脑跟 SLV)
    if gld_sig_df is not None and slv_s_date in gld_sig_df.index:
        ma_skip = bool(gld_sig_df.loc[slv_s_date].get("ma_trend_skip", False))
        if ma_skip:
            return False
    return True
