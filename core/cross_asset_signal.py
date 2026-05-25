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
# v3.7.240: CROSS_STRATEGY 不再是固定常量 — 由 select_gld_sync_strategy
# 按 GLD 当日 bp_low + GVZ 决策 BUY CALL vs SELL PUT.
# 2026-03 5/5 GLD BC 全亏 (sum -334%) 主因正是固定 BUY CALL 不感知 IV,
# 在高 IV 深破日 (GVZ ≥25, bp_low ≤0.10) 应改 SELL PUT.

# Tunables for the IV-aware selector (DEC-5: trend targets, not hard limits).
CROSS_GVZ_HIGH_THRESHOLD = 25.0   # 高 IV regime 下沿
CROSS_BP_LOW_DEEP_BREAK = 0.10    # 深破阈值
CROSS_GVZ_STALE_MAX_DAYS = 2      # GVZ stale 容忍 (trading days vs signal_date)

# Shadow log location (caller writes; selector itself is pure)
CROSS_SHADOW_LOG_PATH = "/Users/yhdong/Gold/data/cross_asset_shadow_log.jsonl"
CROSS_LIVE_CUTOVER_MIN_DAYS = 14   # 14 个日历日 shadow 累积才允许 live flip


def select_gld_sync_strategy(signal_date,
                                 gld_signal_row,
                                 gvz_value,
                                 gvz_asof_date) -> dict:
    """Pure IV-aware cross-asset strategy selector (v3.7.240).

    Args:
        signal_date: SLV-S 触发日 (pd.Timestamp). 用作 GVZ staleness 的参考时间锚,
            NOT wall-clock today (回放语义一致).
        gld_signal_row: GLD 当日 sig_df 单行 (Series/dict). 必须含 'bp_low'.
        gvz_value: GLD GVZ at signal_date (float or None/NaN).
        gvz_asof_date: GVZ 数据点对应的日期 (pd.Timestamp or None).

    Returns:
        ``{"strategy": "BUY CALL" | "SELL PUT", "reason": str, "gvz_status": str}``

    Truth table (evaluated in order):
        1. ``gvz_value`` 是 None / NaN, 或 ``signal_date − gvz_asof_date``
           超过 ``CROSS_GVZ_STALE_MAX_DAYS`` 交易日 →
           ``BUY CALL`` reason='GVZ_UNAVAILABLE' status='missing'|'stale'.
        2. ``gld_signal_row.bp_low ≤ CROSS_BP_LOW_DEEP_BREAK`` AND
           ``gvz_value ≥ CROSS_GVZ_HIGH_THRESHOLD`` →
           ``SELL PUT`` reason='DEEP_BREAK_HIGH_IV' status='fresh'.
        3. 其他 → ``BUY CALL`` reason='DEFAULT' status='fresh'.

    This function MUST be pure (no I/O, no globals beyond constants). Shadow
    logging is the caller's responsibility via ``write_shadow_record``.
    """
    import math
    sig_d = pd.Timestamp(signal_date).normalize()
    # ---- GVZ availability + staleness ----
    if gvz_value is None or (isinstance(gvz_value, float) and math.isnan(gvz_value)):
        return {"strategy": "BUY CALL", "reason": "GVZ_UNAVAILABLE",
                  "gvz_status": "missing"}
    if gvz_asof_date is None:
        return {"strategy": "BUY CALL", "reason": "GVZ_UNAVAILABLE",
                  "gvz_status": "missing"}
    asof = pd.Timestamp(gvz_asof_date).normalize()
    if asof > sig_d:
        # v3.7.250 (review fix P2#2): asof > signal_date is a future leak
        # under the pure-function contract — must be rejected as invalid.
        # Production caller currently guards this with `gvz_close.loc[:d]`,
        # but the selector must defend itself too.
        return {"strategy": "BUY CALL", "reason": "GVZ_UNAVAILABLE",
                  "gvz_status": "future_asof_invalid"}
    # Trading-day gap (Mon-Fri); see core.data_freshness._trading_day_gap.
    gap_td = max(0, len(pd.bdate_range(asof + pd.Timedelta(days=1), sig_d)))
    if gap_td > CROSS_GVZ_STALE_MAX_DAYS:
        return {"strategy": "BUY CALL", "reason": "GVZ_UNAVAILABLE",
                  "gvz_status": "stale"}
    # ---- bp_low extraction ----
    try:
        bp_low = float(gld_signal_row["bp_low"]
                          if hasattr(gld_signal_row, "__getitem__")
                          else gld_signal_row.bp_low)
    except (KeyError, AttributeError, TypeError, ValueError):
        # GLD signal row missing bp_low → fallback to default BC, do not raise
        return {"strategy": "BUY CALL", "reason": "GLD_BP_LOW_MISSING",
                  "gvz_status": "fresh"}
    if isinstance(bp_low, float) and math.isnan(bp_low):
        return {"strategy": "BUY CALL", "reason": "GLD_BP_LOW_MISSING",
                  "gvz_status": "fresh"}
    # ---- Truth table ----
    if (bp_low <= CROSS_BP_LOW_DEEP_BREAK and
        float(gvz_value) >= CROSS_GVZ_HIGH_THRESHOLD):
        return {"strategy": "SELL PUT", "reason": "DEEP_BREAK_HIGH_IV",
                  "gvz_status": "fresh"}
    return {"strategy": "BUY CALL", "reason": "DEFAULT",
              "gvz_status": "fresh"}


def write_shadow_record(decision: dict,
                            signal_date,
                            slv_tier: str,
                            inputs: dict,
                            log_path: str = CROSS_SHADOW_LOG_PATH) -> None:
    """Append a shadow-log record to JSONL. Caller-side I/O (v3.7.240).

    Args:
        decision: 返回自 ``select_gld_sync_strategy``.
        signal_date: SLV-S 触发日.
        slv_tier: 触发的 SLV tier (通常 'S' 或 'A').
        inputs: 选择器 inputs 的字典 (bp_low, gvz_value, gvz_asof_date),
            用于事后审计.
        log_path: 输出 JSONL 路径.

    Each line is a self-contained JSON record. Append-only, line-buffered
    flush. Safe for concurrent appends because each line is whole.
    """
    import json
    from pathlib import Path
    rec = {
        "signal_date": pd.Timestamp(signal_date).normalize().isoformat(),
        "slv_tier": slv_tier,
        "decision": decision,
        "inputs": {
            "bp_low": (None if inputs.get("bp_low") is None
                        else float(inputs["bp_low"])),
            "gvz_value": (None if inputs.get("gvz_value") is None
                           else float(inputs["gvz_value"])),
            "gvz_asof_date": (None if inputs.get("gvz_asof_date") is None
                               else pd.Timestamp(inputs["gvz_asof_date"])
                                       .normalize().isoformat()),
        },
        "written_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def live_cutover_allowed(today,
                            log_path: str = CROSS_SHADOW_LOG_PATH,
                            min_days: int = CROSS_LIVE_CUTOVER_MIN_DAYS) -> tuple:
    """Check whether shadow log has accumulated ≥ ``min_days`` calendar days.

    Args:
        today: 当前评估日 (pd.Timestamp).
        log_path: JSONL log location.
        min_days: 最小累积天数 (默认 14).

    Returns:
        ``(allowed: bool, first_record_at: Optional[str], days_accumulated: int)``
    """
    import json
    from pathlib import Path
    today = pd.Timestamp(today).normalize()
    p = Path(log_path)
    if not p.exists():
        return (False, None, 0)
    first = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                d = pd.Timestamp(rec.get("written_at",
                                              rec.get("signal_date")))
                # 统一为 tz-naive 后比较, 避免 jsonl 含 tz-aware
                # written_at 时与 caller-supplied tz-naive today 直接相减失败.
                if d.tzinfo is not None:
                    d = d.tz_convert("UTC").tz_localize(None)
                d = d.normalize()
                if first is None or d < first:
                    first = d
    except Exception:
        return (False, None, 0)
    if first is None:
        return (False, None, 0)
    days = (today - first).days
    return (days >= min_days, first.isoformat(), days)


# v3.7.240: legacy alias kept for one release; new code should import the
# selector and call it through the caller-side write path.
CROSS_STRATEGY = "BUY CALL"      # deprecated default; subject to selector override


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
