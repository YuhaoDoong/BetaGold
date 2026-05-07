"""集中管理所有策略 TP/SL/Hold/Leverage 参数 (v3.7.167).

设计原则:
  1. 单一来源 — 修改参数只在这里
  2. Per-asset 可覆盖 — GLD vs SLV 波动差异
  3. 各策略 dataclass 透明导出, 便于回测/dashboard 引用同一实例

用法:
    from core.strategy_configs import get_config
    cfg = get_config(asset='SLV', strategy='FUTURES_LONG')
    cfg.leverage  # → 10
    cfg.sl_margin_pct  # → 50.0
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from core.strategies.futures_long import FuturesConfig
from core.strategies.short_vol import ShortVolConfig
from core.strategies.sell_put import SPConfig
from core.strategies.buy_call import BCConfig
from core.strategies.straddle import StraddleConfig


# ── 期货 (v3.7.170 WR-first grid 5y COMEX) ──
# 评分指标 scoreB = WR² × log(1+n) × avg (高杠杆首要 WR, 频率 log 防过拟合)
# 决策: WR ≥ 75% 内挑 max scoreB (用户"确保胜率"约束)
#
# 5y grid (n=151 GLD GC=F / n=230 SLV SI=F, 2020-04 至今):
#
# GLD 候选:
#   现行 20× TP200/SL50/h15:    wr=63% sum=+2952% scoreB=+39
#   纯 sum 20× TP150/SL100/h20: wr=82% sum=+4867% scoreB=+107 (高 lev 高 avg)
#   纯 WR  10× TP100/SL100/h20: wr=85% sum=+2645% scoreB=+63
#   选: 10× TP200/SL100/h20 (WR≥75% + lev 适中 + 不限 TP 让大涨发挥)
#
# SLV 候选:
#   现行 10× TP200/SL50/h15:   wr=56% sum=+1101% scoreB=+8
#   纯 sum 15× TP200/SL100/h20: wr=65% sum=+3398% scoreB=+34 (但 SLV 15× 危险)
#   纯 WR   5× TP150/SL100/h20: wr=76% sum=+1600% scoreB=+22
#   选: 5× TP200/SL100/h20 (WR ≥ 75% + 5× 安全 + 用户可加仓加大暴露)
#
# 用户原则: "胜率重要, 收益率通过仓位变化". 默认参数保 WR, 仓位灵活.
# SL=100% margin 在两 asset 均最优 (= 等价 liq-only, 避免被噪声打)
# hold=20d 给金/银长期趋势走完
# v3.7.174 — wick-safe leverage:
# 3 月实测 GC=F 单日 wick (Open→Low):
#   3-19 -5.72%   3-23 -5.79%   收盘后大都回拉 -2~-4%
# lev=10× 爆仓距离 = 1/lev - mm_rate ≈ 5% spot → 单日 wick 必爆
# lev=5×  爆仓距离 ≈ 19.5% spot → 5-8% wick 安全, 收盘回拉时持仓存活
# lev=3×  爆仓距离 ≈ 33% spot → 极端 wick 也安全
# SP 期权按 expiry 日线 close 计 mark, wick 不影响 → 解释 3-19/23 SP 盈利但 FUT 爆仓
FUTURES_GLD = FuturesConfig(
    leverage=5,             # v3.7.174: 10→5× (爆仓 5%→19.5% spot, wick 安全)
    tp_margin_pct=200.0,
    sl_margin_pct=100.0,    # = liq-only (爆仓即出)
    hold_max_days=20,
    early_tp_locks=(
        (3, 10.0),   # 3d ≥ +10% spot (+50% margin @5×) 锁利
        (7, 6.0),    # 7d ≥ +6% spot (+30% margin)
        (12, 2.0),   # 12d ≥ +2% spot (+10% margin)
    ),
)
FUTURES_SLV = FuturesConfig(
    leverage=3,             # v3.7.174: 5→3× (银日波 3-5%, 爆仓 19→33% spot)
    tp_margin_pct=200.0,
    sl_margin_pct=100.0,
    hold_max_days=20,
    early_tp_locks=(
        (3, 15.0),   # 3d ≥ +15% spot (+45% margin @3×) 锁利
        (7, 10.0),   # 7d ≥ +10% spot (+30% margin)
        (12, 4.0),   # 12d ≥ +4% spot (+12% margin)
    ),
)

# ── SHORT_VOL Iron Condor ──
# v3.7.177: 当前实战 WR=6% (n=16), 大波动期失效. 建议停用直到 GVZ < 22 重启.
# 入场逻辑应加: GVZ 低 + RV 低 + 无重大事件日 三条件.
SHORT_VOL_DEFAULT = ShortVolConfig(
    profit_target_credit_pct=50.0,
    stop_loss_pct=50.0,
    hold_max_days=30,
    base_dte=30,
)
SHORT_VOL_DISABLED = True  # ★ v3.7.177: 当前默认停用

# ── SELL_PUT credit spread ──
SELL_PUT_DEFAULT = SPConfig()

# ── BUY_CALL ──
BUY_CALL_DEFAULT = BCConfig()

# ── STRADDLE ──
STRADDLE_DEFAULT = StraddleConfig()


_REGISTRY = {
    ("GLD", "FUTURES_LONG"): FUTURES_GLD,
    ("SLV", "FUTURES_LONG"): FUTURES_SLV,
    ("GLD", "SHORT_VOL"): SHORT_VOL_DEFAULT,
    ("SLV", "SHORT_VOL"): SHORT_VOL_DEFAULT,
    ("GLD", "SELL PUT"): SELL_PUT_DEFAULT,
    ("SLV", "SELL PUT"): SELL_PUT_DEFAULT,
    ("GLD", "BUY CALL"): BUY_CALL_DEFAULT,
    ("SLV", "BUY CALL"): BUY_CALL_DEFAULT,
    ("GLD", "STRADDLE"): STRADDLE_DEFAULT,
    ("SLV", "STRADDLE"): STRADDLE_DEFAULT,
}


def get_config(asset: str, strategy: str):
    """根据 asset + strategy 拉对应 config dataclass.
    若没特化配置返回 None.
    """
    return _REGISTRY.get((asset.upper(), strategy.upper()
                          if strategy != "SHORT_VOL" and strategy != "FUTURES_LONG"
                          else strategy))


def get_futures_config(asset: str) -> FuturesConfig:
    """期货 config 强类型 helper."""
    cfg = _REGISTRY.get((asset.upper(), "FUTURES_LONG"))
    return cfg if cfg else FUTURES_GLD


def summary() -> str:
    """所有策略参数一览 (debug / dashboard)."""
    lines = ["=== 策略参数中心 (v3.7.167) ==="]
    for (asset, strat), cfg in _REGISTRY.items():
        if cfg is None: continue
        if strat == "FUTURES_LONG":
            lines.append(f"  {asset} {strat}: lev={cfg.leverage}× "
                          f"TP=+{cfg.tp_margin_pct}% margin "
                          f"SL=-{cfg.sl_margin_pct}% margin "
                          f"hold≤{cfg.hold_max_days}d")
        elif strat == "SHORT_VOL":
            lines.append(f"  {asset} {strat}: TP=+{cfg.profit_target_credit_pct}% credit "
                          f"SL=-{cfg.stop_loss_pct}% (max_risk) "
                          f"hold≤{cfg.hold_max_days}d DTE={cfg.base_dte}")
        else:
            lines.append(f"  {asset} {strat}: {cfg}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
