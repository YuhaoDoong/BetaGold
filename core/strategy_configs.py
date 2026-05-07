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


# ── 期货 (v3.7.168 grid-validated) ──
# 近 1y grid 结果 (n=15 GLD / n=30 SLV):
#   GLD: SL=50%→sum -221% (wr 33%); SL=75%→sum +47% (wr 47%); SL=100% 同
#       hold=10>15>20 (热信号尽快锁利)
#   SLV: SL=50%→sum -528% (wr 33%); SL=100%→sum +3% (wr 60%); TP=100%>200%
# 解读: SL=50% margin 在波动期被噪声打止损, 然后趋势继续我们错过.
#       SL 放宽 + 早平 (基于 hold/margin) 才是正确机制.
FUTURES_GLD = FuturesConfig(
    leverage=20,
    tp_margin_pct=200.0,    # +200% margin = +10% spot
    sl_margin_pct=75.0,     # v3.7.168: 50→75% margin (-3.75% spot, 给震荡空间)
    hold_max_days=10,       # v3.7.168: 15→10d (热信号尽快锁利)
    early_tp_locks=(
        (3, 5.0),    # 3d ≥ +5% spot (+100% margin) 锁利
        (5, 3.0),    # 5d ≥ +3% spot (+60% margin)
        (7, 1.0),    # 7d ≥ +1% spot (+20% margin)
    ),
)
FUTURES_SLV = FuturesConfig(
    leverage=10,
    tp_margin_pct=100.0,    # v3.7.168: 200→100% margin (+10% spot, grid 偏好近 TP)
    sl_margin_pct=100.0,    # v3.7.168: 50→100% margin (-10% spot, 等价 liq-only;
                              # SL=50%(5% spot) 在 SLV 日波 3-5% 区间会被噪声打)
    hold_max_days=15,
    early_tp_locks=(
        (3, 5.0),    # 3d ≥ +5% spot (+50% margin)
        (7, 3.0),    # 7d ≥ +3% spot (+30% margin)
        (10, 1.0),   # 10d ≥ +1% spot (+10% margin)
    ),
)

# ── SHORT_VOL Iron Condor ──
SHORT_VOL_DEFAULT = ShortVolConfig(
    profit_target_credit_pct=50.0,
    stop_loss_pct=50.0,
    hold_max_days=30,
    base_dte=30,
)

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
