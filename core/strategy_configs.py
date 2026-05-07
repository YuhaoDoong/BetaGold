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


# ── 期货 (v3.7.169 grid-validated 5y COMEX) ──
# 5y grid (n=151 GLD GC=F / n=230 SLV SI=F, source=comex 2020-04 至今):
#   GLD lev=20×:
#     现行 TP200/SL50/h15:  wr=63% sum=+2952% sharpe=+0.31
#     最优 TP150/SL100/h20: wr=82% sum=+4867% sharpe=+0.51 (+1900%)
#   SLV lev=10×:
#     现行 TP200/SL50/h15:  wr=56% sum=+1101% sharpe=+0.086
#     最优 (10× SL100 h20): wr=73% sum=+2600% sharpe=+0.18 (+1500%)
#     绝对最优 lev15×: wr=65% sum=+3395% (但爆仓风险 ↑)
# 1y grid (n=15-30) 与 5y 反向 — 短期噪声不可信. 5y 大样本主导决策.
# 核心结论: SL=100% margin (≈ liq-only) 在两个 asset 都明显优于紧 SL.
#          SL 太紧噪声打止损, 错过黄金/白银长期趋势.
#          hold 20d > 15d (给趋势时间走完)
FUTURES_GLD = FuturesConfig(
    leverage=20,
    tp_margin_pct=150.0,    # v3.7.169: 200→150% (5y grid 略优)
    sl_margin_pct=100.0,    # v3.7.169: 50→100% margin (-5% spot, 等价 liq-only)
                              # 5y wr 提升 19pp (63→82%)
    hold_max_days=20,       # v3.7.169: 15→20d (黄金趋势走得久)
    early_tp_locks=(
        (3, 5.0),    # 3d ≥ +5% spot (+100% margin) 锁利
        (7, 3.0),    # 7d ≥ +3% spot (+60% margin)
        (12, 1.0),   # 12d ≥ +1% spot (+20% margin)
    ),
)
FUTURES_SLV = FuturesConfig(
    leverage=10,
    tp_margin_pct=200.0,    # v3.7.169: 保 200% (TP100 没明显优势)
    sl_margin_pct=100.0,    # v3.7.169: 50→100% margin (= 等价 liq-only)
                              # 5y wr 提升 17pp (56→73%)
    hold_max_days=20,       # v3.7.169: 15→20d
    early_tp_locks=(
        (3, 5.0),    # 3d ≥ +5% spot (+50% margin)
        (7, 3.0),    # 7d ≥ +3% spot (+30% margin)
        (12, 1.0),   # 12d ≥ +1% spot (+10% margin)
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
