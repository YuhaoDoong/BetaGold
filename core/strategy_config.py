"""可调节参数集中管理 — 支持 per-asset 校准.

全局精度规范 (v3.7.33):
  RV %tile 网格步长统一 0.01 (跨方向性 / SHORT_VOL / STRADDLE 三处)
  绝对 RV 步长 1% (整数百分比)
  Score 步长 1 (整数)



设计目标:
  1. 集中: 所有策略可调阈值放一处, 方便 grid search / 定期重测
  2. Per-asset: 每个资产有独立配置 (GLD / SLV / 未来 QQQ 等)
  3. 默认兜底: 资产没配置时 fallback 到 DEFAULT
  4. 加载方式: 各 strategy 模块用 get_config(asset).<param>

每次定期重测后, 只需改这一个文件即可全局生效.

Re-tune cadence (建议):
  - 每月跑一次 grid search (scripts/tune_thresholds.py)
  - 重大市场变化 (regime 切换, 新资产) 立即重跑
  - 参数变化 > 10% 再考虑切换 (避免噪音)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AssetConfig:
    """单个资产的全部可调阈值."""

    # ── 方向性策略 (BUY CALL / SELL PUT) ──
    buy_bp: float = 0.30                # 买入开窗 bp_low 阈值
    exit_bp: float = 0.90               # BandExit 退出 bp_high 阈值
    stop_loss_pct: float = 3.0          # 单笔止损 %
    pullback_gain: float = 2.0          # Pullback 启动峰值涨幅
    pullback_dd: float = 1.5            # Pullback 启动回撤 %
    consecutive_stop: int = 99          # 连续止损熔断 (99 = 实质禁用)
    max_hold_days: int = 30             # 持仓上限 (Timeout 安全帽)

    # ── RV 极值过滤 (核心: 方向性 BUY/SELL 入场过滤) ──
    rv_filter_enabled: bool = True
    rv_filter_low: float = 0.50         # < 此值 → BUY CALL
    rv_filter_high: float = 0.80        # > 此值 → SELL PUT

    # ── STRADDLE 做多波动率 ──
    straddle_rv_threshold: float = 20.0   # RV < 此值 +2 分
    straddle_rv_abs_max: float = 25.0     # RV > 此值不触发
    straddle_event_days: int = 3          # 距事件 ≤ 此天数加分
    straddle_rv_drop_pct: float = 30.0    # RV 相对均值下降 > 此 %
    straddle_rv_pctile_max: float = 0.50  # v3.7.32: RV %tile > 此值不入场
    straddle_hold_days: int = 5
    straddle_priority_score: int = 6      # ≥ 此分单走 STRADDLE

    # ── SHORT_VOL Iron Condor 做空波动率 ──
    short_vol_rv_pctile_lo: float = 0.45
    short_vol_rv_pctile_hi: float = 0.80
    short_vol_rv_abs_min: float = 13.0
    short_vol_rv_abs_max: float = 28.0
    short_vol_fomc_buffer: int = 10
    short_vol_nfp_buffer: int = 7
    short_vol_opex_buffer: int = 5
    short_vol_score_trigger: int = 7
    short_vol_strike_sigma: float = 1.6
    short_vol_wing_sigma: float = 3.0
    short_vol_premium_ratio: float = 0.40
    short_vol_priority_score: int = 6

    # ── MIXED 组合 ──
    vol_dir_both_strong: int = 4

    # ── 元数据 (方便追溯) ──
    last_tuned: str = ""               # 最后一次 grid search 日期 ISO
    tune_period_days: int = 5*365      # grid search 用的回测窗口
    notes: str = ""


# ── 默认基线 (与 GLD v3.7.29 一致) ──
DEFAULT_CONFIG = AssetConfig(
    last_tuned="2026-04-29",
    notes="GLD-derived defaults from v3.7.29 grid search",
)


# ── 各资产专属配置 ──
# 校准来源: 5y 全量 grid search (步长 0.025)
# 详见 docs/EXPERIMENTS.md §13-14, /tmp/rv_grid_*.csv

ASSET_CONFIGS: Dict[str, AssetConfig] = {

    # GLD: v3.7.29 网格搜索最优
    # 5y 数据, 38 笔 BUY/SELL, 胜 82%, 总 +48.9%, Sharpe 0.638
    "GLD": AssetConfig(
        rv_filter_enabled=False,        # v3.7.39: 真实期权 P&L 验证, 关 RV 过滤
        rv_filter_low=0.50, rv_filter_high=0.85,  # 兜底 (rv_filter_enabled=False 不用)
        short_vol_rv_pctile_lo=0.45,    # IC 保留, 没新数据
        short_vol_rv_pctile_hi=0.80,
        straddle_rv_pctile_max=1.00,    # v3.7.39: 移除 STRADDLE 过滤 (屏蔽 +141.8%)
        last_tuned="2026-04-30",
        notes="v3.7.39 真实期权 P&L 验证 - 方向性 RV 过滤反向有害, 关闭",
    ),

    # SLV: v3.7.30 SLV 单独 grid search
    # 方向性 5y, 0.50/0.75: 73 笔 81% +99.9% Sharpe 0.490
    # SHORT_VOL 5y, 0.25/0.775: 77 笔 88% +73.7% Sharpe 0.848
    # 与 GLD 显著不同 — SLV 笔数翻倍, 单笔波动更大
    "SLV": AssetConfig(
        rv_filter_enabled=False,        # v3.7.39: 真实数据 SLV BUY CALL 无过滤 +265%
        rv_filter_low=0.50, rv_filter_high=0.85,  # 兜底
        short_vol_rv_pctile_lo=0.25,
        short_vol_rv_pctile_hi=0.775,
        straddle_rv_pctile_max=1.00,    # 移除 (SLV STRADDLE 整体 Sharpe 差)
        last_tuned="2026-04-30",
        notes="v3.7.39 真实期权 P&L 验证 — RV 过滤反向, 关闭",
    ),

    # 未来扩展示例 (留位):
    # "QQQ": AssetConfig(...),
    # "SPY": AssetConfig(...),
}


def get_config(asset: str) -> AssetConfig:
    """根据资产返回对应配置, 找不到则用默认."""
    return ASSET_CONFIGS.get(asset.upper(), DEFAULT_CONFIG)


def list_assets() -> list:
    """已校准资产列表."""
    return list(ASSET_CONFIGS.keys())


def tunable_params() -> list:
    """所有可调参数名 (用于 grid search 脚本)."""
    return [f.name for f in AssetConfig.__dataclass_fields__.values()
            if f.name not in ("last_tuned", "tune_period_days", "notes")]


# 全局网格搜索精度 (v3.7.33+ 统一标准)
GRID_PRECISION = {
    "rv_pctile": 0.01,    # RV %tile 网格步长 (方向性/STRADDLE/SHORT_VOL 全用)
    "rv_abs": 1.0,        # 绝对 RV 步长 (%)
    "score": 1,           # Score 阈值步长
    "bp": 0.025,          # Band position 步长
}

