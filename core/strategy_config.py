"""可调节参数集中管理 — 支持 per-asset 校准.

全局精度规范 (v3.7.33):
  RV %tile 网格步长统一 0.01 (跨方向性 / SHORT_VOL / STRADDLE 三处)
  绝对 RV 步长 1% (整数百分比)
  Score 步长 1 (整数)

方向性 RV 切点框架 (v3.7.46 实证):
  rv_filter_low/high 是 BUY_CALL ↔ SELL_PUT 切换点 (单切, lo=hi).
  rv_pctile < 切点 → BUY_CALL (低 IV 期权便宜, 直接做多)
  rv_pctile ≥ 切点 → SELL_PUT (高 IV 替代, 收 premium 而非付)
  v3.7.46 测试: 切点之上加 SKIP 屏蔽都使期望胜次下降 — 不加.



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

    # ── v3.7.117: GVZ IV 三阶过滤 (实证 v3.7.116 grid) ──
    # 真实期权数据验证: 高 IV 时方向性几乎全错向 (BC 3.8% wr GLD / 20.7% SLV).
    # 高 IV + 深破 0.10 + 切 SP only → 50-58% wr 保本.
    iv_filter_enabled: bool = True
    iv_filter_low_max: float = 22.0     # GVZ < 此值 = 低 IV, 方向正常
    iv_filter_high_min: float = 28.0    # GVZ > 此值 = 高 IV, 触发深破规则
    iv_high_bp_low_max: float = 0.10    # 高 IV 时 bp_low 必须 ≤ 此值才入场
    iv_high_force_sp: bool = True        # 高 IV 时强制 SELL PUT (BC 全错)
    iv_mid_dual_confirm: bool = True     # 22-28 中 IV 需二次确认 (技术指标 align)

    # ── v3.7.123: SP score 多因子选 BC vs SP (paired-grid 验证) ──
    # 替代单一 RV 切点 — 综合 IV/区间/技术指标决定 BC vs SP.
    # 实证 (paired_grid_multi.py):
    #   RSI < 30 (超卖):     SP wr 100% (n=13, 跨 GLD/SLV) ★最强
    #   IV-RV gap > 0:       SP 优于 BC 跨多区间稳定
    #   bp_close < 0.30:     SP 优 (close 在 band 下方)
    #   GVZ IV >= 28:        SP 优
    #   Stoch %K 40-60:      SP 优
    #   MACD hist 中位空头:  SP 优
    sp_score_enabled: bool = True
    sp_score_threshold: float = 3.0      # score >= 此值 → SELL PUT, 否则 BUY CALL
    sp_score_w_iv_rv_gap: float = 1.5    # iv_rv_gap_pct > 0
    sp_score_w_bp_low_deep: float = 1.0  # bp_low < 0.05
    sp_score_w_bp_close_low: float = 1.0 # bp_close < 0.30
    sp_score_w_gvz_high: float = 1.0     # gvz >= 28
    sp_score_w_rsi_oversold: float = 2.0 # rsi_14 < 30 (最强)
    sp_score_w_stoch_low: float = 0.5    # stoch_k < 40
    sp_score_w_macd_bear: float = 0.5    # macd_hist < -0.5

    # ── v3.7.128: ma_trend (MA20/MA50) 入场过滤 (per-asset grid 最优) ──
    # ma_trend < threshold → buy_signal 跳过 (下行趋势接飞刀概率高)
    # GLD: 0.975 (paired sum +3639% vs 0.99 +3598%, +41% 提升, 5-4 边界信号能通过)
    # SLV: 0.990 (paired sum +934%, 已是 grid 最优)
    ma_trend_filter_enabled: bool = True
    ma_trend_threshold: float = 0.99

    # ── v3.7.201: 信号双因子硬过滤 (GLD 3y 信号深度 grid 验证) ──
    # rv_pctile_max: rv >= 此值跳过 BUY (高波动接飞刀)
    # ret_20d_min: 近20日 spot 跌幅 <= 此值跳过 BUY (暴跌中接飞刀)
    # GLD grid: rv<0.75 + ret>-3% n=82 (砍 43%) WR 70.6→75.6% +5pp, Q1 拦 19/20
    # 默认禁用 (rv_max=1.0 / ret_min=-1.0 即任何 RV/任何 ret 都过), per-asset 开启
    rv_pctile_max_hard: float = 1.0
    ret_20d_min_hard: float = -1.0

    # ── v3.7.202: 信号 S/A/B 三级 tier 标注 (不过滤, 仅打分) ──
    # 用户诉求: 100%胜率阈值下的信号要能区分出来, 让人看到是最优信号
    # GLD 3y grid (signal_filter_deep.py):
    #   S (最优): rv<tier_s_rv AND ret>tier_s_ret AND bp<=tier_s_bp
    #     n=13, WR5d=84.6%, WR20d=100%, mean5d=+1.31%, max_loss5d=-1.5%
    #   A (强): rv<tier_a_rv AND ret>tier_a_ret AND bp<=tier_a_bp (not S)
    #     n=13, WR5d=84.6%, WR20d=92.3%, mean5d=+1.75%, max_loss5d=-1.5%
    #   B (标准): pass hard_filter, neither S nor A (n=56, WR5d=71%, WR20d=88%)
    tier_s_rv_max: float = 0.65
    tier_s_ret_20d_min: float = 0.0
    tier_s_bp_low_max: float = 0.20
    tier_a_rv_max: float = 0.75
    tier_a_ret_20d_min: float = -0.01
    tier_a_bp_low_max: float = 0.20

    # ── STRADDLE 做多波动率 ──
    straddle_rv_threshold: float = 20.0   # RV < 此值 +2 分
    straddle_rv_abs_max: float = 25.0     # RV > 此值不触发
    straddle_event_days: int = 3          # 距事件 ≤ 此天数加分
    straddle_rv_drop_pct: float = 30.0    # RV 相对均值下降 > 此 %
    straddle_rv_pctile_max: float = 0.50  # v3.7.32: RV %tile > 此值不入场
    # v3.7.40: 事件邻近过滤 (实证: 距 FOMC ≤5 天 58% 胜率, >5 天 19%)
    straddle_event_proximity_only: bool = True   # True: 仅 FOMC ≤ N 天才入场
    straddle_event_max_days: int = 5      # 距 FOMC 必须 ≤ 此值才触发
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
        # v3.7.123 — sp_score 替代单切 (paired 验证 acc 42% → 86%, mean +22% → +66%)
        rv_filter_enabled=True,
        rv_filter_low=0.45,              # 兜底 (sp_score 关闭时用)
        rv_filter_high=0.45,
        sp_score_enabled=True,
        sp_score_threshold=3.5,           # paired chosen sum 最大 (thr=3.5 +3279% vs 3.0 +3164%)
        ma_trend_threshold=0.975,          # v3.7.146 实证: 滤 7 笔全输 (-290%累计), 真 alpha
        # v3.7.200: 三 lever grid 验证 — 中高 IV (GVZ 25-28) 漏网导致 5/12-14 BC 全亏 -85%.
        # 28→25 拦 5/12-14 全 3 笔 (10y 总损失 sum -2% / WR +0.1%, 远好于 ma_trend 0.99)
        # 不仅拦 BC, 也让 GVZ 26 深破时强制转 SP (用户提的关键 alpha)
        iv_filter_high_min=25.0,
        # v3.7.201: 信号双因子硬过滤 (signal_filter_deep.py 3y grid 验证)
        # rv<0.75 + ret_20d>-3% → n 143→82 (砍 43%), WR 70.6→75.6%, Q1 拦 19/20
        # 推荐方案 B (用户选): 频率/质量平衡, 季度分布均衡
        rv_pctile_max_hard=0.75,
        ret_20d_min_hard=-0.03,
        short_vol_rv_pctile_lo=0.45,
        short_vol_rv_pctile_hi=0.80,
        straddle_rv_abs_max=30.0,
        straddle_rv_pctile_max=1.00,
        last_tuned="2026-05-15",
        notes="v3.7.201 双因子硬过滤 rv<0.75 + ret_20d>-3% (Q1 拦 19/20, WR +5pp)",
    ),

    # SLV: v3.7.30 SLV 单独 grid search
    # 方向性 5y, 0.50/0.75: 73 笔 81% +99.9% Sharpe 0.490
    # SHORT_VOL 5y, 0.25/0.775: 77 笔 88% +73.7% Sharpe 0.848
    # 与 GLD 显著不同 — SLV 笔数翻倍, 单笔波动更大
    "SLV": AssetConfig(
        # v3.7.123 — sp_score 替代单切 (paired 验证 acc 38% → 68%, sum +113% → +862%)
        rv_filter_enabled=True,
        rv_filter_low=0.75,               # 兜底
        rv_filter_high=0.75,
        sp_score_enabled=True,
        sp_score_threshold=2.5,           # paired chosen sum 最大 (thr=2.5 +862% vs 3.0 +738%)
        ma_trend_threshold=0.0,            # v3.7.146 实证: SLV 0.99 滤 67% 信号换 +17%
                                          # 但 0~0.98 累计几乎一样 (+801 vs +821);
                                          # 设 0 = 不过滤, 月月有信号, 累计 -14% 可接受
        short_vol_rv_pctile_lo=0.25,
        short_vol_rv_pctile_hi=0.775,
        straddle_rv_abs_max=25.0,
        straddle_rv_pctile_max=1.00,
        straddle_priority_score=6,
        last_tuned="2026-05-06",
        notes="v3.7.123 sp_score thr=2.5 (paired 68% acc / 54% wr / +23%/笔)",
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

