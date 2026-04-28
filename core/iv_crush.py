"""IV Crush 测量与预测 — 用真实 GVZ 数据驱动期权策略风控.

数据源:
  FRED CBOE Gold ETF Volatility Index (GVZCLS), 2009-09-18 至今
  Series: features['gvz'] (已加载到本系统)

经验值 vs 实证 (GLD GVZ, 2025-2026):
                 业内通用值     GLD 实证 mean   GLD 实证 median
  FOMC           30% (SPX)      -0.1%           +0.1%
  NFP            15%             -2.8%           -2.8%
  OPEX           10%             -5.4%           -0.6%

GLD 与 SPX/VIX 截然不同 — 黄金对 FOMC 反应延迟 (看实际利率/美元),
IV 不会立即衰减. 因此本系统用实证值, 不照搬 SPX 经验.

按 pre-event GVZ 水平分组:
  低 IV (pre < 中位): FOMC 后反而 +2.1% (IV 不降反升)
  高 IV (pre ≥ 中位): FOMC 后轻微 -2.2% crush

判断 crush 风险的核心指标 — IV/RV ratio:
  > 1.5: 显著事件溢价 → crush 风险高
  1.2-1.5: 中等溢价
  < 1.2: 几乎无 crush 风险
"""
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd


# 实证均值 (基于 2025-2026 真实 GVZ 数据)
IV_CRUSH_EMPIRICAL_MEAN: Dict[str, float] = {
    "FOMC": -0.001,   # 几乎 0
    "NFP": -0.028,
    "OPEX": -0.054,
}

# 保守上限 (考虑高 IV 状态下可能出现的 crush)
IV_CRUSH_CONSERVATIVE: Dict[str, float] = {
    "FOMC": 0.05,
    "NFP": 0.05,
    "OPEX": 0.05,
}

# IV/RV 比率阈值 (用于 crush 风险标签)
RATIO_HIGH_THRESHOLD = 1.5      # > 1.5 → 显著溢价
RATIO_MEDIUM_THRESHOLD = 1.2    # 1.2-1.5 → 中等
# < 1.2 → 无显著溢价


def iv_rv_ratio(gvz: pd.Series, rv_10d: pd.Series) -> pd.Series:
    """计算 IV/RV ratio (GVZ / RV 10日年化).

    > 1.5: 期权市场为事件加价 (event premium baked in)
    ≈ 1.0: IV 与实际波动一致
    < 1.0: IV 反而低于实际, crush 不会发生
    """
    return gvz / rv_10d


def crush_risk_label(ratio: float) -> Tuple[str, str]:
    """从 IV/RV ratio 返回风险等级和说明.

    Returns:
        (level, description): level ∈ {"高", "中", "低"}
    """
    if ratio is None or np.isnan(ratio):
        return ("?", "数据缺失")
    if ratio > RATIO_HIGH_THRESHOLD:
        return ("高", f"IV/RV={ratio:.2f}, 显著事件溢价, crush 风险高")
    if ratio > RATIO_MEDIUM_THRESHOLD:
        return ("中", f"IV/RV={ratio:.2f}, 中等溢价")
    return ("低", f"IV/RV={ratio:.2f}, 无显著溢价, crush 概率低")


def measure_event_iv_crush(gvz: pd.Series,
                             event_dates: List[pd.Timestamp],
                             event_name: str = "Event"
                             ) -> Dict[str, float]:
    """对一组事件日, 计算 GVZ 在事件前后的相对变化.

    Args:
        gvz: GVZ 时间序列 (可来自 features['gvz'])
        event_dates: 事件日列表 (pd.Timestamp)
        event_name: 标签

    Returns:
        统计 dict: n / mean / median / std / pre_mean / post_mean /
                  pct_significant_crush / pct_inverse (反向上升)
    """
    drops = []
    pre_levels = []
    post_levels = []
    detail = []

    for ev in event_dates:
        ev_ts = pd.Timestamp(ev)
        if ev_ts not in gvz.index:
            continue
        idx = gvz.index.get_loc(ev_ts)
        if idx < 1 or idx >= len(gvz) - 1:
            continue
        pre = gvz.iloc[idx - 1]
        post = gvz.iloc[idx + 1]
        if pre <= 0 or post <= 0:
            continue
        rel_drop = (pre - post) / pre
        drops.append(rel_drop)
        pre_levels.append(pre)
        post_levels.append(post)
        detail.append({
            "date": ev_ts, "pre": pre, "on_event": gvz.iloc[idx],
            "post": post, "rel_drop": rel_drop,
        })

    arr = np.array(drops) if drops else np.array([])
    if len(arr) == 0:
        return {
            "event": event_name, "n": 0,
            "mean": np.nan, "median": np.nan,
            "pre_mean": np.nan, "post_mean": np.nan,
            "details": [],
        }

    return {
        "event": event_name,
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "pre_mean": float(np.array(pre_levels).mean()),
        "post_mean": float(np.array(post_levels).mean()),
        "pct_significant_crush": float(sum(1 for d in arr if d > 0.05) / len(arr)),
        "pct_inverse": float(sum(1 for d in arr if d < 0) / len(arr)),
        "details": detail,
    }


def predict_event_crush(d: pd.Timestamp,
                          gvz: pd.Series,
                          rv: pd.Series,
                          event_type: str,
                          mode: str = "dynamic") -> float:
    """预测某日触发的事件 crush 风险 (用于实时风控).

    Args:
        d: 当前日期
        gvz: GVZ 时间序列
        rv: RV 10日年化时间序列
        event_type: 'FOMC' / 'NFP' / 'OPEX'
        mode: 'dynamic' 按 IV/RV 比率动态调整,
              'conservative' 取保守上限,
              'empirical' 取历史均值

    Returns:
        预计 IV crush 比例 (0-1, e.g. 0.05 = 5%)
    """
    if mode == "conservative":
        return IV_CRUSH_CONSERVATIVE.get(event_type, 0.05)

    if mode == "empirical":
        return abs(IV_CRUSH_EMPIRICAL_MEAN.get(event_type, 0.0))

    # dynamic: 基于当前 IV/RV ratio 调整
    if d not in gvz.index or d not in rv.index:
        return IV_CRUSH_CONSERVATIVE.get(event_type, 0.05)

    g = gvz.get(d, np.nan)
    r = rv.get(d, np.nan)
    if np.isnan(g) or np.isnan(r) or r <= 0:
        return IV_CRUSH_CONSERVATIVE.get(event_type, 0.05)

    ratio = g / r
    base = IV_CRUSH_CONSERVATIVE.get(event_type, 0.05)

    # 比率越高, crush 越大 (按线性外推)
    if ratio > RATIO_HIGH_THRESHOLD:
        return base * 2.0      # 显著溢价: 2x 基线 (~10%)
    if ratio > RATIO_MEDIUM_THRESHOLD:
        return base * 1.0      # 中等: 1x 基线 (~5%)
    return base * 0.3          # 低溢价: 0.3x (~1.5%, 接近实证 -0.1%)


def event_crush_summary(gvz: pd.Series, rv_10d: pd.Series,
                          event_dates_by_type: Dict[str, list]
                          ) -> Dict[str, Dict]:
    """生成所有事件类型的 IV crush 统计概览.

    Args:
        gvz: GVZ 序列
        rv_10d: RV 序列
        event_dates_by_type: {'FOMC': [d1, d2, ...], 'NFP': [...], ...}

    Returns:
        {event_type: stats_dict} 同 measure_event_iv_crush 输出
    """
    return {
        ev: measure_event_iv_crush(gvz, dates, ev)
        for ev, dates in event_dates_by_type.items()
    }
