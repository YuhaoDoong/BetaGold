"""OI 因子区间修正 — 效果可视化对比.

对比模型原始预测区间 vs OI修正后区间:
  1. Max Pain 引力 (pin effect)
  2. Call Wall 压制上界
  3. Put Wall 支撑下界
  4. 到期临近放大效应
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang HK",
                                "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════
# OI 因子计算
# ══════════════════════════════════════════════════════════

def compute_oi_factors(eod_df, spot, dte_range=(7, 60)):
    """从 EOD 快照计算 OI 因子."""
    df = eod_df[(eod_df["dte"] >= dte_range[0]) &
                (eod_df["dte"] <= dte_range[1]) &
                (eod_df["option_open_interest"] > 0)].copy()

    calls = df[df["option_type"] == "CALL"]
    puts = df[df["option_type"] == "PUT"]

    call_oi = calls.groupby("option_strike_price")["option_open_interest"].sum()
    put_oi = puts.groupby("option_strike_price")["option_open_interest"].sum()

    # Max Pain
    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    pain = []
    for k in all_strikes:
        c_pain = sum(max(k - s, 0) * oi for s, oi in call_oi.items())
        p_pain = sum(max(s - k, 0) * oi for s, oi in put_oi.items())
        pain.append((k, c_pain + p_pain))
    max_pain = min(pain, key=lambda x: x[1])[0]

    # Call Wall / Put Wall (OI 最大的 strike)
    call_wall = float(call_oi.idxmax()) if len(call_oi) > 0 else spot
    put_wall = float(put_oi.idxmax()) if len(put_oi) > 0 else spot

    # Gamma Exposure 估算 (近 ATM ±5% 范围)
    atm_range = (spot * 0.95, spot * 1.05)
    near_atm = df[(df["option_strike_price"] >= atm_range[0]) &
                   (df["option_strike_price"] <= atm_range[1])]
    # 做市商通常 short call + short put → gamma 来自 OI
    # Net GEX > 0 → long gamma (压制波动)
    # 简化: call gamma OI - put gamma OI (正=long gamma)
    call_gex = near_atm[near_atm["option_type"] == "CALL"].apply(
        lambda r: r["option_gamma"] * r["option_open_interest"] * 100, axis=1
    ).sum() if len(near_atm) > 0 else 0
    put_gex = near_atm[near_atm["option_type"] == "PUT"].apply(
        lambda r: r["option_gamma"] * r["option_open_interest"] * 100, axis=1
    ).sum() if len(near_atm) > 0 else 0

    # 最近到期 DTE
    nearest_dte = int(df["dte"].min()) if len(df) > 0 else 30

    return {
        "max_pain": max_pain,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_gex": call_gex,
        "put_gex": put_gex,
        "net_gex": call_gex - put_gex,
        "nearest_dte": nearest_dte,
        "call_oi": call_oi,
        "put_oi": put_oi,
    }


# ══════════════════════════════════════════════════════════
# 区间修正
# ══════════════════════════════════════════════════════════

def adjust_range(upper_price, lower_price, spot, oi):
    """用 OI 因子修正预测区间.

    Returns: (adj_upper, adj_lower, details_dict)
    """
    max_pain = oi["max_pain"]
    call_wall = oi["call_wall"]
    put_wall = oi["put_wall"]
    nearest_dte = oi["nearest_dte"]
    net_gex = oi["net_gex"]

    # 到期临近因子: DTE 越小, 效应越强 (7天→1.0, 30天→0.3, 60天→0.1)
    expiry_factor = np.clip(1.0 - (nearest_dte - 7) / 30, 0.1, 1.0)

    details = {
        "max_pain": max_pain,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "nearest_dte": nearest_dte,
        "expiry_factor": expiry_factor,
    }

    adj_upper = upper_price
    adj_lower = lower_price

    # 1. Max Pain 引力 — 把区间中心拉向 max pain
    #    如果 spot > max_pain, 上界被压 (上涨更难)
    #    如果 spot < max_pain, 下界被托 (下跌更难)
    mp_dist_pct = (spot - max_pain) / spot  # 正值=spot在max pain上方
    gravity_strength = 0.15 * expiry_factor  # 最大修正 15%
    gravity = mp_dist_pct * gravity_strength

    if mp_dist_pct > 0:
        # spot 在 max pain 上方 → 压缩上界
        adj_upper = upper_price * (1 - gravity)
        details["mp_upper_adj"] = (adj_upper - upper_price)
    else:
        # spot 在 max pain 下方 → 抬升下界
        adj_lower = lower_price * (1 - gravity)  # gravity负, 所以抬升
        details["mp_lower_adj"] = (adj_lower - lower_price)

    # 2. Call Wall 压制 — 如果 call_wall 低于模型上界, 压缩上界
    if call_wall < adj_upper:
        blend = 0.3 * expiry_factor  # 30% 权重给 call wall
        adj_upper = adj_upper * (1 - blend) + call_wall * blend
        details["cw_adj"] = True
    else:
        details["cw_adj"] = False

    # 3. Put Wall 支撑 — 如果 put_wall 高于模型下界, 抬升下界
    if put_wall > adj_lower:
        blend = 0.2 * expiry_factor  # 20% 权重给 put wall
        adj_lower = adj_lower * (1 - blend) + put_wall * blend
        details["pw_adj"] = True
    else:
        details["pw_adj"] = False

    # 4. Gamma 效应 — long gamma 压缩区间, short gamma 扩大
    if net_gex > 0:
        # Long gamma → 压缩 (做市商逆势对冲)
        gamma_compress = min(net_gex / 1e6 * 0.002, 0.03) * expiry_factor
        mid = (adj_upper + adj_lower) / 2
        adj_upper = mid + (adj_upper - mid) * (1 - gamma_compress)
        adj_lower = mid + (adj_lower - mid) * (1 - gamma_compress)
        details["gamma_effect"] = f"long gamma, 压缩{gamma_compress*100:.1f}%"
    elif net_gex < 0:
        # Short gamma → 扩大 (做市商顺势追)
        gamma_expand = min(abs(net_gex) / 1e6 * 0.002, 0.03) * expiry_factor
        mid = (adj_upper + adj_lower) / 2
        adj_upper = mid + (adj_upper - mid) * (1 + gamma_expand)
        adj_lower = mid + (adj_lower - mid) * (1 + gamma_expand)
        details["gamma_effect"] = f"short gamma, 扩大{gamma_expand*100:.1f}%"
    else:
        details["gamma_effect"] = "中性"

    details["adj_upper"] = adj_upper
    details["adj_lower"] = adj_lower
    details["upper_change"] = (adj_upper / upper_price - 1) * 100
    details["lower_change"] = (adj_lower / lower_price - 1) * 100

    return adj_upper, adj_lower, details


# ══════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════

def main():
    from core.data import (load_config, load_gld, load_oos_predictions,
                           load_latest_eod_snapshot)
    from core.signals import build_band

    cfg = load_config()
    gld = load_gld(cfg)
    range_df = load_oos_predictions(cfg)
    eod_df, snap_date = load_latest_eod_snapshot(cfg)

    close = gld["Close"]
    last_date = close.index[-1]
    spot = float(close.iloc[-1])

    print(f"Spot: ${spot:.2f}  Date: {last_date.date()}")
    print(f"EOD Snapshot: {snap_date}")
    print()

    # 1. 原始模型预测
    if last_date in range_df.index:
        pred_u_pct = range_df.loc[last_date, "pred_upper_pct"]
        pred_l_pct = range_df.loc[last_date, "pred_lower_pct"]
    else:
        pred_u_pct, pred_l_pct = 3.0, -2.0  # fallback

    orig_upper = spot * (1 + pred_u_pct / 100)
    orig_lower = spot * (1 + pred_l_pct / 100)

    # 2. 原始 Hybrid Band
    upper_band, lower_band, bp = build_band(
        range_df, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    band_upper = float(upper_band.iloc[-1]) if len(upper_band) > 0 else orig_upper
    band_lower = float(lower_band.iloc[-1]) if len(lower_band) > 0 else orig_lower

    # 3. OI 因子
    oi = compute_oi_factors(eod_df, spot)

    print(f"{'═' * 60}")
    print(f"OI 因子")
    print(f"{'═' * 60}")
    print(f"  Max Pain:   ${oi['max_pain']:.0f} "
          f"({(oi['max_pain']/spot-1)*100:+.1f}% from spot)")
    print(f"  Call Wall:  ${oi['call_wall']:.0f} "
          f"({(oi['call_wall']/spot-1)*100:+.1f}%)")
    print(f"  Put Wall:   ${oi['put_wall']:.0f} "
          f"({(oi['put_wall']/spot-1)*100:+.1f}%)")
    print(f"  Net GEX:    {oi['net_gex']:,.0f} "
          f"({'long' if oi['net_gex']>0 else 'short'} gamma)")
    print(f"  Nearest DTE: {oi['nearest_dte']}d")

    # 4. 修正 5日预测区间
    adj_upper, adj_lower, det = adjust_range(
        orig_upper, orig_lower, spot, oi)

    print(f"\n{'═' * 60}")
    print(f"5日预测区间修正")
    print(f"{'═' * 60}")
    print(f"  原始上界:  ${orig_upper:.2f} (+{pred_u_pct:.1f}%)")
    print(f"  修正上界:  ${adj_upper:.2f} ({det['upper_change']:+.1f}%)")
    print(f"  原始下界:  ${orig_lower:.2f} ({pred_l_pct:.1f}%)")
    print(f"  修正下界:  ${adj_lower:.2f} ({det['lower_change']:+.1f}%)")
    print(f"  Call Wall 压制: {'是' if det['cw_adj'] else '否'}")
    print(f"  Put Wall 支撑:  {'是' if det['pw_adj'] else '否'}")
    print(f"  Gamma 效应:     {det['gamma_effect']}")
    print(f"  到期因子:       {det['expiry_factor']:.2f} "
          f"(DTE={det['nearest_dte']})")

    # 5. 修正 Hybrid Band
    adj_band_upper, adj_band_lower, det_band = adjust_range(
        band_upper, band_lower, spot, oi)

    print(f"\n{'═' * 60}")
    print(f"Hybrid Band 修正")
    print(f"{'═' * 60}")
    print(f"  原始 Band:  ${band_lower:.2f} ~ ${band_upper:.2f}")
    print(f"  修正 Band:  ${adj_band_lower:.2f} ~ ${adj_band_upper:.2f}")
    orig_bp = (spot - band_lower) / (band_upper - band_lower) \
        if band_upper != band_lower else 0.5
    adj_bp = (spot - adj_band_lower) / (adj_band_upper - adj_band_lower) \
        if adj_band_upper != adj_band_lower else 0.5
    print(f"  原始 bp:    {orig_bp:.3f}")
    print(f"  修正 bp:    {adj_bp:.3f}")

    # 修正后的买入/平仓阈值
    adj_bp030 = adj_band_lower + 0.30 * (adj_band_upper - adj_band_lower)
    adj_bp090 = adj_band_lower + 0.90 * (adj_band_upper - adj_band_lower)
    orig_bp030 = band_lower + 0.30 * (band_upper - band_lower)
    orig_bp090 = band_lower + 0.90 * (band_upper - band_lower)

    print(f"\n  买入阈值 (bp<0.30):")
    print(f"    原始: ${orig_bp030:.2f}")
    print(f"    修正: ${adj_bp030:.2f}")
    print(f"  平仓阈值 (bp>0.90):")
    print(f"    原始: ${orig_bp090:.2f}")
    print(f"    修正: ${adj_bp090:.2f}")

    # ══════════════════════════════════════════════════════════
    # 画图
    # ══════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"OI 因子区间修正效果  |  GLD ${spot:.2f}  |  "
                 f"EOD {snap_date}", fontsize=14, fontweight="bold")

    # Panel 1: OI 分布 + 关键价位
    ax = axes[0, 0]
    call_oi = oi["call_oi"]
    put_oi = oi["put_oi"]
    # 只画 spot ±15% 范围
    plot_range = (spot * 0.88, spot * 1.15)
    c_plot = call_oi[(call_oi.index >= plot_range[0]) &
                      (call_oi.index <= plot_range[1])]
    p_plot = put_oi[(put_oi.index >= plot_range[0]) &
                     (put_oi.index <= plot_range[1])]

    ax.bar(c_plot.index, c_plot.values, width=0.8, color="green",
           alpha=0.5, label="Call OI")
    ax.bar(p_plot.index, -p_plot.values, width=0.8, color="red",
           alpha=0.5, label="Put OI")

    ax.axvline(spot, color="black", lw=2, ls="-", label=f"Spot ${spot:.0f}")
    ax.axvline(oi["max_pain"], color="orange", lw=2, ls="--",
               label=f"Max Pain ${oi['max_pain']:.0f}")
    ax.axvline(oi["call_wall"], color="green", lw=2, ls=":",
               label=f"Call Wall ${oi['call_wall']:.0f}")
    ax.axvline(oi["put_wall"], color="red", lw=2, ls=":",
               label=f"Put Wall ${oi['put_wall']:.0f}")

    ax.set_title("OI 分布 + 关键价位", fontsize=12, fontweight="bold")
    ax.set_xlabel("Strike ($)")
    ax.set_ylabel("Open Interest")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: 5日预测区间对比
    ax = axes[0, 1]
    labels = ["模型原始", "OI修正后"]
    uppers = [orig_upper, adj_upper]
    lowers = [orig_lower, adj_lower]
    colors = ["#2196F3", "#FF9800"]

    for i, (lab, u, l, c) in enumerate(
            zip(labels, uppers, lowers, colors)):
        ax.barh(i, u - l, left=l, height=0.4, color=c, alpha=0.6,
                edgecolor=c, linewidth=2, label=lab)
        ax.text(u + 0.5, i, f"${u:.1f}", va="center", fontsize=10,
                fontweight="bold", color=c)
        ax.text(l - 0.5, i, f"${l:.1f}", va="center", fontsize=10,
                fontweight="bold", color=c, ha="right")

    ax.axvline(spot, color="black", lw=2, ls="-", label=f"Spot ${spot:.0f}")
    ax.axvline(oi["max_pain"], color="orange", lw=1.5, ls="--",
               label=f"Max Pain ${oi['max_pain']:.0f}")
    ax.axvline(oi["call_wall"], color="green", lw=1.5, ls=":",
               alpha=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_title("5日预测区间对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("GLD ($)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 3: Hybrid Band 对比
    ax = axes[1, 0]

    # 近30天价格 + band
    lookback = 30
    recent = close.iloc[-lookback:]

    ax.plot(recent.index, recent.values, "k-", lw=1.8, label="GLD")

    # 原始 band
    ub = upper_band.reindex(recent.index).dropna()
    lb = lower_band.reindex(recent.index).dropna()
    ax.plot(ub.index, ub.values, color="#2196F3", lw=1, ls="--",
            alpha=0.7, label="原始 Band")
    ax.plot(lb.index, lb.values, color="#2196F3", lw=1, ls="--",
            alpha=0.7)
    cidx = ub.index.intersection(lb.index)
    if len(cidx) > 0:
        ax.fill_between(cidx, lb.loc[cidx], ub.loc[cidx],
                         alpha=0.06, color="#2196F3")

    # 修正 band (只画最后一天延伸)
    from datetime import timedelta
    fut_start = last_date + timedelta(days=1)
    fut_end = last_date + timedelta(days=6)

    # 原始预测区间
    ax.fill_between([fut_start, fut_end],
                     [orig_lower, orig_lower],
                     [orig_upper, orig_upper],
                     alpha=0.12, color="#2196F3", label="原始5日区间")
    ax.plot([fut_start, fut_end], [orig_upper, orig_upper],
            color="#2196F3", lw=1.2, ls="--", alpha=0.5)
    ax.plot([fut_start, fut_end], [orig_lower, orig_lower],
            color="#2196F3", lw=1.2, ls="--", alpha=0.5)

    # 修正预测区间
    ax.fill_between([fut_start, fut_end],
                     [adj_lower, adj_lower],
                     [adj_upper, adj_upper],
                     alpha=0.15, color="#FF9800", label="OI修正区间")
    ax.plot([fut_start, fut_end], [adj_upper, adj_upper],
            color="#FF9800", lw=1.5, ls="-", alpha=0.8)
    ax.plot([fut_start, fut_end], [adj_lower, adj_lower],
            color="#FF9800", lw=1.5, ls="-", alpha=0.8)

    # OI 关键价位
    ax.axhline(oi["max_pain"], color="orange", lw=1, ls="--", alpha=0.6)
    ax.annotate(f"Max Pain ${oi['max_pain']:.0f}",
                xy=(recent.index[-1], oi["max_pain"]),
                fontsize=8, color="orange", fontweight="bold")
    if oi["call_wall"] < spot * 1.15:
        ax.axhline(oi["call_wall"], color="green", lw=1, ls=":",
                   alpha=0.5)
        ax.annotate(f"Call Wall ${oi['call_wall']:.0f}",
                    xy=(recent.index[-1], oi["call_wall"]),
                    fontsize=8, color="green", fontweight="bold")

    ax.set_title("价格 + Band + OI修正区间", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(True, alpha=0.3)

    # Panel 4: 修正明细表
    ax = axes[1, 1]
    ax.axis("off")

    table_data = [
        ["Max Pain", f"${oi['max_pain']:.0f}",
         f"{(oi['max_pain']/spot-1)*100:+.1f}%", "↓上界" if spot > oi['max_pain'] else "↑下界"],
        ["Call Wall", f"${oi['call_wall']:.0f}",
         f"{(oi['call_wall']/spot-1)*100:+.1f}%",
         "压制" if det['cw_adj'] else "不影响"],
        ["Put Wall", f"${oi['put_wall']:.0f}",
         f"{(oi['put_wall']/spot-1)*100:+.1f}%",
         "支撑" if det['pw_adj'] else "不影响"],
        ["Net Gamma", f"{oi['net_gex']:,.0f}",
         "long" if oi['net_gex'] > 0 else "short",
         det['gamma_effect']],
        ["Nearest DTE", f"{oi['nearest_dte']}d", "", ""],
        ["到期因子", f"{det['expiry_factor']:.2f}", "", ""],
        ["", "", "", ""],
        ["5日上界", f"${orig_upper:.1f} → ${adj_upper:.1f}",
         f"{det['upper_change']:+.1f}%", ""],
        ["5日下界", f"${orig_lower:.1f} → ${adj_lower:.1f}",
         f"{det['lower_change']:+.1f}%", ""],
        ["Band上界", f"${band_upper:.1f} → ${adj_band_upper:.1f}",
         f"{det_band['upper_change']:+.1f}%", ""],
        ["Band下界", f"${band_lower:.1f} → ${adj_band_lower:.1f}",
         f"{det_band['lower_change']:+.1f}%", ""],
        ["买入阈值", f"${orig_bp030:.1f} → ${adj_bp030:.1f}",
         f"{(adj_bp030/orig_bp030-1)*100:+.1f}%", ""],
        ["平仓阈值", f"${orig_bp090:.1f} → ${adj_bp090:.1f}",
         f"{(adj_bp090/orig_bp090-1)*100:+.1f}%", ""],
    ]
    headers = ["指标", "值", "变化", "效应"]
    tbl = ax.table(cellText=table_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    for i in range(len(headers)):
        tbl[0, i].set_facecolor("#E0E0E0")
        tbl[0, i].set_text_props(fontweight="bold")
    ax.set_title("修正明细", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "oi_adjust_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")


if __name__ == "__main__":
    main()
