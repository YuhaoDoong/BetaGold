"""
数据采集主入口
统一调度所有数据源的采集
"""

import os
import sys
import logging
from datetime import datetime

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data.market.market_data import MarketDataCollector
from src.data.volatility.vol_data import VolDataCollector
from src.data.positioning.cot_data import COTDataCollector
from src.data.positioning.central_bank_gold import CentralBankGoldCollector
from src.data.events.economic_events import EconomicEventsCollector
from src.config_loader import load_config


def setup_logging():
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_collection_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def collect_market_data(config: dict) -> dict:
    """采集市场数据"""
    print("\n" + "=" * 60)
    print("[1/7] 采集市场行情数据 (Yahoo Finance)")
    print("=" * 60)

    collector = MarketDataCollector(config)
    market_data = collector.fetch_all_market_data()
    collector.save_market_data(market_data)

    summary = {}
    for name, df in market_data.items():
        summary[name] = {
            "rows": len(df),
            "start": df.index[0].strftime("%Y-%m-%d"),
            "end": df.index[-1].strftime("%Y-%m-%d"),
        }

    return summary


def collect_macro_data(config: dict) -> dict:
    """采集宏观经济数据"""
    print("\n" + "=" * 60)
    print("[2/7] 采集宏观经济数据 (FRED)")
    print("=" * 60)

    api_key = config.get("fred_api_key", "")
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        print("\n  跳过 FRED 数据采集 - 尚未配置 API Key")
        print("  请访问 https://fred.stlouisfed.org/docs/api/api_key.html 免费申请")
        print("  然后将 Key 填入 config/settings.yaml")
        return {"status": "skipped", "reason": "no_api_key"}

    from src.data.macro.macro_data import MacroDataCollector

    collector = MacroDataCollector(config)
    macro_data = collector.fetch_all_macro_data()
    collector.save_macro_data(macro_data)

    panel = collector.build_macro_panel(macro_data)
    if not panel.empty:
        panel_path = os.path.join(config["paths"]["raw_data"], "macro", "macro_panel.csv")
        panel.to_csv(panel_path)

    summary = {}
    for name, series in macro_data.items():
        summary[name] = {
            "rows": len(series),
            "valid": int(series.notna().sum()),
        }

    return summary


def collect_vol_data(config: dict) -> dict:
    """采集波动率数据"""
    print("\n" + "=" * 60)
    print("[3/7] 采集波动率数据")
    print("=" * 60)

    collector = VolDataCollector(config)

    gvz = collector.fetch_gvz()
    vix_term = collector.fetch_vix_term_structure()
    move = collector.fetch_move_index()
    collector.save_vol_data(gvz, vix_term, move)

    return {
        "gvz": len(gvz) if not gvz.empty else 0,
        "vix_term": len(vix_term) if not vix_term.empty else 0,
        "move": len(move) if not move.empty else 0,
    }


def collect_cot_data(config: dict) -> dict:
    """采集 COT 持仓数据"""
    print("\n" + "=" * 60)
    print("[4/7] 采集 CFTC COT 持仓数据")
    print("=" * 60)

    collector = COTDataCollector(config)
    cot_data = collector.fetch_cot_historical(start_year=2006)

    if cot_data.empty:
        return {"status": "failed"}

    collector.save_cot_data(cot_data)

    return {
        "rows": len(cot_data),
        "start": cot_data["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end": cot_data["Date"].iloc[-1].strftime("%Y-%m-%d"),
    }


def collect_central_bank_gold(config: dict) -> dict:
    """处理央行购金数据"""
    print("\n" + "=" * 60)
    print("[5/7] 处理央行购金数据 (WGC Excel)")
    print("=" * 60)

    collector = CentralBankGoldCollector(config)
    monthly = collector.parse_monthly_changes()

    if monthly.empty:
        print("  未找到 WGC Excel 文件，跳过")
        return {"status": "skipped"}

    annual = collector.parse_annual_changes()
    features = collector.build_model_features(monthly)
    collector.save_data(monthly, annual, features)

    return {
        "months": len(monthly),
        "start": monthly.index[0].strftime("%Y-%m"),
        "end": monthly.index[-1].strftime("%Y-%m"),
        "features": len(features.columns),
    }


def collect_economic_events(config: dict) -> dict:
    """采集经济事件日历"""
    print("\n" + "=" * 60)
    print("[6/7] 采集经济事件日历")
    print("=" * 60)

    collector = EconomicEventsCollector(config)
    events = collector.fetch_all_economic_events()

    if events.empty:
        return {"status": "failed"}

    collector.save_events(events)

    summary = {}
    for etype in events["Event"].unique():
        summary[etype] = len(events[events["Event"] == etype])
    return summary


def collect_gld_options(config: dict) -> dict:
    """采集 GLD 期权链（当前快照）"""
    print("\n" + "=" * 60)
    print("[7/7] 采集 GLD 当前期权链 (Yahoo Finance)")
    print("=" * 60)

    collector = MarketDataCollector(config)
    options = collector.fetch_gld_options_chain()

    if options:
        collector.save_options_chain(options)
        return {"expirations": len(options)}
    return {"status": "failed"}


def archive_options_snapshot(config: dict) -> dict:
    """期权链历史存档 (Phase 1.5)"""
    print("\n" + "-" * 40)
    print("附加: 期权链历史存档 (Phase 1.5)")
    print("-" * 40)

    try:
        from src.data.options.options_archive import OptionsArchiver
        archiver = OptionsArchiver(config)
        path = archiver.save_eod_snapshot()
        if path:
            return {"status": "saved", "path": path}
        return {"status": "failed"}
    except ConnectionRefusedError:
        print("  Moomoo OpenD 未启动，跳过")
        return {"status": "opend_not_running"}
    except Exception as e:
        print(f"  期权链存档失败: {e}")
        return {"status": "failed"}


def collect_moomoo_options(config: dict) -> dict:
    """采集 Moomoo 期权数据（需要 OpenD 运行）"""
    print("\n" + "-" * 40)
    print("附加: 采集 Moomoo 期权数据 (需要 OpenD)")
    print("-" * 40)

    try:
        from src.data.options.moomoo_data import MoomooDataCollector
        collector = MoomooDataCollector()
        option_data = collector.get_full_option_data(num_expirations=6)
        if not option_data.empty:
            collector.save_option_data(option_data, label="full")
            collector.close()
            return {"contracts": len(option_data)}
        collector.close()
        return {"status": "no_data"}
    except ConnectionRefusedError:
        print("  Moomoo OpenD 未启动，跳过")
        return {"status": "opend_not_running"}
    except Exception as e:
        print(f"  Moomoo 采集失败: {e}")
        return {"status": "failed"}


def main():
    log_file = setup_logging()
    config = load_config()

    print("=" * 60)
    print("  GLD Options Trading System - 数据采集")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据范围: {config['data']['start_date']} ~ 今天")
    print(f"  日志: {log_file}")
    print("=" * 60)

    results = {}

    # 1. 市场行情数据 (Yahoo Finance)
    results["market"] = collect_market_data(config)

    # 2. 宏观经济数据 (FRED)
    results["macro"] = collect_macro_data(config)

    # 3. 波动率数据
    results["volatility"] = collect_vol_data(config)

    # 4. COT 持仓数据
    results["cot"] = collect_cot_data(config)

    # 5. 央行购金数据 (WGC Excel)
    results["central_bank"] = collect_central_bank_gold(config)

    # 6. 经济事件日历
    results["events"] = collect_economic_events(config)

    # 7. GLD 期权链 (Yahoo Finance)
    results["options"] = collect_gld_options(config)

    # 附加: Moomoo 期权数据 (如果 OpenD 运行中)
    results["moomoo"] = collect_moomoo_options(config)

    # 附加: 期权链历史存档 (Phase 1.5, 如果 OpenD 运行中)
    results["options_archive"] = archive_options_snapshot(config)

    # 汇总报告
    print("\n" + "=" * 60)
    print("  数据采集汇总报告")
    print("=" * 60)

    print("\n[1. 市场数据]")
    if isinstance(results["market"], dict) and "status" not in results["market"]:
        for name, info in results["market"].items():
            print(f"  {name}: {info['rows']} 条 ({info['start']} ~ {info['end']})")

    print("\n[2. 宏观数据]")
    if results["macro"].get("status") == "skipped":
        print("  已跳过 (需配置 FRED API Key)")
    elif isinstance(results["macro"], dict):
        for name, info in results["macro"].items():
            if isinstance(info, dict) and "rows" in info:
                print(f"  {name}: {info['rows']} 条 (有效 {info['valid']})")

    print("\n[3. 波动率数据]")
    vol = results["volatility"]
    print(f"  GVZ: {vol['gvz']} 条")
    print(f"  VIX期限结构: {vol['vix_term']} 条")
    print(f"  MOVE: {vol['move']} 条")

    print("\n[4. COT持仓]")
    cot = results["cot"]
    if "rows" in cot:
        print(f"  黄金COT: {cot['rows']} 条 ({cot['start']} ~ {cot['end']})")
    else:
        print(f"  采集失败")

    print("\n[5. 央行购金]")
    cb = results["central_bank"]
    if "months" in cb:
        print(f"  月度数据: {cb['months']} 个月 ({cb['start']} ~ {cb['end']}), {cb['features']} 个特征")
    else:
        print(f"  {cb.get('status', 'failed')}")

    print("\n[6. 经济事件]")
    ev = results["events"]
    if "status" not in ev:
        for etype, count in ev.items():
            print(f"  {etype}: {count} 条")
    else:
        print(f"  采集失败")

    print("\n[7. GLD期权链]")
    opt = results["options"]
    if "expirations" in opt:
        print(f"  到期日: {opt['expirations']} 个")
    else:
        print(f"  采集失败")

    print("\n[附加: Moomoo期权]")
    moo = results["moomoo"]
    if "contracts" in moo:
        print(f"  合约数: {moo['contracts']}")
    else:
        print(f"  {moo.get('status', 'skipped')}")

    print("\n[附加: 期权链历史存档]")
    arch = results["options_archive"]
    if arch.get("status") == "saved":
        print(f"  已存档: {arch['path']}")
    else:
        print(f"  {arch.get('status', 'skipped')}")

    print(f"\n日志已保存: {log_file}")
    print("数据采集完成!")

    return results


if __name__ == "__main__":
    main()
