"""Ledger 后台 daemon — 持续重建 positions_ledger.json.

绑定 dashboard 进程生命周期:
  - dashboard 启动 → daemon thread 启动 (daemon=True)
  - dashboard 退出 (Ctrl+C / kill) → thread 随 process 自动结束

不需 cron / systemd. 用户视角: dashboard run = 服务部署, dashboard close = 服务停.

间隔: REBUILD_INTERVAL_SEC (默认 300s = 5 min).
"""
from __future__ import annotations
import os
import sys
import time
import threading
import subprocess
from pathlib import Path

REBUILD_INTERVAL_SEC = 300  # 5 min
LEDGER_BUILDER = "/Users/yhdong/GoldDash/scripts/build_positions_ledger.py"
STATS_BUILDER = "/Users/yhdong/GoldDash/scripts/compute_strategy_stats.py"  # v3.7.181
BACKFILL_INTRA = "/Users/yhdong/GoldDash/scripts/backfill_intraday_signals.py"  # v3.7.188
LOG_FILE = "/tmp/ledger_daemon.log"

_DAEMON_LOCK = threading.Lock()
_DAEMON_STARTED = False
_DAEMON_THREAD = None


def _rebuild_loop():
    """循环重建 ledger. daemon thread 主体."""
    while True:
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} 重建 ledger ===\n")
                proc = subprocess.run(
                    [sys.executable, LEDGER_BUILDER, "--days", "90"],
                    stdout=f, stderr=subprocess.STDOUT, timeout=180)
                f.write(f"=== exit code {proc.returncode} ===\n")
                # v3.7.181: ledger 重建后自动算 stats (Dashboard 推荐用)
                proc2 = subprocess.run(
                    [sys.executable, STATS_BUILDER],
                    stdout=f, stderr=subprocess.STDOUT, timeout=60)
                f.write(f"=== stats exit code {proc2.returncode} ===\n")
                # v3.7.188: 自动 backfill intraday signal log (修期货 23h kline 未 detect bug)
                for asset_key in ("GLD", "SLV"):
                    proc3 = subprocess.run(
                        [sys.executable, BACKFILL_INTRA,
                         "--asset", asset_key, "--timeframe", "60"],
                        stdout=f, stderr=subprocess.STDOUT, timeout=120)
                    f.write(f"=== backfill {asset_key} exit {proc3.returncode} ===\n")
        except Exception as e:
            try:
                with open(LOG_FILE, "a") as f:
                    f.write(f"!!! daemon 异常: {e}\n")
            except Exception:
                pass
        time.sleep(REBUILD_INTERVAL_SEC)


def start_daemon_once() -> bool:
    """幂等启动 ledger daemon (process 内单例).

    Returns: True if 本次启动了, False if 已经在跑.
    """
    global _DAEMON_STARTED, _DAEMON_THREAD
    with _DAEMON_LOCK:
        if _DAEMON_STARTED:
            return False
        _DAEMON_THREAD = threading.Thread(
            target=_rebuild_loop, daemon=True, name="ledger-daemon")
        _DAEMON_THREAD.start()
        _DAEMON_STARTED = True
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"daemon 启动 (pid={os.getpid()}, interval={REBUILD_INTERVAL_SEC}s) ###\n")
        except Exception:
            pass
    return True


def is_running() -> bool:
    return _DAEMON_STARTED and (_DAEMON_THREAD is not None
                                  and _DAEMON_THREAD.is_alive())
