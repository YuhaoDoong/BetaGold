"""模型训练状态管理

功能:
  - 检测模型文件最后训练时间 (基于 mtime)
  - 判断是否需要重新训练 (默认 7 天)
  - 启动后台训练任务 (subprocess, 不阻塞 Streamlit)
  - 读取训练进度日志
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


# asset -> (模型文件名, PID/log 文件后缀)
_ASSET_INFO = {
    "gld": {"model": "models/dl_range_v2_model.pkl", "tag": ""},
    "slv": {"model": "models/dl_range_slv_model.pkl", "tag": "_slv"},
}

# 训练日志 + PID 文件 (写入临时目录, 避免污染 data/)
_LOG_DIR = Path("/tmp/golddash_training")
_LOG_DIR.mkdir(exist_ok=True)

# Gold 项目路径 (训练脚本所在位置)
_GOLD_ROOT = "/Users/yhdong/Gold"
_TRAIN_SCRIPT = "src/models/train_dl_range.py"
_CONDA_PYTHON = "/Users/yhdong/miniconda3/envs/gold/bin/python"

# 默认训练新鲜度窗口 (天)
DEFAULT_MAX_AGE_DAYS = 7


def _files(asset: str = "gld"):
    tag = _ASSET_INFO[asset]["tag"]
    return (_LOG_DIR / f"train{tag}.log",
            _LOG_DIR / f"train{tag}.pid",
            _LOG_DIR / f"train{tag}.start")


def _model_path(data_root: str, asset: str = "gld") -> str:
    return os.path.join(data_root, _ASSET_INFO[asset]["model"])


def get_model_mtime(data_root: str, asset: str = "gld") -> datetime | None:
    """返回模型文件最后修改时间; 不存在则返回 None."""
    path = _model_path(data_root, asset)
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path))


def get_model_age_days(data_root: str, asset: str = "gld") -> float | None:
    """返回模型距今天的天数 (float); 不存在返回 None."""
    mt = get_model_mtime(data_root, asset)
    if mt is None:
        return None
    return (datetime.now() - mt).total_seconds() / 86400


def is_stale(data_root: str, max_age_days: int = DEFAULT_MAX_AGE_DAYS,
             asset: str = "gld") -> bool:
    """判断模型是否过期."""
    age = get_model_age_days(data_root, asset)
    return age is None or age > max_age_days


def is_training(asset: str = "gld") -> bool:
    """判断指定 asset 是否有训练任务在进行中."""
    _, pid_file, _ = _files(asset)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False


def start_training(asset: str = "gld") -> tuple[bool, str]:
    """启动后台训练任务.

    asset: "gld" (默认) 或 "slv".
    """
    if asset not in _ASSET_INFO:
        return False, f"未知 asset: {asset}"
    if is_training(asset):
        return False, f"{asset.upper()} 训练任务已在运行中"

    log_file, pid_file, start_file = _files(asset)
    log_file.write_text("")
    start_file.write_text(datetime.now().isoformat())

    cmd = [_CONDA_PYTHON, "-u", _TRAIN_SCRIPT, "--asset", asset]
    env = os.environ.copy()
    env["PYTHONPATH"] = _GOLD_ROOT

    try:
        with open(log_file, "w") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=_GOLD_ROOT,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        pid_file.write_text(str(proc.pid))
        return True, f"{asset.upper()} 训练已启动 (PID={proc.pid}), 预计 40-60 分钟"
    except Exception as e:
        return False, f"启动失败: {e}"


def get_training_log(n_lines: int = 30, asset: str = "gld") -> str:
    """读取指定 asset 训练日志的最后 N 行."""
    log_file, _, _ = _files(asset)
    if not log_file.exists():
        return ""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""


def get_training_elapsed(asset: str = "gld") -> str:
    """指定 asset 训练已运行时长."""
    _, _, start_file = _files(asset)
    if not start_file.exists():
        return ""
    try:
        started = datetime.fromisoformat(start_file.read_text().strip())
        elapsed = (datetime.now() - started).total_seconds()
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m}m{s}s"
        if m > 0:
            return f"{m}m{s}s"
        return f"{s}s"
    except (ValueError, OSError):
        return ""


def stop_training(asset: str = "gld") -> tuple[bool, str]:
    """强行终止训练任务."""
    if not is_training(asset):
        return False, f"没有运行中的 {asset.upper()} 训练任务"
    _, pid_file, _ = _files(asset)
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 15)
        time.sleep(1)
        if pid_file.exists():
            pid_file.unlink()
        return True, f"已发送终止信号 (PID={pid})"
    except Exception as e:
        return False, f"终止失败: {e}"
