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


# 模型文件路径 (相对 data_root)
_MODEL_FILE = "models/dl_range_v2_model.pkl"

# 训练日志 + PID 文件 (写入临时目录, 避免污染 data/)
_LOG_DIR = Path("/tmp/golddash_training")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "train.log"
_PID_FILE = _LOG_DIR / "train.pid"
_START_FILE = _LOG_DIR / "train.start"

# Gold 项目路径 (训练脚本所在位置)
_GOLD_ROOT = "/Users/yhdong/Gold"
_TRAIN_SCRIPT = "src/models/train_dl_range.py"
_CONDA_PYTHON = "/Users/yhdong/miniconda3/envs/gold/bin/python"

# 默认训练新鲜度窗口 (天)
DEFAULT_MAX_AGE_DAYS = 7


def get_model_mtime(data_root: str) -> datetime | None:
    """返回模型文件最后修改时间; 不存在则返回 None."""
    path = os.path.join(data_root, _MODEL_FILE)
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path))


def get_model_age_days(data_root: str) -> float | None:
    """返回模型距今天的天数 (float); 不存在返回 None."""
    mt = get_model_mtime(data_root)
    if mt is None:
        return None
    return (datetime.now() - mt).total_seconds() / 86400


def is_stale(data_root: str, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
    """判断模型是否过期."""
    age = get_model_age_days(data_root)
    return age is None or age > max_age_days


def is_training() -> bool:
    """判断是否有训练任务在进行中."""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        # 检查进程是否还活着
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        # PID 文件过期, 清理
        try:
            _PID_FILE.unlink()
        except OSError:
            pass
        return False


def start_training() -> tuple[bool, str]:
    """启动后台训练任务.

    返回 (success, message).
    """
    if is_training():
        return False, "训练任务已在运行中"

    # 清空旧日志
    _LOG_FILE.write_text("")
    _START_FILE.write_text(datetime.now().isoformat())

    cmd = [_CONDA_PYTHON, "-u", _TRAIN_SCRIPT]
    env = os.environ.copy()
    env["PYTHONPATH"] = _GOLD_ROOT

    try:
        with open(_LOG_FILE, "w") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=_GOLD_ROOT,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        _PID_FILE.write_text(str(proc.pid))
        return True, f"训练已启动 (PID={proc.pid}), 预计 40-60 分钟"
    except Exception as e:
        return False, f"启动失败: {e}"


def get_training_log(n_lines: int = 30) -> str:
    """读取训练日志的最后 N 行."""
    if not _LOG_FILE.exists():
        return ""
    try:
        with open(_LOG_FILE, "r") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""


def get_training_elapsed() -> str:
    """训练已运行时长 (仅在训练中时有效)."""
    if not _START_FILE.exists():
        return ""
    try:
        started = datetime.fromisoformat(_START_FILE.read_text().strip())
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


def stop_training() -> tuple[bool, str]:
    """强行终止训练任务."""
    if not is_training():
        return False, "没有运行中的训练任务"
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 15)  # SIGTERM
        time.sleep(1)
        if _PID_FILE.exists():
            _PID_FILE.unlink()
        return True, f"已发送终止信号 (PID={pid})"
    except Exception as e:
        return False, f"终止失败: {e}"
