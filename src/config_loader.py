"""配置加载器"""

import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 将相对路径转为绝对路径
    for key in ["raw_data", "processed_data", "cache"]:
        config["paths"][key] = os.path.join(PROJECT_ROOT, config["paths"][key])

    return config
