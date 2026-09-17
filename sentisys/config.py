"""配置加载：default.yaml + 可选用户覆盖"""
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(user_path: str | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if user_path:
        with open(user_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    # 数据目录相对路径 → 绝对路径
    for key in ("cache_dir", "factor_dir", "db_path"):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(ROOT / p)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
