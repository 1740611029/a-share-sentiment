"""波段信号模型：研究极端情绪之后 T+0~T+10 个交易日的阶段低点/高点。"""
from .engine import run_model_backtest
from .models import MODELS
from .runner import run_swing

__all__ = ["MODELS", "run_model_backtest", "run_swing"]
