"""
Utility Functions
=================

Helper functions for the music streaming analytics platform.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    def substitute_env_vars(obj):
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default = None
                if ':' in env_var:
                    env_var, default = env_var.split(':', 1)
                return os.getenv(env_var, default)
            return obj
        elif isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(item) for item in obj]
        return obj
    
    return substitute_env_vars(config)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level
    )
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="1 week", level=level)


def format_duration(ms: float) -> str:
    """Format milliseconds to human-readable duration."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_number(n: float, precision: int = 1) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    return f"{n:.{precision}f}"


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    import json
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return str(obj)
    
    # Deep copy and convert to avoid circular references
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        else:
            return convert(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    clean_results = deep_convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Saved results to {filepath}")


__all__ = [
    'load_config', 'setup_logging', 'format_duration', 
    'format_number', 'calculate_metrics', 'save_results'
]
