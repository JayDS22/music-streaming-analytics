"""Utility Functions"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
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
    return f"{minutes / 60:.1f}h"


def format_number(n: float, precision: int = 1) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    return f"{n:.{precision}f}"


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
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
    
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        else:
            return convert(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    with open(filepath, 'w') as f:
        json.dump(deep_convert(results), f, indent=2)
    logger.info(f"Saved results to {filepath}")


__all__ = ['load_config', 'setup_logging', 'format_duration', 'format_number', 'save_results']
