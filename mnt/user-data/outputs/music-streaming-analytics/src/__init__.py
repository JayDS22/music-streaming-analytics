"""
Music Streaming Analytics Platform
===================================

A comprehensive analytics platform for music streaming services.

Modules:
    - api: Spotify API integration
    - data: Data loading and generation
    - features: Feature engineering (50+ features)
    - models: Predictive models (skip prediction, session forecasting)
    - analysis: Cohort and funnel analysis
    - ab_testing: A/B testing framework
    - visualization: Dashboard generation
    - utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from loguru import logger
import sys

# Configure default logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)
