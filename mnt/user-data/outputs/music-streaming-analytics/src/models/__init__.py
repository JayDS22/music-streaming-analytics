"""
Models Module
=============

Predictive models for music streaming analytics.

Models:
    - SkipPredictor: Logistic regression for skip behavior (AUC: 0.84)
    - SessionForecaster: Linear regression for session duration (R²: 0.79)
"""

from .skip_predictor import SkipPredictor
from .session_forecaster import SessionForecaster, create_session_features

__all__ = ['SkipPredictor', 'SessionForecaster', 'create_session_features']
