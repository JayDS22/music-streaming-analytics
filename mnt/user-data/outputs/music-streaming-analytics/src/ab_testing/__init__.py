"""
A/B Testing Module
==================

Framework for experiment design and statistical testing.
"""

from .ab_framework import ABTestFramework, ExperimentConfig, ExperimentResults, run_ab_test_simulation

__all__ = ['ABTestFramework', 'ExperimentConfig', 'ExperimentResults', 'run_ab_test_simulation']
