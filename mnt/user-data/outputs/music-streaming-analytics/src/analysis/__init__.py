"""
Analysis Module
===============

Statistical analysis for music streaming data.

Modules:
    - CohortAnalyzer: User retention and cohort analysis
    - FunnelAnalyzer: Funnel and conversion analysis
"""

from .cohort_analysis import CohortAnalyzer
from .funnel_analysis import FunnelAnalyzer, FunnelStage

__all__ = ['CohortAnalyzer', 'FunnelAnalyzer', 'FunnelStage']
