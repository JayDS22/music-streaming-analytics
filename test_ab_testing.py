"""
Tests for A/B Testing Framework
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_testing.ab_framework import ABTestFramework, ExperimentConfig, ExperimentResults


@pytest.fixture
def ab_framework():
    """Create A/B testing framework instance."""
    return ABTestFramework(significance_level=0.05)


@pytest.fixture
def sample_users():
    """Generate sample user IDs."""
    return [f'user_{i}' for i in range(1000)]


@pytest.fixture
def sample_metric_data():
    """Generate sample metric data with treatment effect."""
    np.random.seed(42)
    
    control_users = [f'user_{i}' for i in range(500)]
    treatment_users = [f'user_{i}' for i in range(500, 1000)]
    
    # Control: mean=0.3
    control_values = np.random.binomial(1, 0.30, 500)
    # Treatment: mean=0.35 (5% lift)
    treatment_values = np.random.binomial(1, 0.35, 500)
    
    return pd.DataFrame({
        'user_id': control_users + treatment_users,
        'metric_value': np.concatenate([control_values, treatment_values])
    })


class TestABTestFramework:
    """Test ABTestFramework class."""
    
    def test_init(self):
        """Test initialization."""
        ab = ABTestFramework(significance_level=0.05)
        assert ab.significance_level == 0.05
        assert len(ab.experiments) == 0
    
    def test_calculate_sample_size(self, ab_framework):
        """Test sample size calculation."""
        n = ab_framework.calculate_sample_size(
            baseline_rate=0.30,
            mde=0.05,
            power=0.80
        )
        
        assert n > 0
        assert isinstance(n, int)
        # Should need reasonable sample size for 5% MDE
        assert 1000 < n < 100000
    
    def test_create_experiment(self, ab_framework, sample_users):
        """Test experiment creation."""
        control = sample_users[:500]
        treatment = sample_users[500:]
        
        ab_framework.create_experiment(
            name="test_exp",
            control_users=control,
            treatment_users=treatment,
            description="Test experiment"
        )
        
        assert "test_exp" in ab_framework.experiments
        assert "test_exp" in ab_framework.assignments
        assert len(ab_framework.assignments["test_exp"]) == 1000
    
    def test_assign_users_randomly(self, ab_framework, sample_users):
        """Test random user assignment."""
        control, treatment = ab_framework.assign_users_randomly(
            sample_users,
            treatment_ratio=0.5,
            seed=42
        )
        
        assert len(control) + len(treatment) == len(sample_users)
        # Roughly 50/50 split
        assert abs(len(control) - len(treatment)) < 100
        # No overlap
        assert len(set(control) & set(treatment)) == 0
    
    def test_analyze_results(self, ab_framework, sample_metric_data):
        """Test results analysis."""
        # Create experiment
        control = [f'user_{i}' for i in range(500)]
        treatment = [f'user_{i}' for i in range(500, 1000)]
        
        ab_framework.create_experiment(
            name="test_exp",
            control_users=control,
            treatment_users=treatment
        )
        
        # Analyze
        results = ab_framework.analyze_results(
            experiment_name="test_exp",
            metric_data=sample_metric_data,
            metric_column="metric_value"
        )
        
        assert isinstance(results, ExperimentResults)
        assert results.control_n == 500
        assert results.treatment_n == 500
        assert 0 <= results.p_value <= 1
        assert isinstance(results.is_significant, (bool, np.bool_))
    
    def test_significant_effect_detection(self, ab_framework):
        """Test that significant effects are detected."""
        np.random.seed(42)
        
        # Create data with large effect
        control_users = [f'user_{i}' for i in range(1000)]
        treatment_users = [f'user_{i}' for i in range(1000, 2000)]
        
        control_values = np.random.normal(0.30, 0.05, 1000)
        treatment_values = np.random.normal(0.40, 0.05, 1000)  # Large effect
        
        metric_data = pd.DataFrame({
            'user_id': control_users + treatment_users,
            'metric_value': np.concatenate([control_values, treatment_values])
        })
        
        ab_framework.create_experiment(
            name="sig_test",
            control_users=control_users,
            treatment_users=treatment_users
        )
        
        results = ab_framework.analyze_results(
            experiment_name="sig_test",
            metric_data=metric_data,
            metric_column="metric_value"
        )
        
        # Should detect significance with large effect
        assert results.is_significant == True
        assert results.p_value < 0.05
    
    def test_no_effect_not_significant(self, ab_framework):
        """Test that no effect is not significant."""
        np.random.seed(42)
        
        # Create data with no effect
        control_users = [f'user_{i}' for i in range(500)]
        treatment_users = [f'user_{i}' for i in range(500, 1000)]
        
        # Same distribution for both
        all_values = np.random.normal(0.30, 0.1, 1000)
        
        metric_data = pd.DataFrame({
            'user_id': control_users + treatment_users,
            'metric_value': all_values
        })
        
        ab_framework.create_experiment(
            name="no_effect_test",
            control_users=control_users,
            treatment_users=treatment_users
        )
        
        results = ab_framework.analyze_results(
            experiment_name="no_effect_test",
            metric_data=metric_data,
            metric_column="metric_value"
        )
        
        # Effect should be close to zero
        assert abs(results.relative_effect) < 0.1
    
    def test_generate_report(self, ab_framework, sample_metric_data):
        """Test report generation."""
        control = [f'user_{i}' for i in range(500)]
        treatment = [f'user_{i}' for i in range(500, 1000)]
        
        ab_framework.create_experiment(
            name="report_test",
            control_users=control,
            treatment_users=treatment
        )
        
        ab_framework.analyze_results(
            experiment_name="report_test",
            metric_data=sample_metric_data,
            metric_column="metric_value"
        )
        
        report = ab_framework.generate_report("report_test")
        
        assert isinstance(report, str)
        assert "report_test" in report
        assert "Control" in report
        assert "Treatment" in report
        assert "P-value" in report or "p-value" in report.lower()


class TestExperimentResults:
    """Test ExperimentResults dataclass."""
    
    def test_results_attributes(self):
        """Test results have expected attributes."""
        results = ExperimentResults(
            control_mean=0.30,
            treatment_mean=0.35,
            control_std=0.1,
            treatment_std=0.1,
            control_n=500,
            treatment_n=500,
            absolute_effect=0.05,
            relative_effect=0.167,
            p_value=0.01,
            confidence_interval=(0.02, 0.08),
            is_significant=True,
            power=0.85,
            effect_size=0.5
        )
        
        assert results.control_mean == 0.30
        assert results.treatment_mean == 0.35
        assert results.is_significant == True
        assert results.p_value == 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
