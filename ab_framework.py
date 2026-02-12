"""
A/B Testing Framework
=====================

Comprehensive A/B testing framework with significance testing (p<0.05).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import hashlib
from loguru import logger


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    control_name: str = "control"
    treatment_name: str = "treatment"
    significance_level: float = 0.05


@dataclass
class ExperimentResults:
    """Results from an A/B test analysis."""
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int
    absolute_effect: float
    relative_effect: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: float
    effect_size: float


class ABTestFramework:
    """A/B Testing framework with significance testing."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.assignments: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, ExperimentResults] = {}
    
    def calculate_sample_size(
        self, baseline_rate: float, mde: float, power: float = 0.80
    ) -> int:
        """Calculate required sample size per group."""
        treatment_rate = baseline_rate * (1 + mde)
        pooled_rate = (baseline_rate + treatment_rate) / 2
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(power)
        effect = abs(treatment_rate - baseline_rate)
        variance = 2 * pooled_rate * (1 - pooled_rate)
        if effect == 0:
            return 10000
        n = variance * ((z_alpha + z_beta) ** 2) / (effect ** 2)
        return int(np.ceil(n))
    
    def create_experiment(
        self, name: str, control_users: List[str], treatment_users: List[str],
        description: str = ""
    ) -> None:
        """Create experiment with user assignments."""
        config = ExperimentConfig(name=name, description=description)
        self.experiments[name] = config
        
        control_df = pd.DataFrame({'user_id': control_users, 'variant': 'control', 'experiment': name})
        treatment_df = pd.DataFrame({'user_id': treatment_users, 'variant': 'treatment', 'experiment': name})
        self.assignments[name] = pd.concat([control_df, treatment_df], ignore_index=True)
        logger.info(f"Created experiment {name}: {len(control_users)} control, {len(treatment_users)} treatment")
    
    def assign_users_randomly(
        self, user_ids: List[str], treatment_ratio: float = 0.5, seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """Randomly assign users to groups."""
        np.random.seed(seed)
        shuffled = np.random.permutation(user_ids)
        split_idx = int(len(shuffled) * (1 - treatment_ratio))
        return list(shuffled[:split_idx]), list(shuffled[split_idx:])
    
    def analyze_results(
        self, experiment_name: str, metric_data: pd.DataFrame, 
        metric_column: str, user_column: str = 'user_id'
    ) -> ExperimentResults:
        """Analyze experiment results with statistical tests."""
        config = self.experiments[experiment_name]
        assignments = self.assignments[experiment_name]
        
        merged = metric_data.merge(assignments, left_on=user_column, right_on='user_id')
        control_data = merged[merged['variant'] == 'control'][metric_column]
        treatment_data = merged[merged['variant'] == 'treatment'][metric_column]
        
        control_mean, treatment_mean = control_data.mean(), treatment_data.mean()
        control_std, treatment_std = control_data.std(), treatment_data.std()
        control_n, treatment_n = len(control_data), len(treatment_data)
        
        absolute_effect = treatment_mean - control_mean
        relative_effect = absolute_effect / control_mean if control_mean != 0 else 0
        
        _, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        pooled_se = np.sqrt((control_std**2/control_n) + (treatment_std**2/treatment_n))
        z = stats.norm.ppf(1 - self.significance_level/2)
        ci = (absolute_effect - z*pooled_se, absolute_effect + z*pooled_se)
        
        pooled_std = np.sqrt(((control_n-1)*control_std**2 + (treatment_n-1)*treatment_std**2)/(control_n+treatment_n-2))
        cohens_d = absolute_effect / pooled_std if pooled_std != 0 else 0
        
        results = ExperimentResults(
            control_mean=control_mean, treatment_mean=treatment_mean,
            control_std=control_std, treatment_std=treatment_std,
            control_n=control_n, treatment_n=treatment_n,
            absolute_effect=absolute_effect, relative_effect=relative_effect,
            p_value=p_value, confidence_interval=ci,
            is_significant=p_value < self.significance_level,
            power=0.80, effect_size=cohens_d
        )
        self.results[experiment_name] = results
        return results
    
    def generate_report(self, experiment_name: str) -> str:
        """Generate experiment report."""
        r = self.results[experiment_name]
        c = self.experiments[experiment_name]
        return f"""
{'='*60}
A/B TEST: {c.name}
{'='*60}
Control: {r.control_n:,} users, Mean: {r.control_mean:.4f}
Treatment: {r.treatment_n:,} users, Mean: {r.treatment_mean:.4f}

Effect: {r.absolute_effect:+.4f} ({r.relative_effect:+.2%})
95% CI: [{r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f}]
P-value: {r.p_value:.4f}
Significant: {'YES' if r.is_significant else 'NO'}
{'='*60}
"""


def run_ab_test_simulation(n_users: int = 10000, effect: float = 0.05) -> ExperimentResults:
    """Run simulated A/B test."""
    np.random.seed(42)
    control_n, treatment_n = n_users // 2, n_users - n_users // 2
    control = np.random.binomial(1, 0.30, control_n)
    treatment = np.random.binomial(1, 0.30 * (1 + effect), treatment_n)
    
    ab = ABTestFramework()
    ab.create_experiment("test", [f"u{i}" for i in range(control_n)], 
                         [f"u{i}" for i in range(control_n, n_users)])
    
    data = pd.DataFrame({
        'user_id': [f"u{i}" for i in range(n_users)],
        'rate': np.concatenate([control, treatment])
    })
    
    results = ab.analyze_results("test", data, "rate")
    print(ab.generate_report("test"))
    return results


__all__ = ['ABTestFramework', 'ExperimentConfig', 'ExperimentResults', 'run_ab_test_simulation']
