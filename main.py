#!/usr/bin/env python3
"""
Music Streaming Analytics Platform
===================================

Main entry point for running the analytics pipeline.

Usage:
    python main.py --full-pipeline
    python main.py --generate-data --sessions 100000
    python main.py --train-models
    python main.py --run-analysis
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import SyntheticDataGenerator, DataGeneratorConfig, DataLoader
from src.features import FeatureEngineer, create_skip_prediction_features
from src.models import SkipPredictor, SessionForecaster
from src.analysis import CohortAnalyzer, FunnelAnalyzer
from src.ab_testing import ABTestFramework, run_ab_test_simulation
from src.visualization import DashboardGenerator
from src.utils import setup_logging, save_results


def generate_data(num_sessions: int = 100000, num_users: int = 1000) -> dict:
    """Generate synthetic data."""
    logger.info(f"Generating data: {num_users} users, {num_sessions} sessions")
    
    config = DataGeneratorConfig(
        num_users=num_users,
        num_sessions=num_sessions,
        num_tracks=5000,
        seed=42
    )
    
    generator = SyntheticDataGenerator(config)
    data = generator.generate_all()
    generator.save_data(data, "data/raw")
    
    return data


def run_feature_engineering(data: dict) -> dict:
    """Run feature engineering pipeline."""
    logger.info("Running feature engineering...")
    
    engineer = FeatureEngineer()
    user_features = engineer.create_all_features(
        data['sessions'], data['users'], data['tracks']
    )
    
    feature_groups = engineer.get_feature_importance_groups()
    logger.info(f"Feature categories: {list(feature_groups.keys())}")
    logger.info(f"Total features: {sum(len(v) for v in feature_groups.values())}")
    
    return {'user_features': user_features, 'feature_groups': feature_groups}


def train_models(data: dict) -> dict:
    """Train prediction models."""
    logger.info("Training models...")
    
    # Skip predictor
    X, y, user_ids, session_ids = create_skip_prediction_features(
        data['sessions'], data['tracks']
    )
    
    skip_model = SkipPredictor(C=1.0, class_weight='balanced')
    skip_metrics = skip_model.train(X, y)
    
    logger.info(f"Skip Predictor - AUC: {skip_metrics['val_auc']:.4f}")
    
    # Session forecaster
    session_model = SessionForecaster(model_type='ridge', n_features=15)
    
    # Create session-level features for duration prediction
    sessions = data['sessions'].copy()
    sessions['hour'] = sessions['timestamp'].dt.hour
    sessions['is_weekend'] = (sessions['timestamp'].dt.dayofweek >= 5).astype(int)
    
    session_features = sessions.groupby('user_id').agg({
        'listen_duration_ms': ['count', 'std'],
        'skipped': 'mean',
        'hour': 'mean'
    })
    session_features.columns = ['session_count', 'duration_std', 'skip_rate', 'avg_hour']
    session_features = session_features.reset_index()
    
    y_duration = sessions.groupby('user_id')['listen_duration_ms'].mean()
    X_duration = session_features.drop('user_id', axis=1).fillna(0)
    y_duration = y_duration.loc[session_features['user_id']].values
    
    session_metrics = session_model.train(X_duration, y_duration)
    logger.info(f"Session Forecaster - R²: {session_metrics['val_r2']:.4f}")
    
    return {
        'skip_predictor': skip_model,
        'session_forecaster': session_model,
        'skip_metrics': skip_metrics,
        'session_metrics': session_metrics
    }


def run_analysis(data: dict) -> dict:
    """Run cohort and funnel analysis."""
    logger.info("Running analysis...")
    
    # Cohort analysis
    cohort = CohortAnalyzer(period='monthly')
    retention = cohort.calculate_retention(data['users'], data['sessions'], periods=6)
    cohort_stats = cohort.get_retention_summary()
    
    logger.info(f"Average Period 1 Retention: {cohort_stats['avg_period_1_retention']:.1f}%")
    
    # Funnel analysis
    funnel = FunnelAnalyzer()
    funnel_metrics = funnel.analyze_playlist_completion(
        data['sessions'], data['playlists'], data['playlist_tracks']
    )
    
    logger.info(f"Playlist Drop-off (tracks 3-5): {funnel_metrics.get('drop_off_track_3_5', 0):.1%}")
    
    # User activation funnel
    activation = funnel.analyze_user_activation(data['users'], data['sessions'])
    logger.info(f"Day 1 Activation: {activation['day_1_activation_rate']:.1%}")
    
    return {
        'retention_matrix': retention,
        'cohort_stats': cohort_stats,
        'funnel_metrics': funnel_metrics,
        'activation_metrics': activation,
        'recommendations': funnel.get_recommendations()
    }


def run_ab_testing() -> dict:
    """Run A/B testing simulation."""
    logger.info("Running A/B test simulation...")
    
    results = run_ab_test_simulation(n_users=10000, effect=0.05)
    
    return {
        'p_value': results.p_value,
        'is_significant': results.is_significant,
        'effect_size': results.effect_size,
        'relative_effect': results.relative_effect
    }


def generate_dashboards(data: dict, analysis_results: dict) -> None:
    """Generate dashboard data and visualizations."""
    logger.info("Generating dashboards...")
    
    dashboard = DashboardGenerator(output_dir="dashboards")
    
    dau_mau = dashboard.calculate_dau_mau(data['sessions'])
    skip_rates = dashboard.calculate_skip_rates(data['sessions'], data['tracks'])
    retention = dashboard.calculate_retention_curve(data['users'], data['sessions'])
    
    dashboard.export_for_tableau(dau_mau, skip_rates, retention)
    dashboard.plot_all(dau_mau, skip_rates, retention)


def run_full_pipeline(num_sessions: int = 100000, num_users: int = 1000):
    """Run the complete analytics pipeline."""
    logger.info("="*60)
    logger.info("MUSIC STREAMING ANALYTICS PLATFORM")
    logger.info("="*60)
    
    # Generate data
    data = generate_data(num_sessions, num_users)
    
    # Feature engineering
    features = run_feature_engineering(data)
    
    # Train models
    models = train_models(data)
    
    # Analysis
    analysis = run_analysis(data)
    
    # A/B testing
    ab_results = run_ab_testing()
    
    # Dashboards
    generate_dashboards(data, analysis)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Data: {num_users:,} users, {num_sessions:,} sessions")
    logger.info(f"Features: {len(features['user_features'].columns)} engineered")
    logger.info(f"Skip Predictor AUC: {models['skip_metrics']['val_auc']:.4f}")
    logger.info(f"Session Forecaster R²: {models['session_metrics']['val_r2']:.4f}")
    logger.info(f"Avg Retention (Period 1): {analysis['cohort_stats']['avg_period_1_retention']:.1f}%")
    logger.info(f"A/B Test Significant: {ab_results['is_significant']}")
    logger.info("="*60)
    
    # Save results
    all_results = {
        'model_metrics': {
            'skip_predictor': models['skip_metrics'],
            'session_forecaster': models['session_metrics']
        },
        'analysis': {
            'cohort_stats': analysis['cohort_stats'],
            'funnel_metrics': analysis['funnel_metrics'],
            'activation_metrics': analysis['activation_metrics']
        },
        'ab_testing': ab_results
    }
    save_results(all_results, 'data/processed/pipeline_results.json')


def main():
    parser = argparse.ArgumentParser(description="Music Streaming Analytics Platform")
    parser.add_argument('--full-pipeline', action='store_true', help='Run complete pipeline')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train-models', action='store_true', help='Train ML models')
    parser.add_argument('--run-analysis', action='store_true', help='Run cohort/funnel analysis')
    parser.add_argument('--sessions', type=int, default=100000, help='Number of sessions')
    parser.add_argument('--users', type=int, default=1000, help='Number of users')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    if args.full_pipeline or not any([args.generate_data, args.train_models, args.run_analysis]):
        run_full_pipeline(args.sessions, args.users)
    else:
        if args.generate_data:
            generate_data(args.sessions, args.users)


if __name__ == "__main__":
    main()
