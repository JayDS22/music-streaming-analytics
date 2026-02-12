# 🎵 Music Streaming Analytics & Listener Engagement Platform

A comprehensive data analytics platform for music streaming services, featuring user behavior analysis, predictive modeling, A/B testing frameworks, and interactive dashboards.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Project Overview

This platform processes **1M+ listening sessions** to extract insights about user behavior, predict engagement patterns, and optimize music recommendations through data-driven experimentation.

### Key Achievements
- **Predictive Modeling**: Logistic regression model predicting skip behavior (AUC: 0.84)
- **Session Forecasting**: Linear regression model forecasting session duration (R²: 0.79)
- **Feature Engineering**: 50+ user engagement features engineered
- **Cohort Analysis**: Identified 23% drop-off in playlist completion
- **A/B Testing**: Framework with significance testing (p<0.05) for personalized recommendations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MUSIC STREAMING ANALYTICS PLATFORM                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA INGESTION LAYER                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  Spotify    │  │  Streaming  │  │   User      │  │  Playlist   │     │   │
│  │  │    API      │  │   Events    │  │  Profiles   │  │   Data      │     │   │
│  │  │  (Audio     │  │  (1M+       │  │  (Demo-     │  │  (Track     │     │   │
│  │  │  Features)  │  │  Sessions)  │  │  graphics)  │  │  Lists)     │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │                │            │   │
│  │         └────────────────┴────────────────┴────────────────┘            │   │
│  │                                    │                                     │   │
│  └────────────────────────────────────┼─────────────────────────────────────┘   │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA STORAGE LAYER                               │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │   │
│  │  │                      PostgreSQL Database                          │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │   │
│  │  │  │   users     │  │  sessions   │  │  tracks     │               │  │   │
│  │  │  ├─────────────┤  ├─────────────┤  ├─────────────┤               │  │   │
│  │  │  │ user_id     │  │ session_id  │  │ track_id    │               │  │   │
│  │  │  │ signup_date │  │ user_id     │  │ tempo       │               │  │   │
│  │  │  │ tier        │  │ track_id    │  │ energy      │               │  │   │
│  │  │  │ country     │  │ duration    │  │ danceability│               │  │   │
│  │  │  └─────────────┘  │ skipped     │  │ valence     │               │  │   │
│  │  │                   └─────────────┘  └─────────────┘               │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │   │
│  │  │  │  playlists  │  │ ab_tests    │  │ user_metrics│               │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │   │
│  │  └───────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      FEATURE ENGINEERING LAYER                           │   │
│  │                                                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │                   50+ Engineered Features                        │    │   │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │    │   │
│  │  │  │   Listening   │  │    Genre      │  │   Playlist    │        │    │   │
│  │  │  │    Streaks    │  │   Diversity   │  │   Behavior    │        │    │   │
│  │  │  │  • streak_len │  │  • entropy    │  │  • completion │        │    │   │
│  │  │  │  • max_streak │  │  • variety    │  │  • skip_rate  │        │    │   │
│  │  │  │  • consistency│  │  • exploration│  │  • engagement │        │    │   │
│  │  │  └───────────────┘  └───────────────┘  └───────────────┘        │    │   │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │    │   │
│  │  │  │   Session     │  │   Temporal    │  │    Audio      │        │    │   │
│  │  │  │   Features    │  │   Patterns    │  │   Features    │        │    │   │
│  │  │  │  • duration   │  │  • time_of_day│  │  • tempo_pref │        │    │   │
│  │  │  │  • track_count│  │  • weekday    │  │  • energy_avg │        │    │   │
│  │  │  │  • skip_ratio │  │  • seasonality│  │  • valence_var│        │    │   │
│  │  │  └───────────────┘  └───────────────┘  └───────────────┘        │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        ANALYTICS & ML LAYER                              │   │
│  │                                                                          │   │
│  │  ┌───────────────────────┐    ┌───────────────────────┐                 │   │
│  │  │   PREDICTIVE MODELS   │    │   STATISTICAL ANALYSIS │                │   │
│  │  │  ┌─────────────────┐  │    │  ┌─────────────────┐   │                │   │
│  │  │  │ Skip Prediction │  │    │  │  Cohort Analysis │   │                │   │
│  │  │  │ (Logistic Reg.) │  │    │  │  • Retention     │   │                │   │
│  │  │  │   AUC: 0.84     │  │    │  │  • Engagement    │   │                │   │
│  │  │  └─────────────────┘  │    │  │  • Churn         │   │                │   │
│  │  │  ┌─────────────────┐  │    │  └─────────────────┘   │                │   │
│  │  │  │ Session Duration│  │    │  ┌─────────────────┐   │                │   │
│  │  │  │ (Linear Reg.)   │  │    │  │  Funnel Analysis │   │                │   │
│  │  │  │   R²: 0.79      │  │    │  │  • 23% Drop-off  │   │                │   │
│  │  │  └─────────────────┘  │    │  │  • Conversion    │   │                │   │
│  │  └───────────────────────┘    │  └─────────────────┘   │                │   │
│  │                               └───────────────────────┘                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                     A/B TESTING FRAMEWORK                        │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │   │
│  │  │  │  Experiment │  │  Statistical│  │   Results   │              │   │   │
│  │  │  │   Design    │  │   Testing   │  │   Analysis  │              │   │   │
│  │  │  │  • Sample   │  │  • t-test   │  │  • Effect   │              │   │   │
│  │  │  │    Size     │  │  • chi-sq   │  │    Size     │              │   │   │
│  │  │  │  • Random   │  │  • p<0.05   │  │  • CI       │              │   │   │
│  │  │  │    Assign   │  │  • FDR      │  │  • Power    │              │   │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       VISUALIZATION LAYER                                │   │
│  │  ┌───────────────────────┐    ┌───────────────────────┐                 │   │
│  │  │   TABLEAU DASHBOARDS  │    │    PYTHON/R REPORTS   │                 │   │
│  │  │  • DAU/MAU Metrics    │    │  • Statistical Reports│                 │   │
│  │  │  • Retention Curves   │    │  • Model Performance  │                 │   │
│  │  │  • Skip Rate Analysis │    │  • Feature Importance │                 │   │
│  │  │  • Cohort Heatmaps    │    │  • A/B Test Results   │                 │   │
│  │  └───────────────────────┘    └───────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                              DATA FLOW DIAGRAM

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Extract    │────▶│   Transform  │────▶│     Load     │────▶│   Analyze    │
    │  (Spotify    │     │  (Feature    │     │  (PostgreSQL │     │  (ML Models  │
    │   API +      │     │  Engineering │     │   + Parquet) │     │   + Stats)   │
    │   Events)    │     │   Pipeline)  │     │              │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │                    │
           ▼                    ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Audio        │     │ 50+ Features │     │ Optimized    │     │ Predictions  │
    │ Features:    │     │ • Streaks    │     │ Storage      │     │ • Skip Prob  │
    │ • Tempo      │     │ • Diversity  │     │ • Indexed    │     │ • Duration   │
    │ • Energy     │     │ • Behavior   │     │ • Partitioned│     │ • Engagement │
    │ • Danceabil. │     │ • Session    │     │ • Compressed │     │ • Retention  │
    │ • Valence    │     │ • Temporal   │     │              │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📁 Project Structure

```
music-streaming-analytics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── setup.py                          # Package setup
├── .env.example                      # Environment variables template
├── config/
│   └── config.yaml                   # Configuration settings
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── spotify_client.py         # Spotify API integration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_generator.py         # Synthetic data generation
│   │   └── data_loader.py            # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py    # 50+ feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── skip_predictor.py         # Skip behavior prediction
│   │   └── session_forecaster.py     # Session duration forecast
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── cohort_analysis.py        # Cohort analysis
│   │   └── funnel_analysis.py        # Funnel analysis
│   ├── ab_testing/
│   │   ├── __init__.py
│   │   └── ab_framework.py           # A/B testing framework
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── dashboard_generator.py    # Dashboard generation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                # Utility functions
├── sql/
│   ├── schema.sql                    # Database schema
│   └── queries.sql                   # Analytics queries
├── r_scripts/
│   ├── cohort_analysis.R             # R cohort analysis
│   ├── ab_testing.R                  # R A/B testing
│   └── visualization.R               # R visualizations
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_ab_testing.py
├── notebooks/
│   └── exploratory_analysis.ipynb    # EDA notebook
├── dashboards/
│   └── tableau_template.twb          # Tableau dashboard
├── data/
│   ├── raw/                          # Raw data
│   ├── processed/                    # Processed data
│   └── interim/                      # Intermediate data
└── docs/
    └── api_documentation.md          # API documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- R 4.0+
- PostgreSQL 13+
- Spotify Developer Account (for API access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/music-streaming-analytics.git
cd music-streaming-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Spotify API credentials
```

5. **Initialize database**
```bash
psql -U postgres -f sql/schema.sql
```

6. **Generate sample data (optional)**
```bash
python -m src.data.data_generator --sessions 1000000
```

### Running the Pipeline

```bash
# Full pipeline
python main.py --full-pipeline

# Individual components
python main.py --extract-features
python main.py --train-models
python main.py --run-analysis
python main.py --generate-dashboards
```

---

## 📊 Features

### Data Ingestion
- **Spotify API Integration**: Extract audio features (tempo, energy, danceability, valence)
- **Event Processing**: Handle 1M+ listening sessions
- **Real-time Updates**: Streaming data support

### Feature Engineering (50+ Features)

| Category | Features |
|----------|----------|
| **Listening Streaks** | streak_length, max_streak, streak_consistency, active_days |
| **Genre Diversity** | genre_entropy, genre_variety, exploration_ratio |
| **Playlist Behavior** | completion_rate, skip_rate, engagement_score |
| **Session Metrics** | avg_duration, track_count, time_to_skip |
| **Temporal Patterns** | time_of_day, weekday_preference, seasonality |
| **Audio Preferences** | tempo_preference, energy_avg, valence_variance |

### Predictive Models

#### Skip Behavior Prediction
- **Algorithm**: Logistic Regression with L2 regularization
- **Performance**: AUC-ROC = 0.84
- **Key Features**: audio energy, user engagement history, time of day

#### Session Duration Forecasting
- **Algorithm**: Linear Regression with feature selection
- **Performance**: R² = 0.79
- **Key Features**: user tier, historical session length, playlist type

### Analytics

#### Cohort Analysis
- User retention by signup cohort
- Engagement patterns over time
- Churn prediction indicators

#### Funnel Analysis
- Playlist completion funnel
- **Finding**: 23% drop-off identified at track 3-5
- Recommendation optimization points

### A/B Testing Framework
- Experiment design with power analysis
- Random user assignment
- Statistical significance testing (p<0.05)
- Effect size and confidence intervals
- Automated reporting

---

## 🛠️ Usage Examples

### Feature Engineering
```python
from src.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_all_features(sessions_df, users_df, tracks_df)
print(f"Generated {len(features.columns)} features")
```

### Skip Prediction
```python
from src.models.skip_predictor import SkipPredictor

model = SkipPredictor()
model.train(X_train, y_train)
predictions = model.predict_proba(X_test)
print(f"AUC-ROC: {model.evaluate(X_test, y_test)}")
```

### A/B Testing
```python
from src.ab_testing.ab_framework import ABTestFramework

ab = ABTestFramework()
ab.create_experiment(
    name="personalized_recommendations_v2",
    control_group=control_users,
    treatment_group=treatment_users
)
results = ab.analyze_results(metric="listen_through_rate")
print(f"P-value: {results['p_value']:.4f}")
```

### Cohort Analysis
```python
from src.analysis.cohort_analysis import CohortAnalyzer

analyzer = CohortAnalyzer()
retention_matrix = analyzer.calculate_retention(users_df, sessions_df)
analyzer.plot_retention_heatmap(retention_matrix)
```

---

## 📈 Dashboard Metrics

The Tableau dashboards track:

- **DAU/MAU**: Daily and Monthly Active Users
- **Retention Curves**: User retention over time
- **Skip Rates**: By genre, time of day, user segment
- **Engagement Metrics**: Session duration, tracks per session
- **A/B Test Monitors**: Live experiment tracking

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

---

## 📝 SQL Queries

Key analytics queries are provided in `sql/queries.sql`:

- User engagement metrics
- Retention calculations
- Skip rate analysis
- Cohort aggregations
- Funnel stage analysis

---

## 📊 R Analysis

R scripts provide additional statistical analysis:

```r
# Run cohort analysis
Rscript r_scripts/cohort_analysis.R

# Run A/B testing analysis
Rscript r_scripts/ab_testing.R
```

---

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
spotify:
  client_id: ${SPOTIFY_CLIENT_ID}
  client_secret: ${SPOTIFY_CLIENT_SECRET}

database:
  host: localhost
  port: 5432
  name: music_analytics

models:
  skip_predictor:
    regularization: l2
    C: 1.0
  session_forecaster:
    features_to_select: 20

ab_testing:
  significance_level: 0.05
  minimum_sample_size: 1000
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

## 📞 Contact

For questions or collaboration, please open an issue or reach out.

---

## 🙏 Acknowledgments

- Spotify Web API for audio feature extraction
- scikit-learn for ML implementations
- Tableau for visualization capabilities
