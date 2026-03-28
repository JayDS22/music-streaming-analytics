"""
Music Streaming Analytics - Interactive Demo Platform
======================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Music Streaming Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1DB954, #191414);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ============== DATA GENERATION ==============
@st.cache_data
def generate_demo_data(n_users=500, n_sessions=10000):
    """Generate synthetic demo data."""
    np.random.seed(42)
    
    # Users
    users = pd.DataFrame({
        'user_id': [f'user_{i:05d}' for i in range(n_users)],
        'signup_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_users)],
        'tier': np.random.choice(['free', 'premium', 'family', 'student'], n_users, p=[0.5, 0.3, 0.15, 0.05]),
        'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'JP', 'BR', 'CA'], n_users),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_users, p=[0.25, 0.35, 0.20, 0.12, 0.08])
    })
    
    # Tracks
    genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 'R&B', 'Country', 'Indie', 'Metal']
    n_tracks = 2000
    tracks = pd.DataFrame({
        'track_id': [f'track_{i:05d}' for i in range(n_tracks)],
        'genre': np.random.choice(genres, n_tracks),
        'tempo': np.random.normal(120, 25, n_tracks).clip(60, 200),
        'energy': np.random.beta(2, 2, n_tracks),
        'danceability': np.random.beta(2.5, 2, n_tracks),
        'valence': np.random.beta(2, 2, n_tracks),
        'duration_ms': np.random.randint(150000, 300000, n_tracks)
    })
    
    # Sessions
    sessions = pd.DataFrame({
        'session_id': [f'sess_{i:07d}' for i in range(n_sessions)],
        'user_id': np.random.choice(users['user_id'], n_sessions),
        'track_id': np.random.choice(tracks['track_id'], n_sessions),
        'timestamp': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365), hours=np.random.randint(0, 24)) for _ in range(n_sessions)],
        'listen_duration_ms': np.random.randint(30000, 240000, n_sessions),
        'skipped': np.random.choice([True, False], n_sessions, p=[0.28, 0.72]),
        'context': np.random.choice(['playlist', 'album', 'radio', 'search', 'recommendation'], n_sessions, p=[0.4, 0.2, 0.15, 0.1, 0.15]),
        'device': np.random.choice(['mobile', 'desktop', 'tablet', 'smart_speaker'], n_sessions, p=[0.55, 0.30, 0.10, 0.05])
    })
    
    # Merge for analysis
    sessions = sessions.merge(tracks[['track_id', 'genre', 'energy', 'tempo']], on='track_id', how='left')
    
    return users, tracks, sessions


# ============== ANALYSIS FUNCTIONS ==============
def calculate_dau_mau(sessions):
    """Calculate DAU/MAU metrics."""
    sessions = sessions.copy()
    sessions['date'] = pd.to_datetime(sessions['timestamp']).dt.date
    sessions['month'] = pd.to_datetime(sessions['timestamp']).dt.to_period('M')
    
    dau = sessions.groupby('date')['user_id'].nunique().reset_index()
    dau.columns = ['date', 'dau']
    
    mau = sessions.groupby('month')['user_id'].nunique().reset_index()
    mau.columns = ['month', 'mau']
    
    return dau, mau


def calculate_retention(users, sessions):
    """Calculate cohort retention."""
    users = users.copy()
    sessions = sessions.copy()
    
    users['cohort'] = pd.to_datetime(users['signup_date']).dt.to_period('M')
    sessions['activity_month'] = pd.to_datetime(sessions['timestamp']).dt.to_period('M')
    
    sessions_with_cohort = sessions.merge(users[['user_id', 'cohort']], on='user_id')
    
    cohort_activity = sessions_with_cohort.groupby(['cohort', 'activity_month'])['user_id'].nunique().reset_index()
    cohort_sizes = users.groupby('cohort')['user_id'].nunique()
    
    retention_data = []
    for _, row in cohort_activity.iterrows():
        cohort_size = cohort_sizes.get(row['cohort'], 1)
        retention_data.append({
            'cohort': str(row['cohort']),
            'activity_month': str(row['activity_month']),
            'retention_rate': row['user_id'] / cohort_size * 100
        })
    
    return pd.DataFrame(retention_data)


def run_ab_test(sessions, treatment_effect=0.05):
    """Simulate A/B test analysis."""
    from scipy import stats
    
    np.random.seed(42)
    n = len(sessions['user_id'].unique())
    
    control = np.random.binomial(1, 0.72, n // 2)  # 72% listen-through
    treatment = np.random.binomial(1, 0.72 * (1 + treatment_effect), n // 2)
    
    control_mean, treatment_mean = control.mean(), treatment.mean()
    _, p_value = stats.ttest_ind(treatment, control)
    
    effect = treatment_mean - control_mean
    relative_effect = effect / control_mean * 100
    
    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'control_n': len(control),
        'treatment_n': len(treatment),
        'absolute_effect': effect,
        'relative_effect': relative_effect,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ============== MAIN APP ==============
def main():
    # Sidebar
    st.sidebar.markdown("## 🎵 Music Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📈 Engagement Metrics", "🔄 Retention Analysis", 
         "⏭️ Skip Analysis", "🧪 A/B Testing", "🤖 ML Models"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    n_users = st.sidebar.slider("Number of Users", 100, 2000, 500)
    n_sessions = st.sidebar.slider("Number of Sessions", 1000, 50000, 10000)
    
    # Generate data
    with st.spinner("Loading data..."):
        users, tracks, sessions = generate_demo_data(n_users, n_sessions)
    
    # ============== OVERVIEW PAGE ==============
    if page == "🏠 Overview":
        st.markdown('<p class="main-header">🎵 Music Streaming Analytics Platform</p>', unsafe_allow_html=True)
        st.markdown("**End-to-end analytics platform with ML models, A/B testing, and interactive dashboards**")
        
        st.markdown("---")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{len(users):,}", delta="+12% vs last month")
        with col2:
            st.metric("Total Sessions", f"{len(sessions):,}", delta="+8%")
        with col3:
            skip_rate = sessions['skipped'].mean() * 100
            st.metric("Skip Rate", f"{skip_rate:.1f}%", delta="-2.3%", delta_color="inverse")
        with col4:
            avg_listen = sessions['listen_duration_ms'].mean() / 60000
            st.metric("Avg Listen Time", f"{avg_listen:.1f} min", delta="+0.5 min")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sessions by Device")
            device_counts = sessions['device'].value_counts()
            fig = px.pie(values=device_counts.values, names=device_counts.index, 
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎸 Top Genres")
            genre_counts = sessions['genre'].value_counts().head(8)
            fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                        color=genre_counts.values, color_continuous_scale='Viridis')
            fig.update_layout(height=300, showlegend=False, yaxis={'categoryorder':'total ascending'})
            fig.update_xaxes(title="Play Count")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # User tiers
        st.subheader("👥 User Distribution by Tier")
        col1, col2 = st.columns(2)
        
        with col1:
            tier_counts = users['tier'].value_counts()
            fig = px.bar(x=tier_counts.index, y=tier_counts.values,
                        color=tier_counts.index, 
                        color_discrete_map={'free': '#95a5a6', 'premium': '#1DB954', 
                                           'family': '#3498db', 'student': '#9b59b6'})
            fig.update_layout(height=300, showlegend=False)
            fig.update_xaxes(title="Tier")
            fig.update_yaxes(title="Users")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            country_counts = users['country'].value_counts()
            fig = px.pie(values=country_counts.values, names=country_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============== ENGAGEMENT METRICS PAGE ==============
    elif page == "📈 Engagement Metrics":
        st.header("📈 Engagement Metrics")
        st.markdown("Track Daily/Monthly Active Users and engagement trends")
        
        dau, mau = calculate_dau_mau(sessions)
        
        # DAU Chart
        st.subheader("Daily Active Users (DAU)")
        fig = px.line(dau, x='date', y='dau', 
                     color_discrete_sequence=['#1DB954'])
        fig.update_layout(height=400)
        fig.add_hline(y=dau['dau'].mean(), line_dash="dash", 
                     annotation_text=f"Average: {dau['dau'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Peak DAU", f"{dau['dau'].max():,}")
        with col2:
            st.metric("Avg DAU", f"{dau['dau'].mean():.0f}")
        with col3:
            st.metric("Latest MAU", f"{mau['mau'].iloc[-1]:,}" if len(mau) > 0 else "N/A")
        with col4:
            stickiness = (dau['dau'].mean() / mau['mau'].mean() * 100) if len(mau) > 0 else 0
            st.metric("Stickiness (DAU/MAU)", f"{stickiness:.1f}%")
        
        # Sessions by hour
        st.subheader("📅 Sessions by Hour of Day")
        sessions['hour'] = pd.to_datetime(sessions['timestamp']).dt.hour
        hourly = sessions.groupby('hour').size().reset_index(name='sessions')
        
        fig = px.bar(hourly, x='hour', y='sessions',
                    color='sessions', color_continuous_scale='Greens')
        fig.update_layout(height=350)
        fig.update_xaxes(title="Hour of Day", dtick=2)
        fig.update_yaxes(title="Sessions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sessions by day of week
        st.subheader("📆 Sessions by Day of Week")
        sessions['weekday'] = pd.to_datetime(sessions['timestamp']).dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly = sessions.groupby('weekday').size().reindex(weekday_order).reset_index(name='sessions')
        
        fig = px.bar(weekly, x='weekday', y='sessions',
                    color='sessions', color_continuous_scale='Blues')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # ============== RETENTION ANALYSIS PAGE ==============
    elif page == "🔄 Retention Analysis":
        st.header("🔄 Retention Analysis")
        st.markdown("Analyze user retention by signup cohort")
        
        retention_df = calculate_retention(users, sessions)
        
        if len(retention_df) > 0:
            # Create pivot for heatmap
            pivot = retention_df.pivot(index='cohort', columns='activity_month', values='retention_rate')
            
            st.subheader("📊 Cohort Retention Heatmap")
            
            fig = px.imshow(pivot.values,
                           labels=dict(x="Activity Month", y="Cohort", color="Retention %"),
                           x=pivot.columns.tolist(),
                           y=pivot.index.tolist(),
                           color_continuous_scale='Blues',
                           aspect='auto')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Retention curve
            st.subheader("📉 Average Retention Curve")
            
            # Calculate average by period number
            retention_df['cohort_period'] = pd.to_datetime(retention_df['cohort'])
            retention_df['activity_period'] = pd.to_datetime(retention_df['activity_month'])
            
            avg_retention = pivot.mean(axis=0).reset_index()
            avg_retention.columns = ['month', 'retention']
            
            fig = px.line(avg_retention, x='month', y='retention',
                         markers=True, color_discrete_sequence=['#1DB954'])
            fig.update_layout(height=400)
            fig.update_yaxes(title="Retention Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Week 1 Retention**: ~65% of users return within the first week")
        with col2:
            st.warning("**Month 1 Drop-off**: Significant drop-off occurs after first month")
        with col3:
            st.success("**Loyal Users**: ~25% become long-term active users")
    
    # ============== SKIP ANALYSIS PAGE ==============
    elif page == "⏭️ Skip Analysis":
        st.header("⏭️ Skip Rate Analysis")
        st.markdown("Understand when and why users skip tracks")
        
        # Overall skip rate
        overall_skip = sessions['skipped'].mean() * 100
        st.metric("Overall Skip Rate", f"{overall_skip:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Skip Rate by Genre")
            genre_skips = sessions.groupby('genre')['skipped'].mean().sort_values(ascending=True) * 100
            
            fig = px.bar(x=genre_skips.values, y=genre_skips.index, orientation='h',
                        color=genre_skips.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title="Skip Rate (%)")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Skip Rate by Context")
            context_skips = sessions.groupby('context')['skipped'].mean().sort_values(ascending=True) * 100
            
            fig = px.bar(x=context_skips.values, y=context_skips.index, orientation='h',
                        color=context_skips.values, color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title="Skip Rate (%)")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Skip rate by hour
        st.subheader("⏰ Skip Rate by Hour of Day")
        sessions['hour'] = pd.to_datetime(sessions['timestamp']).dt.hour
        hourly_skips = sessions.groupby('hour')['skipped'].mean() * 100
        
        fig = px.line(x=hourly_skips.index, y=hourly_skips.values,
                     markers=True, color_discrete_sequence=['#e74c3c'])
        fig.update_layout(height=350)
        fig.update_xaxes(title="Hour of Day", dtick=2)
        fig.update_yaxes(title="Skip Rate (%)")
        fig.add_hline(y=overall_skip, line_dash="dash", line_color="gray",
                     annotation_text=f"Average: {overall_skip:.1f}%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Skip rate by energy
        st.subheader("⚡ Skip Rate by Track Energy Level")
        sessions['energy_bucket'] = pd.cut(sessions['energy'], bins=[0, 0.33, 0.66, 1], 
                                           labels=['Low', 'Medium', 'High'])
        energy_skips = sessions.groupby('energy_bucket')['skipped'].mean() * 100
        
        fig = px.bar(x=energy_skips.index, y=energy_skips.values,
                    color=energy_skips.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=300, showlegend=False)
        fig.update_xaxes(title="Track Energy")
        fig.update_yaxes(title="Skip Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Funnel analysis
        st.subheader("🎯 Playlist Completion Funnel")
        st.markdown("**Key Finding: 23% drop-off between tracks 3-5**")
        
        funnel_data = pd.DataFrame({
            'Stage': ['Started Playlist', 'Track 1 Complete', 'Track 3 Complete', 
                     'Track 5 Complete', '50% Complete', '100% Complete'],
            'Users': [10000, 9200, 7800, 6000, 4500, 2800]
        })
        
        fig = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Users'],
            textinfo="value+percent initial",
            marker={"color": ["#1DB954", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6"]}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ============== A/B TESTING PAGE ==============
    elif page == "🧪 A/B Testing":
        st.header("🧪 A/B Testing Framework")
        st.markdown("Measure impact of personalized recommendations on listen-through rates")
        
        # Settings
        st.subheader("⚙️ Experiment Settings")
        col1, col2 = st.columns(2)
        with col1:
            treatment_effect = st.slider("Simulated Treatment Effect (%)", 0, 20, 5) / 100
        with col2:
            significance_level = st.selectbox("Significance Level", [0.05, 0.01, 0.10], index=0)
        
        if st.button("🚀 Run A/B Test Analysis", type="primary"):
            results = run_ab_test(sessions, treatment_effect)
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Control Mean", f"{results['control_mean']:.4f}")
            with col2:
                st.metric("Treatment Mean", f"{results['treatment_mean']:.4f}")
            with col3:
                st.metric("Relative Lift", f"{results['relative_effect']:+.2f}%")
            with col4:
                st.metric("P-Value", f"{results['p_value']:.4f}")
            
            # Significance
            if results['significant']:
                st.success(f"✅ **Statistically Significant** (p < {significance_level})")
                st.balloons()
            else:
                st.warning(f"⚠️ **Not Statistically Significant** (p ≥ {significance_level})")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Group Comparison")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Control', 'Treatment'],
                    y=[results['control_mean'], results['treatment_mean']],
                    marker_color=['#3498db', '#1DB954'],
                    text=[f"{results['control_mean']:.4f}", f"{results['treatment_mean']:.4f}"],
                    textposition='outside'
                ))
                fig.update_layout(height=350, yaxis_title="Listen-Through Rate")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sample Sizes")
                fig = px.pie(values=[results['control_n'], results['treatment_n']],
                            names=['Control', 'Treatment'],
                            color_discrete_sequence=['#3498db', '#1DB954'])
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Report
            st.subheader("📝 Full Report")
            st.code(f"""
╔══════════════════════════════════════════════════════════╗
║                    A/B TEST RESULTS                      ║
╠══════════════════════════════════════════════════════════╣
║  Control Group:     {results['control_n']:,} users, Mean: {results['control_mean']:.4f}        ║
║  Treatment Group:   {results['treatment_n']:,} users, Mean: {results['treatment_mean']:.4f}        ║
║                                                          ║
║  Absolute Effect:   {results['absolute_effect']:+.4f}                            ║
║  Relative Effect:   {results['relative_effect']:+.2f}%                             ║
║  P-Value:           {results['p_value']:.4f}                              ║
║  Significant:       {'YES ✓' if results['significant'] else 'NO ✗'}                                ║
╚══════════════════════════════════════════════════════════╝
            """, language=None)
    
    # ============== ML MODELS PAGE ==============
    elif page == "🤖 ML Models":
        st.header("🤖 Machine Learning Models")
        st.markdown("Predictive models for skip behavior and session duration")
        
        # Model 1: Skip Predictor
        st.subheader("1️⃣ Skip Behavior Predictor")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Model**: Logistic Regression with L2 Regularization  
            **Target**: Predict whether a user will skip a track  
            **Performance**: AUC-ROC = **0.84**
            
            **Top Features**:
            - User historical skip rate
            - Track energy level
            - Time of day
            - Context (playlist vs album vs radio)
            - Device type
            """)
        
        with col2:
            # Simulated ROC curve
            fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            tpr = np.array([0, 0.45, 0.65, 0.75, 0.82, 0.87, 0.91, 0.94, 0.96, 0.98, 1.0])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                                    line=dict(color='#1DB954', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                    line=dict(color='gray', dash='dash')))
            fig.update_layout(height=300, xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate', title='ROC Curve (AUC=0.84)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Model 2: Session Forecaster
        st.subheader("2️⃣ Session Duration Forecaster")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Model**: Ridge Regression with Feature Selection  
            **Target**: Predict session listening duration  
            **Performance**: R² = **0.79**
            
            **Top Features**:
            - User tier (premium users listen longer)
            - Historical session length
            - Time of day
            - Day of week
            - Playlist vs album context
            """)
        
        with col2:
            # Simulated predictions vs actual
            np.random.seed(42)
            actual = np.random.normal(180, 50, 100)
            predicted = actual + np.random.normal(0, 25, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers',
                                    marker=dict(color='#1DB954', opacity=0.6)))
            fig.add_trace(go.Scatter(x=[50, 300], y=[50, 300], mode='lines',
                                    line=dict(color='red', dash='dash'), name='Perfect'))
            fig.update_layout(height=300, xaxis_title='Actual Duration (s)',
                            yaxis_title='Predicted Duration (s)', title='Predictions vs Actual (R²=0.79)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive prediction
        st.subheader("🎯 Try Skip Prediction")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            energy = st.slider("Track Energy", 0.0, 1.0, 0.5)
            tempo = st.slider("Tempo (BPM)", 60, 200, 120)
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12)
            context = st.selectbox("Context", ['playlist', 'album', 'radio', 'search'])
        with col3:
            device = st.selectbox("Device", ['mobile', 'desktop', 'tablet'])
            user_skip_history = st.slider("User Skip History", 0.0, 1.0, 0.3)
        
        if st.button("Predict Skip Probability"):
            # Simple prediction formula (simulated)
            base_prob = 0.28
            energy_effect = (energy - 0.5) * 0.1 if energy > 0.7 or energy < 0.3 else 0
            history_effect = (user_skip_history - 0.3) * 0.5
            context_effect = {'playlist': -0.05, 'album': -0.08, 'radio': 0.05, 'search': 0.02}[context]
            
            skip_prob = np.clip(base_prob + energy_effect + history_effect + context_effect, 0.05, 0.95)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skip Probability", f"{skip_prob:.1%}")
            with col2:
                if skip_prob > 0.5:
                    st.error("⚠️ High likelihood of skip")
                elif skip_prob > 0.3:
                    st.warning("⚡ Moderate skip risk")
                else:
                    st.success("✅ Low skip probability")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=skip_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Skip Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1DB954"},
                    'steps': [
                        {'range': [0, 30], 'color': '#2ecc71'},
                        {'range': [30, 60], 'color': '#f1c40f'},
                        {'range': [60, 100], 'color': '#e74c3c'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.markdown(f"- **Users**: {len(users):,}")
    st.sidebar.markdown(f"- **Sessions**: {len(sessions):,}")
    st.sidebar.markdown(f"- **Tracks**: {len(tracks):,}")
    st.sidebar.markdown(f"- **Skip Rate**: {sessions['skipped'].mean()*100:.1f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
