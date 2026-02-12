-- Music Streaming Analytics - SQL Queries
-- ==========================================

-- ============================================
-- USER ENGAGEMENT METRICS
-- ============================================

-- DAU/MAU with stickiness ratio
WITH daily_users AS (
    SELECT 
        DATE(timestamp) as date,
        DATE_TRUNC('month', timestamp) as month,
        user_id
    FROM sessions
),
dau AS (
    SELECT date, COUNT(DISTINCT user_id) as dau
    FROM daily_users
    GROUP BY date
),
mau AS (
    SELECT month, COUNT(DISTINCT user_id) as mau
    FROM daily_users
    GROUP BY month
)
SELECT 
    d.date,
    d.dau,
    m.mau,
    ROUND(d.dau::DECIMAL / m.mau * 100, 2) as stickiness_ratio
FROM dau d
JOIN mau m ON DATE_TRUNC('month', d.date) = m.month
ORDER BY d.date;

-- User engagement segments
SELECT 
    CASE 
        WHEN session_count >= 100 THEN 'Power User'
        WHEN session_count >= 30 THEN 'Active'
        WHEN session_count >= 10 THEN 'Casual'
        ELSE 'Light'
    END as user_segment,
    COUNT(*) as user_count,
    ROUND(AVG(total_listen_hours), 2) as avg_listen_hours,
    ROUND(AVG(skip_rate) * 100, 2) as avg_skip_rate
FROM (
    SELECT 
        user_id,
        COUNT(*) as session_count,
        SUM(listen_duration_ms) / 3600000.0 as total_listen_hours,
        AVG(CASE WHEN skipped THEN 1 ELSE 0 END) as skip_rate
    FROM sessions
    GROUP BY user_id
) user_stats
GROUP BY user_segment
ORDER BY user_count DESC;

-- ============================================
-- RETENTION ANALYSIS
-- ============================================

-- Cohort retention matrix (Monthly)
WITH user_cohorts AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', signup_date) as cohort_month
    FROM users
),
user_activities AS (
    SELECT DISTINCT
        s.user_id,
        DATE_TRUNC('month', s.timestamp) as activity_month
    FROM sessions s
),
cohort_data AS (
    SELECT 
        uc.cohort_month,
        ua.activity_month,
        COUNT(DISTINCT uc.user_id) as users
    FROM user_cohorts uc
    JOIN user_activities ua ON uc.user_id = ua.user_id
    WHERE ua.activity_month >= uc.cohort_month
    GROUP BY uc.cohort_month, ua.activity_month
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT user_id) as cohort_size
    FROM user_cohorts
    GROUP BY cohort_month
)
SELECT 
    cd.cohort_month,
    cs.cohort_size,
    cd.activity_month,
    EXTRACT(MONTH FROM AGE(cd.activity_month, cd.cohort_month)) as months_since_signup,
    cd.users,
    ROUND(cd.users::DECIMAL / cs.cohort_size * 100, 2) as retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.activity_month;

-- ============================================
-- SKIP RATE ANALYSIS
-- ============================================

-- Skip rate by genre
SELECT 
    t.genre,
    COUNT(*) as total_plays,
    SUM(CASE WHEN s.skipped THEN 1 ELSE 0 END) as skips,
    ROUND(AVG(CASE WHEN s.skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate,
    ROUND(AVG(s.listen_duration_ms) / 1000, 2) as avg_listen_seconds
FROM sessions s
JOIN tracks t ON s.track_id = t.track_id
GROUP BY t.genre
ORDER BY skip_rate DESC;

-- Skip rate by time of day
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour,
    COUNT(*) as sessions,
    ROUND(AVG(CASE WHEN skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM sessions
GROUP BY hour
ORDER BY hour;

-- Skip rate by audio features (energy levels)
SELECT 
    CASE 
        WHEN t.energy < 0.3 THEN 'Low Energy'
        WHEN t.energy < 0.7 THEN 'Medium Energy'
        ELSE 'High Energy'
    END as energy_level,
    COUNT(*) as plays,
    ROUND(AVG(CASE WHEN s.skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM sessions s
JOIN tracks t ON s.track_id = t.track_id
GROUP BY energy_level
ORDER BY skip_rate;

-- ============================================
-- FUNNEL ANALYSIS
-- ============================================

-- Playlist completion funnel
WITH playlist_sessions AS (
    SELECT 
        user_id,
        DATE(timestamp) as session_date,
        COUNT(*) as tracks_played,
        SUM(CASE WHEN skipped THEN 1 ELSE 0 END) as tracks_skipped
    FROM sessions
    WHERE context = 'playlist'
    GROUP BY user_id, DATE(timestamp)
)
SELECT 
    'Started Playlist' as stage,
    COUNT(*) as users
FROM playlist_sessions

UNION ALL

SELECT 
    'Played 3+ Tracks' as stage,
    COUNT(*) as users
FROM playlist_sessions
WHERE tracks_played >= 3

UNION ALL

SELECT 
    'Played 5+ Tracks' as stage,
    COUNT(*) as users
FROM playlist_sessions
WHERE tracks_played >= 5

UNION ALL

SELECT 
    'Low Skip Rate (<20%)' as stage,
    COUNT(*) as users
FROM playlist_sessions
WHERE tracks_skipped::DECIMAL / NULLIF(tracks_played, 0) < 0.2;

-- ============================================
-- A/B TEST QUERIES
-- ============================================

-- A/B test results summary
SELECT 
    a.test_name,
    a.variant,
    COUNT(DISTINCT a.user_id) as users,
    AVG(r.metric_value) as avg_metric,
    STDDEV(r.metric_value) as stddev_metric
FROM ab_test_assignments a
LEFT JOIN ab_test_results r ON a.user_id = r.user_id AND a.test_name = r.test_name
GROUP BY a.test_name, a.variant
ORDER BY a.test_name, a.variant;

-- Per-test statistical comparison
WITH test_stats AS (
    SELECT 
        a.test_name,
        a.variant,
        COUNT(DISTINCT a.user_id) as n,
        AVG(r.metric_value) as mean,
        VARIANCE(r.metric_value) as variance
    FROM ab_test_assignments a
    JOIN ab_test_results r ON a.user_id = r.user_id AND a.test_name = r.test_name
    GROUP BY a.test_name, a.variant
)
SELECT 
    c.test_name,
    c.mean as control_mean,
    t.mean as treatment_mean,
    t.mean - c.mean as absolute_lift,
    ROUND((t.mean - c.mean) / NULLIF(c.mean, 0) * 100, 2) as relative_lift_pct,
    c.n as control_n,
    t.n as treatment_n
FROM test_stats c
JOIN test_stats t ON c.test_name = t.test_name
WHERE c.variant = 'control' AND t.variant = 'treatment';

-- ============================================
-- CONTENT INSIGHTS
-- ============================================

-- Top performing tracks
SELECT 
    t.track_id,
    t.genre,
    COUNT(*) as play_count,
    ROUND(AVG(CASE WHEN s.skipped THEN 0 ELSE 1 END) * 100, 2) as listen_through_rate,
    ROUND(AVG(s.listen_duration_ms) / t.duration_ms * 100, 2) as avg_completion_pct
FROM sessions s
JOIN tracks t ON s.track_id = t.track_id
GROUP BY t.track_id, t.genre, t.duration_ms
HAVING COUNT(*) >= 10
ORDER BY listen_through_rate DESC
LIMIT 100;

-- Genre trends over time
SELECT 
    DATE_TRUNC('week', s.timestamp) as week,
    t.genre,
    COUNT(*) as plays,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('week', s.timestamp)), 2) as genre_share
FROM sessions s
JOIN tracks t ON s.track_id = t.track_id
GROUP BY week, t.genre
ORDER BY week, plays DESC;

-- ============================================
-- USER BEHAVIOR PATTERNS
-- ============================================

-- Listening patterns by user tier
SELECT 
    u.tier,
    COUNT(DISTINCT u.user_id) as users,
    COUNT(*) as total_sessions,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT u.user_id), 2) as sessions_per_user,
    ROUND(AVG(s.listen_duration_ms) / 60000, 2) as avg_listen_minutes,
    ROUND(AVG(CASE WHEN s.skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM users u
JOIN sessions s ON u.user_id = s.user_id
GROUP BY u.tier
ORDER BY sessions_per_user DESC;

-- Device usage patterns
SELECT 
    device,
    COUNT(*) as sessions,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_sessions,
    ROUND(AVG(listen_duration_ms) / 60000, 2) as avg_listen_minutes,
    ROUND(AVG(CASE WHEN skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM sessions
GROUP BY device
ORDER BY sessions DESC;

-- Listening context analysis
SELECT 
    context,
    COUNT(*) as sessions,
    ROUND(AVG(CASE WHEN skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate,
    ROUND(AVG(listen_duration_ms) / 60000, 2) as avg_duration_min
FROM sessions
GROUP BY context
ORDER BY sessions DESC;
