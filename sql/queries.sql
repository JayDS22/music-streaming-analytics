-- Music Streaming Analytics SQL Queries

-- DAU/MAU with Stickiness
WITH daily AS (
    SELECT DATE(timestamp) as date, DATE_TRUNC('month', timestamp) as month, user_id
    FROM sessions
),
dau AS (SELECT date, COUNT(DISTINCT user_id) as dau FROM daily GROUP BY date),
mau AS (SELECT month, COUNT(DISTINCT user_id) as mau FROM daily GROUP BY month)
SELECT d.date, d.dau, m.mau, ROUND(d.dau::DECIMAL / m.mau * 100, 2) as stickiness
FROM dau d JOIN mau m ON DATE_TRUNC('month', d.date) = m.month;

-- Skip Rate by Genre
SELECT t.genre, COUNT(*) as plays, 
       ROUND(AVG(CASE WHEN s.skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM sessions s JOIN tracks t ON s.track_id = t.track_id
GROUP BY t.genre ORDER BY skip_rate DESC;

-- Skip Rate by Hour
SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as sessions,
       ROUND(AVG(CASE WHEN skipped THEN 1 ELSE 0 END) * 100, 2) as skip_rate
FROM sessions GROUP BY hour ORDER BY hour;

-- Cohort Retention
WITH cohorts AS (
    SELECT user_id, DATE_TRUNC('month', signup_date) as cohort FROM users
),
activity AS (
    SELECT DISTINCT s.user_id, DATE_TRUNC('month', s.timestamp) as activity_month
    FROM sessions s
)
SELECT c.cohort, a.activity_month,
       COUNT(DISTINCT c.user_id) as users,
       ROUND(COUNT(DISTINCT c.user_id)::DECIMAL / 
             (SELECT COUNT(*) FROM cohorts WHERE cohort = c.cohort) * 100, 2) as retention
FROM cohorts c JOIN activity a ON c.user_id = a.user_id
WHERE a.activity_month >= c.cohort
GROUP BY c.cohort, a.activity_month ORDER BY c.cohort, a.activity_month;

-- User Engagement Segments
SELECT 
    CASE WHEN session_count >= 100 THEN 'Power' WHEN session_count >= 30 THEN 'Active'
         WHEN session_count >= 10 THEN 'Casual' ELSE 'Light' END as segment,
    COUNT(*) as users, ROUND(AVG(listen_hours), 2) as avg_hours
FROM (
    SELECT user_id, COUNT(*) as session_count, SUM(listen_duration_ms)/3600000.0 as listen_hours
    FROM sessions GROUP BY user_id
) stats GROUP BY segment;

-- A/B Test Results
SELECT a.test_name, a.variant, COUNT(DISTINCT a.user_id) as users,
       ROUND(AVG(CASE WHEN s.skipped THEN 0 ELSE 1 END), 4) as listen_through_rate
FROM ab_test_assignments a
JOIN sessions s ON a.user_id = s.user_id
GROUP BY a.test_name, a.variant;

-- Top Tracks by Listen-Through Rate
SELECT t.track_id, t.genre, COUNT(*) as plays,
       ROUND(AVG(CASE WHEN s.skipped THEN 0 ELSE 1 END) * 100, 2) as listen_through_rate
FROM sessions s JOIN tracks t ON s.track_id = t.track_id
GROUP BY t.track_id, t.genre HAVING COUNT(*) >= 10
ORDER BY listen_through_rate DESC LIMIT 100;
