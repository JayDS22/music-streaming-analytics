-- Music Streaming Analytics Database Schema
-- PostgreSQL 13+

-- Drop existing tables if they exist
DROP TABLE IF EXISTS playlist_tracks CASCADE;
DROP TABLE IF EXISTS playlists CASCADE;
DROP TABLE IF EXISTS ab_test_results CASCADE;
DROP TABLE IF EXISTS ab_test_assignments CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS tracks CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Users table
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    tier VARCHAR(20) NOT NULL CHECK (tier IN ('free', 'premium', 'family', 'student')),
    country VARCHAR(2) NOT NULL,
    age_group VARCHAR(10),
    gender VARCHAR(10),
    preferred_genre VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tracks table with audio features
CREATE TABLE tracks (
    track_id VARCHAR(50) PRIMARY KEY,
    tempo DECIMAL(6,2),
    energy DECIMAL(4,3),
    danceability DECIMAL(4,3),
    valence DECIMAL(4,3),
    acousticness DECIMAL(4,3),
    instrumentalness DECIMAL(4,3),
    liveness DECIMAL(4,3),
    speechiness DECIMAL(4,3),
    loudness DECIMAL(5,2),
    duration_ms INTEGER,
    genre VARCHAR(50),
    artist_id VARCHAR(50),
    release_year INTEGER,
    popularity DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table (listening events)
CREATE TABLE sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    track_id VARCHAR(50) NOT NULL REFERENCES tracks(track_id),
    timestamp TIMESTAMP NOT NULL,
    listen_duration_ms INTEGER NOT NULL,
    track_duration_ms INTEGER,
    skipped BOOLEAN DEFAULT FALSE,
    skip_time_ms INTEGER,
    context VARCHAR(20) CHECK (context IN ('playlist', 'album', 'radio', 'search', 'recommendation')),
    device VARCHAR(20) CHECK (device IN ('mobile', 'desktop', 'tablet', 'smart_speaker')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlists table
CREATE TABLE playlists (
    playlist_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    name VARCHAR(255),
    created_date TIMESTAMP,
    num_tracks INTEGER,
    is_public BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlist tracks (many-to-many)
CREATE TABLE playlist_tracks (
    playlist_id VARCHAR(50) NOT NULL REFERENCES playlists(playlist_id),
    track_id VARCHAR(50) NOT NULL REFERENCES tracks(track_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (playlist_id, track_id)
);

-- A/B Test assignments
CREATE TABLE ab_test_assignments (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    test_name VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL CHECK (variant IN ('control', 'treatment')),
    assignment_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, test_name)
);

-- A/B Test results
CREATE TABLE ab_test_results (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    test_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User metrics (aggregated)
CREATE TABLE user_metrics (
    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
    total_sessions INTEGER DEFAULT 0,
    total_listen_time_ms BIGINT DEFAULT 0,
    skip_rate DECIMAL(4,3),
    avg_session_duration_ms INTEGER,
    unique_tracks_played INTEGER,
    unique_genres INTEGER,
    last_active_date DATE,
    engagement_score DECIMAL(4,3),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_timestamp ON sessions(timestamp);
CREATE INDEX idx_sessions_track_id ON sessions(track_id);
CREATE INDEX idx_sessions_context ON sessions(context);
CREATE INDEX idx_users_signup_date ON users(signup_date);
CREATE INDEX idx_users_tier ON users(tier);
CREATE INDEX idx_users_country ON users(country);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_ab_test_assignments_test ON ab_test_assignments(test_name);
CREATE INDEX idx_ab_test_results_test ON ab_test_results(test_name);

-- View: Daily Active Users
CREATE OR REPLACE VIEW daily_active_users AS
SELECT 
    DATE(timestamp) as date,
    COUNT(DISTINCT user_id) as dau,
    COUNT(*) as total_sessions,
    AVG(listen_duration_ms) as avg_listen_duration,
    AVG(CASE WHEN skipped THEN 1 ELSE 0 END) as skip_rate
FROM sessions
GROUP BY DATE(timestamp)
ORDER BY date;

-- View: Monthly Active Users
CREATE OR REPLACE VIEW monthly_active_users AS
SELECT 
    DATE_TRUNC('month', timestamp) as month,
    COUNT(DISTINCT user_id) as mau,
    COUNT(*) as total_sessions
FROM sessions
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month;

-- View: User Cohort Summary
CREATE OR REPLACE VIEW user_cohort_summary AS
SELECT 
    DATE_TRUNC('month', signup_date) as cohort_month,
    tier,
    COUNT(*) as users,
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - signup_date))/86400) as avg_account_age_days
FROM users
GROUP BY DATE_TRUNC('month', signup_date), tier
ORDER BY cohort_month;

-- Function to update user metrics
CREATE OR REPLACE FUNCTION update_user_metrics(p_user_id VARCHAR(50))
RETURNS VOID AS $$
BEGIN
    INSERT INTO user_metrics (user_id, total_sessions, total_listen_time_ms, skip_rate, 
                              avg_session_duration_ms, unique_tracks_played, last_active_date, updated_at)
    SELECT 
        user_id,
        COUNT(*) as total_sessions,
        SUM(listen_duration_ms) as total_listen_time_ms,
        AVG(CASE WHEN skipped THEN 1 ELSE 0 END) as skip_rate,
        AVG(listen_duration_ms) as avg_session_duration_ms,
        COUNT(DISTINCT track_id) as unique_tracks_played,
        MAX(DATE(timestamp)) as last_active_date,
        CURRENT_TIMESTAMP
    FROM sessions
    WHERE user_id = p_user_id
    GROUP BY user_id
    ON CONFLICT (user_id) DO UPDATE SET
        total_sessions = EXCLUDED.total_sessions,
        total_listen_time_ms = EXCLUDED.total_listen_time_ms,
        skip_rate = EXCLUDED.skip_rate,
        avg_session_duration_ms = EXCLUDED.avg_session_duration_ms,
        unique_tracks_played = EXCLUDED.unique_tracks_played,
        last_active_date = EXCLUDED.last_active_date,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
