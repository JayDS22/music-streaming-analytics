-- Music Streaming Analytics Database Schema (PostgreSQL)

CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    tier VARCHAR(20) CHECK (tier IN ('free', 'premium', 'family', 'student')),
    country VARCHAR(2),
    age_group VARCHAR(10),
    preferred_genre VARCHAR(50)
);

CREATE TABLE tracks (
    track_id VARCHAR(50) PRIMARY KEY,
    tempo DECIMAL(6,2),
    energy DECIMAL(4,3),
    danceability DECIMAL(4,3),
    valence DECIMAL(4,3),
    acousticness DECIMAL(4,3),
    instrumentalness DECIMAL(4,3),
    loudness DECIMAL(5,2),
    duration_ms INTEGER,
    genre VARCHAR(50),
    artist_id VARCHAR(50),
    popularity DECIMAL(5,2)
);

CREATE TABLE sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    track_id VARCHAR(50) REFERENCES tracks(track_id),
    timestamp TIMESTAMP NOT NULL,
    listen_duration_ms INTEGER NOT NULL,
    track_duration_ms INTEGER,
    skipped BOOLEAN DEFAULT FALSE,
    context VARCHAR(20),
    device VARCHAR(20)
);

CREATE TABLE playlists (
    playlist_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    name VARCHAR(255),
    created_date TIMESTAMP,
    num_tracks INTEGER
);

CREATE TABLE playlist_tracks (
    playlist_id VARCHAR(50) REFERENCES playlists(playlist_id),
    track_id VARCHAR(50) REFERENCES tracks(track_id),
    position INTEGER,
    PRIMARY KEY (playlist_id, track_id)
);

CREATE TABLE ab_test_assignments (
    user_id VARCHAR(50) REFERENCES users(user_id),
    test_name VARCHAR(100),
    variant VARCHAR(50),
    assignment_date TIMESTAMP,
    PRIMARY KEY (user_id, test_name)
);

-- Indexes
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_timestamp ON sessions(timestamp);
CREATE INDEX idx_sessions_track ON sessions(track_id);
CREATE INDEX idx_users_signup ON users(signup_date);

-- View: Daily Active Users
CREATE VIEW daily_active_users AS
SELECT DATE(timestamp) as date, COUNT(DISTINCT user_id) as dau
FROM sessions GROUP BY DATE(timestamp);

-- View: Monthly Active Users
CREATE VIEW monthly_active_users AS
SELECT DATE_TRUNC('month', timestamp) as month, COUNT(DISTINCT user_id) as mau
FROM sessions GROUP BY DATE_TRUNC('month', timestamp);
