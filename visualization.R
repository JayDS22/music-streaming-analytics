# Music Streaming Analytics - Visualizations in R
# ================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

theme_set(theme_minimal(base_size = 12))

# Color palette
COLORS <- c(
  primary = "#1DB954",    # Spotify green
  secondary = "#191414",  # Dark
  accent1 = "#1ed760",
  accent2 = "#535353",
  light = "#b3b3b3"
)

# ----------------------------------
# DAU/MAU VISUALIZATION
# ----------------------------------

plot_dau_mau <- function(sessions, save_path = NULL) {
  sessions$date <- as.Date(sessions$timestamp)
  sessions$month <- format(sessions$date, "%Y-%m")
  
  dau <- sessions %>%
    group_by(date) %>%
    summarise(dau = n_distinct(user_id), .groups = "drop")
  
  mau <- sessions %>%
    group_by(month) %>%
    summarise(mau = n_distinct(user_id), .groups = "drop")
  
  dau$month <- format(dau$date, "%Y-%m")
  metrics <- dau %>% left_join(mau, by = "month")
  metrics$stickiness <- metrics$dau / metrics$mau * 100
  
  p1 <- ggplot(metrics, aes(x = date, y = dau)) +
    geom_line(color = COLORS["primary"], size = 1) +
    geom_smooth(method = "loess", se = FALSE, color = COLORS["accent2"], linetype = "dashed") +
    labs(title = "Daily Active Users", x = "", y = "DAU") +
    scale_y_continuous(labels = comma) +
    theme(plot.title = element_text(face = "bold"))
  
  p2 <- ggplot(metrics, aes(x = date, y = stickiness)) +
    geom_line(color = COLORS["primary"], size = 1) +
    geom_hline(yintercept = mean(metrics$stickiness), linetype = "dashed", color = "red") +
    labs(title = "Stickiness (DAU/MAU %)", x = "", y = "Stickiness %") +
    theme(plot.title = element_text(face = "bold"))
  
  if (!is.null(save_path)) {
    library(gridExtra)
    combined <- grid.arrange(p1, p2, ncol = 1)
    ggsave(save_path, combined, width = 12, height = 8, dpi = 150)
  }
  
  list(dau_plot = p1, stickiness_plot = p2, data = metrics)
}

# ----------------------------------
# SKIP RATE ANALYSIS
# ----------------------------------

plot_skip_rates <- function(sessions, tracks, save_path = NULL) {
  merged <- sessions %>%
    left_join(tracks %>% select(track_id, genre, energy), by = "track_id")
  
  merged$hour <- as.numeric(format(as.POSIXct(merged$timestamp), "%H"))
  
  # By genre
  by_genre <- merged %>%
    group_by(genre) %>%
    summarise(skip_rate = mean(skipped) * 100, plays = n(), .groups = "drop") %>%
    arrange(desc(skip_rate))
  
  p1 <- ggplot(by_genre, aes(x = reorder(genre, -skip_rate), y = skip_rate, fill = skip_rate)) +
    geom_col() +
    scale_fill_gradient(low = COLORS["primary"], high = "red") +
    labs(title = "Skip Rate by Genre", x = "", y = "Skip Rate (%)") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
  
  # By hour
  by_hour <- merged %>%
    group_by(hour) %>%
    summarise(skip_rate = mean(skipped) * 100, .groups = "drop")
  
  p2 <- ggplot(by_hour, aes(x = hour, y = skip_rate)) +
    geom_line(color = COLORS["primary"], size = 1.2) +
    geom_point(color = COLORS["primary"], size = 3) +
    labs(title = "Skip Rate by Hour of Day", x = "Hour", y = "Skip Rate (%)") +
    scale_x_continuous(breaks = seq(0, 23, 3))
  
  # By energy level
  merged$energy_level <- cut(merged$energy, breaks = c(0, 0.33, 0.66, 1),
                             labels = c("Low", "Medium", "High"))
  
  by_energy <- merged %>%
    filter(!is.na(energy_level)) %>%
    group_by(energy_level) %>%
    summarise(skip_rate = mean(skipped) * 100, .groups = "drop")
  
  p3 <- ggplot(by_energy, aes(x = energy_level, y = skip_rate, fill = energy_level)) +
    geom_col() +
    scale_fill_manual(values = c("Low" = "#6baed6", "Medium" = "#3182bd", "High" = "#08519c")) +
    labs(title = "Skip Rate by Energy Level", x = "Track Energy", y = "Skip Rate (%)") +
    theme(legend.position = "none")
  
  if (!is.null(save_path)) {
    library(gridExtra)
    combined <- grid.arrange(p1, p2, p3, ncol = 2, nrow = 2)
    ggsave(save_path, combined, width = 14, height = 10, dpi = 150)
  }
  
  list(by_genre = p1, by_hour = p2, by_energy = p3)
}

# ----------------------------------
# USER ENGAGEMENT
# ----------------------------------

plot_engagement_segments <- function(sessions, save_path = NULL) {
  user_stats <- sessions %>%
    group_by(user_id) %>%
    summarise(
      sessions = n(),
      listen_hours = sum(listen_duration_ms) / 3600000,
      skip_rate = mean(skipped) * 100,
      .groups = "drop"
    ) %>%
    mutate(segment = case_when(
      sessions >= 100 ~ "Power User",
      sessions >= 30 ~ "Active",
      sessions >= 10 ~ "Casual",
      TRUE ~ "Light"
    ))
  
  segment_summary <- user_stats %>%
    group_by(segment) %>%
    summarise(
      users = n(),
      avg_sessions = mean(sessions),
      avg_hours = mean(listen_hours),
      .groups = "drop"
    )
  
  segment_summary$segment <- factor(segment_summary$segment, 
                                    levels = c("Light", "Casual", "Active", "Power User"))
  
  p <- ggplot(segment_summary, aes(x = segment, y = users, fill = segment)) +
    geom_col() +
    geom_text(aes(label = comma(users)), vjust = -0.5) +
    scale_fill_manual(values = c("Light" = "#b3b3b3", "Casual" = "#6baed6", 
                                 "Active" = "#3182bd", "Power User" = COLORS["primary"])) +
    labs(title = "User Engagement Segments", x = "", y = "Number of Users") +
    theme(legend.position = "none")
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 10, height = 6, dpi = 150)
  
  list(plot = p, summary = segment_summary)
}

# ----------------------------------
# LISTENING PATTERNS
# ----------------------------------

plot_listening_heatmap <- function(sessions, save_path = NULL) {
  sessions$hour <- as.numeric(format(as.POSIXct(sessions$timestamp), "%H"))
  sessions$weekday <- factor(weekdays(as.Date(sessions$timestamp)),
                             levels = c("Monday", "Tuesday", "Wednesday", 
                                       "Thursday", "Friday", "Saturday", "Sunday"))
  
  heatmap_data <- sessions %>%
    group_by(weekday, hour) %>%
    summarise(sessions = n(), .groups = "drop")
  
  p <- ggplot(heatmap_data, aes(x = hour, y = weekday, fill = sessions)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = COLORS["primary"], name = "Sessions") +
    labs(title = "Listening Patterns: Day x Hour", x = "Hour of Day", y = "") +
    scale_x_continuous(breaks = seq(0, 23, 3)) +
    theme(panel.grid = element_blank())
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 12, height = 6, dpi = 150)
  
  p
}

# ----------------------------------
# GENRE DISTRIBUTION
# ----------------------------------

plot_genre_distribution <- function(sessions, tracks, save_path = NULL) {
  merged <- sessions %>%
    left_join(tracks %>% select(track_id, genre), by = "track_id")
  
  genre_counts <- merged %>%
    count(genre, sort = TRUE) %>%
    mutate(pct = n / sum(n) * 100)
  
  p <- ggplot(genre_counts, aes(x = reorder(genre, pct), y = pct, fill = pct)) +
    geom_col() +
    coord_flip() +
    scale_fill_gradient(low = "#6baed6", high = COLORS["primary"]) +
    labs(title = "Genre Distribution", x = "", y = "Percentage of Plays (%)") +
    theme(legend.position = "none")
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 10, height = 6, dpi = 150)
  
  p
}

# ----------------------------------
# MAIN VISUALIZATION
# ----------------------------------

generate_all_visualizations <- function(data_dir = "data/raw", output_dir = "output/visualizations") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  message("Loading data...")
  sessions <- read.csv(file.path(data_dir, "sessions.csv"), stringsAsFactors = FALSE)
  tracks <- read.csv(file.path(data_dir, "tracks.csv"), stringsAsFactors = FALSE)
  
  message("Generating DAU/MAU plot...")
  plot_dau_mau(sessions, file.path(output_dir, "dau_mau.png"))
  
  message("Generating skip rate plots...")
  plot_skip_rates(sessions, tracks, file.path(output_dir, "skip_rates.png"))
  
  message("Generating engagement segments...")
  plot_engagement_segments(sessions, file.path(output_dir, "engagement_segments.png"))
  
  message("Generating listening heatmap...")
  plot_listening_heatmap(sessions, file.path(output_dir, "listening_heatmap.png"))
  
  message("Generating genre distribution...")
  plot_genre_distribution(sessions, tracks, file.path(output_dir, "genre_distribution.png"))
  
  message(sprintf("All visualizations saved to %s", output_dir))
}

# Run if executed directly
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  data_dir <- ifelse(length(args) >= 1, args[1], "data/raw")
  output_dir <- ifelse(length(args) >= 2, args[2], "output/visualizations")
  generate_all_visualizations(data_dir, output_dir)
}
