# Music Streaming Analytics - Cohort Analysis in R
# =================================================

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)

# Set theme
theme_set(theme_minimal())

# ----------------------------------
# DATA LOADING
# ----------------------------------

load_data <- function(data_dir = "data/raw") {
  users <- read.csv(file.path(data_dir, "users.csv"), stringsAsFactors = FALSE)
  sessions <- read.csv(file.path(data_dir, "sessions.csv"), stringsAsFactors = FALSE)
  
  # Parse dates
  users$signup_date <- as.Date(users$signup_date)
  sessions$timestamp <- as.POSIXct(sessions$timestamp)
  
  list(users = users, sessions = sessions)
}

# ----------------------------------
# COHORT ANALYSIS FUNCTIONS
# ----------------------------------

create_cohort <- function(users, period = "month") {
  users %>%
    mutate(cohort = floor_date(signup_date, period))
}

calculate_retention <- function(users, sessions, periods = 12) {
  # Add cohort to users
  users <- users %>%
    mutate(cohort = floor_date(signup_date, "month"))
  
  # Add activity period to sessions
  sessions <- sessions %>%
    mutate(activity_period = floor_date(as.Date(timestamp), "month"))
  
  # Merge to get cohort for each session
  session_cohorts <- sessions %>%
    inner_join(users %>% select(user_id, cohort), by = "user_id")
  
  # Calculate period number
  session_cohorts <- session_cohorts %>%
    mutate(period_number = interval(cohort, activity_period) %/% months(1))
  
  # Get unique users per cohort-period
  cohort_activity <- session_cohorts %>%
    filter(period_number >= 0, period_number < periods) %>%
    group_by(cohort, period_number) %>%
    summarise(active_users = n_distinct(user_id), .groups = "drop")
  
  # Get cohort sizes
  cohort_sizes <- users %>%
    group_by(cohort) %>%
    summarise(cohort_size = n_distinct(user_id), .groups = "drop")
  
  # Calculate retention rates
  retention <- cohort_activity %>%
    left_join(cohort_sizes, by = "cohort") %>%
    mutate(retention_rate = active_users / cohort_size * 100)
  
  retention
}

# Create retention matrix for visualization
create_retention_matrix <- function(retention_data) {
  retention_data %>%
    select(cohort, period_number, retention_rate) %>%
    pivot_wider(names_from = period_number, values_from = retention_rate)
}

# ----------------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------------

plot_retention_heatmap <- function(retention_data, save_path = NULL) {
  p <- ggplot(retention_data, aes(x = factor(period_number), y = cohort, fill = retention_rate)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.1f%%", retention_rate)), size = 3) +
    scale_fill_gradient2(
      low = "#f7fbff", mid = "#6baed6", high = "#08306b",
      midpoint = 50, limits = c(0, 100),
      name = "Retention %"
    ) +
    labs(
      title = "Monthly Cohort Retention Analysis",
      subtitle = "Percentage of users active in each period after signup",
      x = "Months Since Signup",
      y = "Signup Cohort"
    ) +
    theme(
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 8),
      plot.title = element_text(size = 14, face = "bold")
    )
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 14, height = 8, dpi = 150)
    message(paste("Saved plot to", save_path))
  }
  
  p
}

plot_retention_curve <- function(retention_data, save_path = NULL) {
  avg_retention <- retention_data %>%
    group_by(period_number) %>%
    summarise(
      avg_retention = mean(retention_rate, na.rm = TRUE),
      se = sd(retention_rate, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(avg_retention, aes(x = period_number, y = avg_retention)) +
    geom_ribbon(aes(ymin = avg_retention - 1.96*se, ymax = avg_retention + 1.96*se), 
                alpha = 0.3, fill = "steelblue") +
    geom_line(color = "steelblue", size = 1.2) +
    geom_point(color = "steelblue", size = 3) +
    labs(
      title = "Average Retention Curve",
      subtitle = "With 95% confidence interval",
      x = "Months Since Signup",
      y = "Retention Rate (%)"
    ) +
    scale_y_continuous(limits = c(0, 100)) +
    theme(plot.title = element_text(size = 14, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6, dpi = 150)
  }
  
  p
}

# ----------------------------------
# COHORT ENGAGEMENT ANALYSIS
# ----------------------------------

calculate_cohort_engagement <- function(users, sessions) {
  users <- users %>%
    mutate(cohort = floor_date(signup_date, "month"))
  
  sessions_with_cohort <- sessions %>%
    inner_join(users %>% select(user_id, cohort), by = "user_id")
  
  engagement <- sessions_with_cohort %>%
    group_by(cohort) %>%
    summarise(
      unique_users = n_distinct(user_id),
      total_sessions = n(),
      total_listen_time_hrs = sum(listen_duration_ms) / (1000 * 60 * 60),
      avg_listen_duration_min = mean(listen_duration_ms) / (1000 * 60),
      skip_rate = mean(skipped) * 100,
      sessions_per_user = n() / n_distinct(user_id),
      .groups = "drop"
    )
  
  engagement
}

# ----------------------------------
# CHURN ANALYSIS
# ----------------------------------

identify_churned_users <- function(users, sessions, inactive_days = 30) {
  last_activity <- sessions %>%
    group_by(user_id) %>%
    summarise(last_active = max(as.Date(timestamp)), .groups = "drop")
  
  current_date <- max(as.Date(sessions$timestamp))
  
  churn_analysis <- users %>%
    left_join(last_activity, by = "user_id") %>%
    mutate(
      days_inactive = as.numeric(current_date - last_active),
      is_churned = is.na(days_inactive) | days_inactive >= inactive_days
    )
  
  churn_analysis
}

# ----------------------------------
# MAIN ANALYSIS
# ----------------------------------

run_cohort_analysis <- function(data_dir = "data/raw", output_dir = "output") {
  message("Loading data...")
  data <- load_data(data_dir)
  
  message("Calculating retention...")
  retention <- calculate_retention(data$users, data$sessions, periods = 12)
  
  message("Creating retention matrix...")
  retention_matrix <- create_retention_matrix(retention)
  
  message("Calculating cohort engagement...")
  engagement <- calculate_cohort_engagement(data$users, data$sessions)
  
  message("Identifying churned users...")
  churn <- identify_churned_users(data$users, data$sessions, inactive_days = 30)
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save results
  write.csv(retention, file.path(output_dir, "retention_data.csv"), row.names = FALSE)
  write.csv(retention_matrix, file.path(output_dir, "retention_matrix.csv"), row.names = FALSE)
  write.csv(engagement, file.path(output_dir, "cohort_engagement.csv"), row.names = FALSE)
  
  # Generate plots
  plot_retention_heatmap(retention, file.path(output_dir, "retention_heatmap.png"))
  plot_retention_curve(retention, file.path(output_dir, "retention_curve.png"))
  
  # Summary statistics
  summary_stats <- list(
    total_users = nrow(data$users),
    total_sessions = nrow(data$sessions),
    num_cohorts = n_distinct(retention$cohort),
    avg_period_1_retention = mean(retention$retention_rate[retention$period_number == 1], na.rm = TRUE),
    churn_rate = mean(churn$is_churned, na.rm = TRUE) * 100
  )
  
  message("\n--- COHORT ANALYSIS SUMMARY ---")
  message(sprintf("Total Users: %d", summary_stats$total_users))
  message(sprintf("Total Sessions: %d", summary_stats$total_sessions))
  message(sprintf("Number of Cohorts: %d", summary_stats$num_cohorts))
  message(sprintf("Avg Period 1 Retention: %.1f%%", summary_stats$avg_period_1_retention))
  message(sprintf("Overall Churn Rate: %.1f%%", summary_stats$churn_rate))
  
  list(
    retention = retention,
    retention_matrix = retention_matrix,
    engagement = engagement,
    churn = churn,
    summary = summary_stats
  )
}

# Run analysis if executed directly
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  data_dir <- ifelse(length(args) >= 1, args[1], "data/raw")
  output_dir <- ifelse(length(args) >= 2, args[2], "output")
  
  results <- run_cohort_analysis(data_dir, output_dir)
}
