# Cohort Analysis in R
library(dplyr)
library(tidyr)
library(ggplot2)

# Load data
load_data <- function(data_dir = "data/raw") {
  users <- read.csv(file.path(data_dir, "users.csv"))
  sessions <- read.csv(file.path(data_dir, "sessions.csv"))
  users$signup_date <- as.Date(users$signup_date)
  sessions$timestamp <- as.POSIXct(sessions$timestamp)
  list(users = users, sessions = sessions)
}

# Calculate retention matrix
calculate_retention <- function(users, sessions, periods = 12) {
  users$cohort <- format(users$signup_date, "%Y-%m")
  sessions$activity_month <- format(sessions$timestamp, "%Y-%m")
  
  session_cohorts <- merge(sessions, users[, c("user_id", "cohort")], by = "user_id")
  
  cohort_activity <- session_cohorts %>%
    group_by(cohort, activity_month) %>%
    summarise(active_users = n_distinct(user_id), .groups = "drop")
  
  cohort_sizes <- users %>% group_by(cohort) %>% summarise(size = n_distinct(user_id))
  
  retention <- merge(cohort_activity, cohort_sizes, by = "cohort")
  retention$retention_rate <- retention$active_users / retention$size * 100
  retention
}

# Plot retention heatmap
plot_retention <- function(retention, save_path = NULL) {
  p <- ggplot(retention, aes(x = activity_month, y = cohort, fill = retention_rate)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Cohort Retention Heatmap", x = "Activity Month", y = "Cohort") +
    theme_minimal()
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 12, height = 8)
  p
}

# Main
run_analysis <- function(data_dir = "data/raw", output_dir = "output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  data <- load_data(data_dir)
  retention <- calculate_retention(data$users, data$sessions)
  
  write.csv(retention, file.path(output_dir, "retention_data.csv"), row.names = FALSE)
  plot_retention(retention, file.path(output_dir, "retention_heatmap.png"))
  
  message("Analysis complete!")
}

if (!interactive()) run_analysis()
