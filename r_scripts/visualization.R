# Visualization in R
library(dplyr)
library(ggplot2)

# Plot DAU/MAU
plot_dau_mau <- function(sessions, save_path = NULL) {
  sessions$date <- as.Date(sessions$timestamp)
  dau <- sessions %>% group_by(date) %>% summarise(dau = n_distinct(user_id))
  
  p <- ggplot(dau, aes(x = date, y = dau)) +
    geom_line(color = "steelblue") +
    labs(title = "Daily Active Users", x = "Date", y = "DAU") +
    theme_minimal()
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 12, height = 6)
  p
}

# Plot skip rates
plot_skip_rates <- function(sessions, tracks, save_path = NULL) {
  merged <- merge(sessions, tracks[, c("track_id", "genre")], by = "track_id")
  
  by_genre <- merged %>%
    group_by(genre) %>%
    summarise(skip_rate = mean(skipped) * 100)
  
  p <- ggplot(by_genre, aes(x = reorder(genre, -skip_rate), y = skip_rate)) +
    geom_col(fill = "steelblue") +
    labs(title = "Skip Rate by Genre", x = "Genre", y = "Skip Rate (%)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 10, height = 6)
  p
}

# Main
generate_visualizations <- function(data_dir = "data/raw", output_dir = "output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  sessions <- read.csv(file.path(data_dir, "sessions.csv"))
  tracks <- read.csv(file.path(data_dir, "tracks.csv"))
  
  plot_dau_mau(sessions, file.path(output_dir, "dau_mau.png"))
  plot_skip_rates(sessions, tracks, file.path(output_dir, "skip_rates.png"))
  
  message("Visualizations complete!")
}

if (!interactive()) generate_visualizations()
