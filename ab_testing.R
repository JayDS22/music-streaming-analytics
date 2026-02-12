# Music Streaming Analytics - A/B Testing in R
# =============================================

library(dplyr)
library(ggplot2)
theme_set(theme_minimal())

# Sample size calculation
calculate_sample_size <- function(baseline_rate, mde, power = 0.80, alpha = 0.05) {
  treatment_rate <- baseline_rate * (1 + mde)
  pooled_rate <- (baseline_rate + treatment_rate) / 2
  z_alpha <- qnorm(1 - alpha/2)
  z_beta <- qnorm(power)
  effect <- abs(treatment_rate - baseline_rate)
  variance <- 2 * pooled_rate * (1 - pooled_rate)
  ceiling(variance * (z_alpha + z_beta)^2 / effect^2)
}

# A/B Test Analysis
analyze_ab_test <- function(control_data, treatment_data, metric = "value", alpha = 0.05) {
  control <- control_data[[metric]]
  treatment <- treatment_data[[metric]]
  
  control_mean <- mean(control, na.rm = TRUE)
  treatment_mean <- mean(treatment, na.rm = TRUE)
  control_sd <- sd(control, na.rm = TRUE)
  treatment_sd <- sd(treatment, na.rm = TRUE)
  control_n <- length(na.omit(control))
  treatment_n <- length(na.omit(treatment))
  
  absolute_effect <- treatment_mean - control_mean
  relative_effect <- absolute_effect / control_mean
  
  t_test <- t.test(treatment, control, var.equal = FALSE)
  p_value <- t_test$p.value
  
  pooled_se <- sqrt((control_sd^2 / control_n) + (treatment_sd^2 / treatment_n))
  z <- qnorm(1 - alpha/2)
  ci_lower <- absolute_effect - z * pooled_se
  ci_upper <- absolute_effect + z * pooled_se
  
  pooled_sd <- sqrt(((control_n - 1) * control_sd^2 + (treatment_n - 1) * treatment_sd^2) / 
                      (control_n + treatment_n - 2))
  cohens_d <- absolute_effect / pooled_sd
  
  list(
    control_mean = control_mean, treatment_mean = treatment_mean,
    control_sd = control_sd, treatment_sd = treatment_sd,
    control_n = control_n, treatment_n = treatment_n,
    absolute_effect = absolute_effect, relative_effect = relative_effect,
    p_value = p_value, ci_lower = ci_lower, ci_upper = ci_upper,
    is_significant = p_value < alpha, cohens_d = cohens_d,
    t_statistic = t_test$statistic
  )
}

# Generate report
generate_report <- function(results, test_name = "A/B Test") {
  cat("\n", strrep("=", 60), "\n")
  cat(sprintf("A/B TEST: %s\n", test_name))
  cat(strrep("=", 60), "\n")
  cat(sprintf("Control: n=%d, mean=%.4f\n", results$control_n, results$control_mean))
  cat(sprintf("Treatment: n=%d, mean=%.4f\n", results$treatment_n, results$treatment_mean))
  cat(sprintf("Effect: %+.4f (%+.2f%%)\n", results$absolute_effect, results$relative_effect * 100))
  cat(sprintf("95%% CI: [%.4f, %.4f]\n", results$ci_lower, results$ci_upper))
  cat(sprintf("P-value: %.4f\n", results$p_value))
  cat(sprintf("Significant: %s\n", ifelse(results$is_significant, "YES", "NO")))
  cat(sprintf("Cohen's d: %.3f\n", results$cohens_d))
  cat(strrep("=", 60), "\n\n")
}

# Plot results
plot_ab_results <- function(results, save_path = NULL) {
  data <- data.frame(
    group = c("Control", "Treatment"),
    mean = c(results$control_mean, results$treatment_mean),
    se = c(results$control_sd, results$treatment_sd) / sqrt(c(results$control_n, results$treatment_n))
  )
  
  p <- ggplot(data, aes(x = group, y = mean, fill = group)) +
    geom_col(width = 0.6) +
    geom_errorbar(aes(ymin = mean - 1.96*se, ymax = mean + 1.96*se), width = 0.2) +
    scale_fill_manual(values = c("Control" = "#6baed6", "Treatment" = "#08519c")) +
    labs(title = "A/B Test Results",
         subtitle = sprintf("p=%.4f | %s", results$p_value, 
                           ifelse(results$is_significant, "Significant", "Not Significant")),
         x = "", y = "Metric") +
    theme(legend.position = "none")
  
  if (!is.null(save_path)) ggsave(save_path, p, width = 8, height = 6, dpi = 150)
  p
}

# Simulation
run_ab_simulation <- function(n_users = 10000, baseline = 0.30, effect = 0.05, seed = 42) {
  set.seed(seed)
  
  control_n <- n_users %/% 2
  treatment_n <- n_users - control_n
  
  control_data <- data.frame(
    user_id = paste0("u", 1:control_n),
    value = rbinom(control_n, 1, baseline)
  )
  
  treatment_data <- data.frame(
    user_id = paste0("u", (control_n+1):n_users),
    value = rbinom(treatment_n, 1, baseline * (1 + effect))
  )
  
  results <- analyze_ab_test(control_data, treatment_data)
  generate_report(results, "Personalized Recommendations")
  
  results
}

# Main
run_ab_analysis <- function(output_dir = "output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  message("Running A/B test simulation...")
  results <- run_ab_simulation(n_users = 10000, baseline = 0.30, effect = 0.05)
  
  plot_ab_results(results, file.path(output_dir, "ab_test_results.png"))
  
  # Save results
  results_df <- data.frame(
    metric = names(unlist(results)),
    value = unlist(results)
  )
  write.csv(results_df, file.path(output_dir, "ab_test_results.csv"), row.names = FALSE)
  
  message("Analysis complete!")
  results
}

# Run if executed directly
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  output_dir <- ifelse(length(args) >= 1, args[1], "output")
  run_ab_analysis(output_dir)
}
