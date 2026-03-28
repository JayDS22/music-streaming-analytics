# A/B Testing in R
library(dplyr)
library(ggplot2)

# Analyze A/B test
analyze_ab_test <- function(control, treatment, alpha = 0.05) {
  t_result <- t.test(treatment, control, var.equal = FALSE)
  
  list(
    control_mean = mean(control), treatment_mean = mean(treatment),
    control_n = length(control), treatment_n = length(treatment),
    effect = mean(treatment) - mean(control),
    p_value = t_result$p.value,
    significant = t_result$p.value < alpha,
    ci = t_result$conf.int
  )
}

# Generate report
generate_report <- function(results) {
  cat("\n========== A/B TEST RESULTS ==========\n")
  cat(sprintf("Control: n=%d, mean=%.4f\n", results$control_n, results$control_mean))
  cat(sprintf("Treatment: n=%d, mean=%.4f\n", results$treatment_n, results$treatment_mean))
  cat(sprintf("Effect: %+.4f\n", results$effect))
  cat(sprintf("P-value: %.4f\n", results$p_value))
  cat(sprintf("Significant: %s\n", ifelse(results$significant, "YES", "NO")))
  cat("======================================\n\n")
}

# Run simulation
run_simulation <- function(n = 10000, baseline = 0.30, effect = 0.05) {
  set.seed(42)
  control <- rbinom(n/2, 1, baseline)
  treatment <- rbinom(n/2, 1, baseline * (1 + effect))
  
  results <- analyze_ab_test(control, treatment)
  generate_report(results)
  results
}

if (!interactive()) run_simulation()
