# TensorShadow Analytics & Visualisation Script
# Purpose: Statistical analysis of model performance and data trends.

library(ggplot2)
library(dplyr)
library(jsonlite)

# Load data - in production, this targets /data directory
# For demonstration, we create a performance dataset
performance_data <- data.frame(
  epoch = 1:20,
  accuracy = c(seq(0.6, 0.95, length.out = 15), rep(0.96, 5)),
  loss = c(seq(0.8, 0.1, length.out = 15), rep(0.09, 5)),
  env = sample(c("Prod", "Staging"), 20, replace = TRUE)
)

print("Running TensorShadow Statistical Analysis...")

# 1. Generate Accuracy Trend Plot
accuracy_plot <- ggplot(performance_data, aes(x = epoch, y = accuracy, color = env)) +
  geom_line(size = 1.2) +
  geom_point() +
  theme_minimal() +
  labs(
    title = "TensorShadow: Training Accuracy Trend",
    subtitle = "Analysis of model convergence over epochs",
    x = "Epoch",
    y = "Accuracy",
    color = "Environment"
  ) +
  scale_color_brewer(palette = "Set1")

# Save plot to /docs or /data/reports
ggsave("docs/accuracy_trend.png", plot = accuracy_plot, width = 10, height = 6)

# 2. Calculate summary statistics
stats_summary <- performance_data %>%
  group_by(env) %>%
  summarize(
    mean_accuracy = mean(accuracy),
    max_accuracy = max(accuracy),
    min_loss = min(loss)
  )

# 3. Export to JSON for Go Backend consumption
write_json(stats_summary, "data/stats_summary.json", pretty = TRUE)

print("Analysis complete. Reports saved to /docs and /data.")


