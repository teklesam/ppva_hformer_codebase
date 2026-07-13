#!/usr/bin/env Rscript
# dose_curves.R -- PSNR vs noise (dose) severity for every arm; key arms highlighted.
# Deps: readr, dplyr, ggplot2
# Usage: Rscript dose_curves.R --csv results/per_image_metrics.csv --out figures/dose_curves.pdf
suppressMessages({library(readr); library(dplyr); library(ggplot2)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv <- getarg("--csv", "results/per_image_metrics.csv")
out <- getarg("--out", "dose_curves.pdf")

lab <- c(arm_a_l2 = "A (L2)", arm_d_nll_ssim_ffl = "D (NLL+SSIM+FFL)", arm_h_kl_cyc_fb = "H (PP-VAE)",
         arm_o_prelu = "O (PReLU)", swinir = "SwinIR")
key <- names(lab)

agg <- read_csv(csv, show_col_types = FALSE) |>
  mutate(noise = factor(noise_level, levels = c("low","mid","high"), labels = c("Low","Mid","High"))) |>
  group_by(arm, noise) |>
  summarise(m = mean(psnr), lo = m - 1.96 * sd(psnr)/sqrt(n()), hi = m + 1.96 * sd(psnr)/sqrt(n()), .groups = "drop") |>
  mutate(keyarm = arm %in% key, Arm = ifelse(keyarm, lab[arm], arm))

p <- ggplot(agg, aes(noise, m, group = arm)) +
  geom_line(data = filter(agg, !keyarm), color = "grey82", linewidth = 0.4) +
  geom_ribbon(data = filter(agg, keyarm), aes(ymin = lo, ymax = hi, fill = Arm), alpha = 0.15, color = NA) +
  geom_line(data = filter(agg, keyarm), aes(color = Arm), linewidth = 1.1) +
  geom_point(data = filter(agg, keyarm), aes(color = Arm), size = 2.2) +
  scale_color_brewer(palette = "Set1") + scale_fill_brewer(palette = "Set1") +
  labs(x = "noise severity preset", y = "PSNR (dB)", color = "key arm", fill = "key arm",
       title = "Reconstruction quality across dose (noise) levels",
       caption = "grey = the remaining arms; ribbons = 95% CI of the mean") +
  theme_minimal(base_size = 11) + theme(plot.title = element_text(face = "bold"))

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 8, height = 5.4)
cat("saved", out, "\n")
