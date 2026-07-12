#!/usr/bin/env Rscript
# compare_arms_ggstats.R -- annotated between-arm comparison (ggstatsplot::ggbetweenstats).
# Violin + box + jittered points for each arm, with a Welch ANOVA, Games-Howell
# pairwise comparisons (Welch, so unequal variances are handled) and Hedges' g
# effect sizes -- the same effect-size-first framing as the thesis Methods.
#
# Deps: readr, dplyr, ggstatsplot, ggplot2
# Usage:
#   Rscript compare_arms_ggstats.R --arms arm_a_l2,arm_i_l1,arm_d_nll_ssim_ffl,arm_h_kl_cyc_fb \
#       --metric psnr --noise mid --out figures/compare_reconstruction.pdf

suppressMessages({library(readr); library(dplyr); library(ggstatsplot); library(ggplot2)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- match(flag, args); if (is.na(i)) default else args[i + 1]
}
csv    <- getarg("--csv",    "results/per_image_metrics.csv")
metric <- getarg("--metric", "psnr")
noise  <- getarg("--noise",  "mid")
arms   <- strsplit(getarg("--arms", "arm_a_l2,arm_i_l1,arm_d_nll_ssim_ffl,arm_h_kl_cyc_fb"), ",")[[1]]
out    <- getarg("--out",    "compare_arms.pdf")
title  <- getarg("--title",  "Reconstruction across representative arms")

labmap <- c(arm_a_l2 = "A\n(L2)", arm_i_l1 = "I\n(L1)", arm_b_nll = "B\n(NLL)",
            arm_c_nll_ssim = "C\n(NLL+SSIM)", arm_d_nll_ssim_ffl = "D\n(NLL+SSIM+FFL)",
            arm_h_kl_cyc_fb = "H\n(PP-VAE)", arm_f_kl_cyc = "F\n(cyclic KL)",
            arm_e_ppvae = "E\n(linear KL)", arm_o_prelu = "O\n(PReLU)")
lab <- function(a) ifelse(a %in% names(labmap), labmap[a], a)

df <- read_csv(csv, show_col_types = FALSE) |>
  filter(noise_level == noise, arm %in% arms) |>
  mutate(Arm = factor(lab(arm), levels = lab(arms)), y = .data[[metric]])
stopifnot(nrow(df) > 0)

p <- ggbetweenstats(
  data = df, x = Arm, y = y, type = "parametric",
  pairwise.display = "all", p.adjust.method = "bonferroni", effsize.type = "unbiased",
  xlab = NULL, ylab = paste0(toupper(metric), "  (", noise, " noise)"),
  ggtheme = theme_ggstatsplot()
) + labs(title = title)

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 2.2 * length(arms) + 2, height = 6.8)
cat("saved", out, "\n")
