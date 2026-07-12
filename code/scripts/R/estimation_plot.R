#!/usr/bin/env Rscript
# estimation_plot.R -- Gardner-Altman / Cumming estimation plots (dabestr) for the
# ablation arms. Shows the full per-image distribution alongside the mean difference
# with a bootstrap 95% CI, complementing the effect-size-first analysis of the thesis
# (Methods, Statistical Analysis). First arm listed is the shared control.
#
# Deps: readr, dplyr, dabestr, ggplot2
# Usage:
#   Rscript estimation_plot.R --csv results/per_image_metrics.csv --metric psnr \
#       --noise mid --arms arm_a_l2,arm_d_nll_ssim_ffl --out figures/estimation_cost.pdf
#   Rscript estimation_plot.R --arms arm_a_l2,arm_i_l1,arm_d_nll_ssim_ffl,arm_h_kl_cyc_fb \
#       --out figures/estimation_reconstruction.pdf

suppressMessages({library(readr); library(dplyr); library(dabestr); library(ggplot2)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- match(flag, args); if (is.na(i)) default else args[i + 1]
}
csv    <- getarg("--csv",    "results/per_image_metrics.csv")
metric <- getarg("--metric", "psnr")
noise  <- getarg("--noise",  "mid")
arms   <- strsplit(getarg("--arms", "arm_a_l2,arm_d_nll_ssim_ffl"), ",")[[1]]
out    <- getarg("--out",    "estimation_plot.pdf")

# short, readable arm labels
labmap <- c(arm_a_l2 = "A (L2)", arm_i_l1 = "I (L1)", arm_b_nll = "B (NLL)",
            arm_c_nll_ssim = "C (NLL+SSIM)", arm_d_nll_ssim_ffl = "D (NLL+SSIM+FFL)",
            arm_h_kl_cyc_fb = "H (PP-VAE)", arm_o_prelu = "O (PReLU)",
            arm_f_kl_cyc = "F (cyclic KL)", arm_e_ppvae = "E (linear KL)")
lab <- function(a) ifelse(a %in% names(labmap), labmap[a], a)

df <- read_csv(csv, show_col_types = FALSE) |>
  filter(noise_level == noise, arm %in% arms) |>
  mutate(y = .data[[metric]], Arm = factor(lab(arm), levels = lab(arms)))
stopifnot(nrow(df) > 0)

db <- dabestr::load(df, x = Arm, y = y, idx = list(lab(arms)))
md <- mean_diff(db)
cat("Mean differences vs control (", lab(arms)[1], "), ", metric, ", ", noise, " noise:\n", sep = "")
print(as.data.frame(md$boot_result[, c("test_group", "difference", "bca_ci_low", "bca_ci_high")]))

ylab <- toupper(metric)
p <- dabest_plot(md, raw_marker_size = 0.35, raw_marker_alpha = 0.15,
                 swarm_label = ylab, contrast_label = paste("mean difference in", ylab))

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 2.8 + 1.7 * length(arms), height = 5.2)
cat("saved", out, "\n")
