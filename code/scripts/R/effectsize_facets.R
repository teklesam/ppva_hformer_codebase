#!/usr/bin/env Rscript
# effectsize_facets.R -- Durga effect-size (Cohen's d, BCa bootstrap 95% CI) for the
# decision-relevant pairwise comparisons, FACETED across metrics (PSNR, LPIPS, FSIM).
# Each metric is oriented so that positive d = Model 1 is BETTER (LPIPS is lower-is-better,
# so its sign is flipped). Metrics without per-image values are skipped automatically.
#
# Deps: Durga, readr, dplyr, tidyr, ggplot2
# Usage: Rscript effectsize_facets.R --csv results/per_image_metrics.csv --out figures/effectsize_facets.pdf
suppressMessages({library(Durga); library(readr); library(dplyr); library(tidyr); library(ggplot2)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv <- getarg("--csv", "results/per_image_metrics.csv")
out <- getarg("--out", "effectsize_facets.pdf")
R   <- as.integer(getarg("--R", "1999"))

spec <- tribble(~m1, ~m2, ~theme, ~lab,
 "arm_a_l2","arm_i_l1","1 · Pixel norm","A vs I",
 "arm_b_nll","arm_d_nll_ssim_ffl","2 · Structural losses","B vs D",
 "arm_d_nll_ssim_ffl","arm_m_full_det","2 · Structural losses","D vs M",
 "arm_e_ppvae","arm_f_kl_cyc","3 · KL schedule","E vs F",
 "arm_f_kl_cyc","arm_h_kl_cyc_fb","3 · KL schedule","F vs H",
 "arm_e_ppvae","arm_p_best","4 · VAE cost","E vs P",
 "arm_a_l2","arm_o_prelu","5 · PReLU failure","A vs O",
 "arm_d_nll_ssim_ffl","arm_o_prelu","5 · PReLU failure","D vs O",
 "arm_a_l2","dncnn_baseline","6 · Baselines","A vs DnCNN",
 "arm_a_l2","swinir","6 · Baselines","A vs SwinIR",
 "arm_d_nll_ssim_ffl","arm_s_ft_d","7 · Two-stage FT","D vs S")

# lower-is-better metrics get their sign flipped so positive d = Model 1 better
FLIP <- c(lpips = TRUE)
df0 <- read_csv(csv, show_col_types = FALSE) |> filter(noise_level == "mid")
metrics <- intersect(c("psnr","ssim","lpips","fsim"), names(df0))
metrics <- metrics[sapply(metrics, function(m) all(!is.na(df0[[m]])))]
mlab <- c(psnr = "PSNR (dB)", ssim = "SSIM", lpips = "LPIPS", fsim = "FSIM")

est_metric <- function(metric) {
  d <- filter(df0, arm %in% unique(c(spec$m1, spec$m2))) |> as.data.frame()
  dd <- DurgaDiff(d, data.col = metric, group.col = "arm", effect.type = "cohens d",
                  contrasts = paste(spec$m1, "-", spec$m2), R = R)
  e <- do.call(rbind, lapply(dd$group.differences,
        function(g) data.frame(d = g$t0, lo = g$bca[1, 4], hi = g$bca[1, 5])))
  s <- if (isTRUE(FLIP[metric])) -1 else 1
  bind_cols(spec, e) |> mutate(d = s*d, lo0 = s*lo, hi0 = s*hi,
                               lo = pmin(lo0, hi0), hi = pmax(lo0, hi0),
                               metric = factor(mlab[metric], levels = unname(mlab[metrics])))
}
res <- bind_rows(lapply(metrics, est_metric)) |>
  mutate(mag = cut(abs(d), c(-1, .2, .5, .8, Inf),
                   labels = c("negligible","small","moderate","large")),
         lab = factor(lab, levels = rev(unique(spec$lab))))

p <- ggplot(res, aes(d, lab, color = mag)) +
  geom_vline(xintercept = 0, color = "grey40") +
  geom_vline(xintercept = c(-.8,-.2,.2,.8), linetype = "dashed", color = "grey80", linewidth = .3) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = .25, linewidth = .6) +
  geom_point(size = 2.1) +
  facet_grid(theme ~ metric, scales = "free_y", space = "free_y", switch = "y") +
  scale_color_manual(values = c(negligible="#8c8c8c", small="#55a868",
                                moderate="#dd8452", large="#c44e52"), name = "effect size",
                     drop = FALSE) +
  labs(x = "Cohen's d  (positive = Model 1 better; LPIPS sign-flipped)", y = NULL,
       title = "Pairwise effect sizes across metrics",
       caption = sprintf("BCa bootstrap 95%% CI, R = %d (Durga); CI excluding 0 = significant; dashed = |d| 0.2 / 0.8", R)) +
  theme_minimal(base_size = 9) +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 0, face = "bold", size = 7.5),
        strip.text.x = element_text(face = "bold"), panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"), legend.position = "top")

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 2.2 + 2.6*length(metrics), height = 7.2)
cat("saved", out, "with metrics:", paste(metrics, collapse=", "), "\n")
