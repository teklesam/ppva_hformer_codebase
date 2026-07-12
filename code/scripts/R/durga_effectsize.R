#!/usr/bin/env Rscript
# durga_effectsize.R -- estimation-statistics forest of the selected pairwise comparisons
# (replaces the Cohen's d bar chart AND the selected-pairwise table). Effect sizes and
# bootstrap 95% CIs are estimated with the Durga package (Khan & McLean); grouped by theme.
#
# Deps: Durga, readr, dplyr, ggplot2
# Usage: Rscript durga_effectsize.R --csv results/per_image_metrics.csv --out figures/durga_effectsize.pdf
suppressMessages({library(Durga); library(readr); library(dplyr); library(ggplot2)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv <- getarg("--csv", "results/per_image_metrics.csv")
out <- getarg("--out", "durga_effectsize.pdf")
R   <- as.integer(getarg("--R", "1999"))

spec <- tribble(~m1, ~m2, ~theme, ~lab,
 "arm_a_l2","arm_i_l1","1 · Pixel norm (L1 vs L2)","A vs I",
 "arm_a_l2","arm_j_l1_ssim_ffl","1 · Pixel norm (L1 vs L2)","A vs J",
 "arm_b_nll","arm_d_nll_ssim_ffl","2 · Structural losses","B vs D",
 "arm_b_nll","arm_l_nll_edge_ffl","2 · Structural losses","B vs L",
 "arm_d_nll_ssim_ffl","arm_m_full_det","2 · Structural losses","D vs M",
 "arm_e_ppvae","arm_f_kl_cyc","3 · KL schedule","E vs F",
 "arm_f_kl_cyc","arm_g_kl_fb","3 · KL schedule","F vs G",
 "arm_f_kl_cyc","arm_h_kl_cyc_fb","3 · KL schedule","F vs H",
 "arm_g_kl_fb","arm_h_kl_cyc_fb","3 · KL schedule","G vs H",
 "arm_e_ppvae","arm_m_full_det","4 · VAE cost","E vs M",
 "arm_e_ppvae","arm_p_best","4 · VAE cost","E vs P",
 "arm_b_nll","arm_k_nll_l1","4 · VAE cost","B vs K",
 "arm_a_l2","arm_o_prelu","5 · PReLU failure","A vs O",
 "arm_d_nll_ssim_ffl","arm_o_prelu","5 · PReLU failure","D vs O",
 "ircnn","ffdnet","6 · Baselines (KAIR)","IRCNN vs FFDNet",
 "ircnn","drunet","6 · Baselines (KAIR)","IRCNN vs DRUNet",
 "ircnn","swinir","6 · Baselines (KAIR)","IRCNN vs SwinIR",
 "arm_a_l2","ircnn","6 · Baselines (KAIR)","A vs IRCNN",
 "arm_e_ppvae","swinir","6 · Baselines (KAIR)","E vs SwinIR")

df <- read_csv(csv, show_col_types = FALSE) |>
  filter(noise_level == "mid", arm %in% unique(c(spec$m1, spec$m2))) |> as.data.frame()

dd <- DurgaDiff(df, data.col = "psnr", group.col = "arm", effect.type = "cohens d",
                contrasts = paste(spec$m1, "-", spec$m2), R = R)
est <- do.call(rbind, lapply(dd$group.differences,
        function(g) data.frame(d = g$t0, lo = g$bca[1, 4], hi = g$bca[1, 5])))
res <- bind_cols(spec, est) |>
  mutate(mag = cut(abs(d), c(-1, .2, .5, .8, Inf),
                   labels = c("negligible", "small", "moderate", "large")),
         lab = factor(lab, levels = rev(lab)))

p <- ggplot(res, aes(d, lab, color = mag)) +
  geom_vline(xintercept = 0, color = "grey40") +
  geom_vline(xintercept = c(-.8, -.2, .2, .8), linetype = "dashed", color = "grey75") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = .25, linewidth = .6) +
  geom_point(size = 2.4) +
  facet_grid(theme ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_color_manual(values = c(negligible = "grey55", small = "#2ca25f",
                                moderate = "#dd8452", large = "#c44e52"), name = "effect size") +
  labs(x = "Cohen's d (Model 1 - Model 2), bootstrap 95% CI (Durga)", y = NULL,
       title = "Selected pairwise effect sizes with bootstrap confidence intervals",
       caption = "dashed lines = |d| 0.2 (negligible) and 0.8 (large) thresholds; R = 1999 BCa bootstrap") +
  theme_minimal(base_size = 10) +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 0, face = "bold", size = 8),
        plot.title = element_text(face = "bold"), panel.grid.major.y = element_blank(),
        legend.position = "top")

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 9, height = 7)
cat("saved", out, "\n")
