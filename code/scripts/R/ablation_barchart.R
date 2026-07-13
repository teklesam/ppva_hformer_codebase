#!/usr/bin/env Rscript
# ablation_barchart.R -- all arms at a glance: mean PSNR (mid noise) per arm, sorted,
# coloured by loss/architecture family, with 95% CI whiskers.
# Deps: readr, dplyr, ggplot2
# Usage: Rscript ablation_barchart.R --csv results/per_image_metrics.csv --out figures/ablation_bar.pdf
suppressMessages({library(readr); library(dplyr); library(ggplot2)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv    <- getarg("--csv",    "results/per_image_metrics.csv")
metric <- getarg("--metric", "psnr")
out    <- getarg("--out",    "ablation_bar.pdf")

lab <- c(arm_a_l2="A · L2",arm_i_l1="I · L1",arm_j_l1_ssim_ffl="J · L1+SSIM+FFL",arm_k_nll_l1="K · NLL+L1",
 arm_b_nll="B · NLL",arm_c_nll_ssim="C · NLL+SSIM",arm_d_nll_ssim_ffl="D · NLL+SSIM+FFL",
 arm_l_nll_edge_ffl="L · NLL+Edge+FFL",arm_n_perc="N · NLL+Perc",arm_m_full_det="M · NLL+SSIM+Edge+FFL",
 arm_e_ppvae="E · VAE linear",arm_f_kl_cyc="F · VAE cyclic",arm_g_kl_fb="G · VAE free-bits",
 arm_h_kl_cyc_fb="H · VAE cyc+fb",arm_p_best="P · VAE (PReLU)",arm_o_prelu="O · PReLU fail",
 ffdnet="FFDNet",ircnn="IRCNN",drunet="DRUNet",swinir="SwinIR",dncnn_baseline="DnCNN",
 arm_q_charb="Q · Charbonnier",arm_r_ft_j="R · FT L1+SSIM+FFL",arm_s_ft_d="S · FT NLL+SSIM+FFL")
famv <- c(arm_a_l2="Pixel-norm",arm_i_l1="Pixel-norm",arm_j_l1_ssim_ffl="Pixel-norm",arm_k_nll_l1="NLL",
 arm_b_nll="NLL",arm_c_nll_ssim="NLL",arm_d_nll_ssim_ffl="NLL",arm_l_nll_edge_ffl="NLL",
 arm_n_perc="NLL",arm_m_full_det="NLL",arm_e_ppvae="VAE",arm_f_kl_cyc="VAE",arm_g_kl_fb="VAE",
 arm_h_kl_cyc_fb="VAE",arm_p_best="VAE",arm_o_prelu="Failure",ffdnet="Baseline",ircnn="Baseline",
 drunet="Baseline",swinir="Baseline",dncnn_baseline="Baseline",arm_q_charb="Pixel-norm",
 arm_r_ft_j="Fine-tune",arm_s_ft_d="Fine-tune")

agg <- read_csv(csv, show_col_types = FALSE) |> filter(noise_level == "mid") |>
  rename(y = all_of(metric)) |>
  group_by(arm) |>
  summarise(m = mean(y), lo = m - 1.96*sd(y)/sqrt(n()), hi = m + 1.96*sd(y)/sqrt(n()), .groups = "drop") |>
  mutate(Arm = lab[arm], Family = factor(famv[arm], levels = c("Pixel-norm","NLL","VAE","Fine-tune","Baseline","Failure")))

p <- ggplot(agg, aes(reorder(Arm, m), m, fill = Family)) +
  geom_col(width = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.25, linewidth = 0.3) +
  coord_flip(ylim = c(floor(min(agg$lo)), ceiling(max(agg$hi)))) +
  scale_fill_manual(values = c("Pixel-norm"="#4c72b0","NLL"="#dd8452","VAE"="#8172b3",
                               "Fine-tune"="#937860","Baseline"="#55a868","Failure"="#c44e52")) +
  labs(x = NULL, y = paste0(toupper(metric), ", mid noise"), title = "All arms at a glance",
       caption = "bars = mean, whiskers = 95% CI (n = 624)") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), panel.grid.major.y = element_blank())

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 8, height = 6.2)
cat("saved", out, "\n")
