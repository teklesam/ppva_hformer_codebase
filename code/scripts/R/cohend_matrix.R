#!/usr/bin/env Rscript
# cohend_matrix.R -- pairwise Cohen's d heatmap (ggplot2 + scico) from pairwise_stats.csv.
# Diverging vik palette (row - col); dots mark pairs that are not significant after
# Bonferroni correction (statistically interchangeable configurations).
#
# Deps: readr, dplyr, ggplot2, scico
# Usage: Rscript cohend_matrix.R --csv results/pairwise_stats.csv --out figures/cohend_matrix.pdf
suppressMessages({library(readr); library(dplyr); library(ggplot2); library(scico)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv <- getarg("--csv", "results/pairwise_stats.csv")
out <- getarg("--out", "cohend_matrix.pdf")

ord <- c("arm_a_l2","arm_i_l1","arm_j_l1_ssim_ffl","arm_k_nll_l1","arm_b_nll","arm_c_nll_ssim",
  "arm_d_nll_ssim_ffl","arm_l_nll_edge_ffl","arm_n_perc","arm_m_full_det","arm_e_ppvae",
  "arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu","dncnn_baseline","ffdnet","ircnn","drunet","swinir")
lab <- c(arm_a_l2="A",arm_i_l1="I",arm_j_l1_ssim_ffl="J",arm_k_nll_l1="K",arm_b_nll="B",arm_c_nll_ssim="C",
  arm_d_nll_ssim_ffl="D",arm_l_nll_edge_ffl="L",arm_n_perc="N",arm_m_full_det="M",arm_e_ppvae="E",
  arm_f_kl_cyc="F",arm_g_kl_fb="G",arm_h_kl_cyc_fb="H",arm_p_best="P",arm_o_prelu="O",
  dncnn_baseline="DnCNN",ffdnet="FFDNet",ircnn="IRCNN",drunet="DRUNet",swinir="SwinIR")
labs <- unname(lab[ord])

idx <- setNames(seq_along(ord), ord)
# keep each pair once as a lower triangle (row = later arm, col = earlier arm; d = row - col)
m <- read_csv(csv, show_col_types = FALSE) |>
     filter(arm1 %in% ord, arm2 %in% ord) |>
     mutate(hi = ifelse(idx[arm1] >= idx[arm2], arm1, arm2),
            lo = ifelse(idx[arm1] >= idx[arm2], arm2, arm1),
            d  = ifelse(idx[arm1] >= idx[arm2], cohens_d, -cohens_d),
            ns = sig == "n.s.",
            row = factor(lab[hi], levels = rev(labs)),
            col = factor(lab[lo], levels = labs),
            dcl = pmin(pmax(d, -3), 3))

p <- ggplot(m, aes(col, row, fill = dcl)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_point(data = filter(m, ns), color = "black", size = 0.7) +
  scale_fill_scico(palette = "vik", limits = c(-3, 3), name = "Cohen's d\n(row - col)") +
  coord_equal() +
  labs(x = NULL, y = NULL, title = "Pairwise effect sizes (PSNR, mid noise)",
       caption = "lower triangle only (matrix is symmetric); dots = not significant after Bonferroni") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1),
        panel.grid = element_blank(), plot.title = element_text(face = "bold", size = 10))

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 8, height = 7.2)
cat("saved", out, "\n")
