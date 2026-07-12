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
  "arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu","ffdnet","ircnn","drunet","swinir")
lab <- c(arm_a_l2="A",arm_i_l1="I",arm_j_l1_ssim_ffl="J",arm_k_nll_l1="K",arm_b_nll="B",arm_c_nll_ssim="C",
  arm_d_nll_ssim_ffl="D",arm_l_nll_edge_ffl="L",arm_n_perc="N",arm_m_full_det="M",arm_e_ppvae="E",
  arm_f_kl_cyc="F",arm_g_kl_fb="G",arm_h_kl_cyc_fb="H",arm_p_best="P",arm_o_prelu="O",
  ffdnet="FFDNet",ircnn="IRCNN",drunet="DRUNet",swinir="SwinIR")
labs <- unname(lab[ord])

df <- read_csv(csv, show_col_types = FALSE)
m <- bind_rows(
       transmute(df, r = arm1, c = arm2, d = cohens_d, ns = sig == "n.s."),
       transmute(df, r = arm2, c = arm1, d = -cohens_d, ns = sig == "n.s.")) |>
     filter(r %in% ord, c %in% ord) |>
     mutate(r = factor(lab[r], levels = labs), c = factor(lab[c], levels = rev(labs)),
            dcl = pmin(pmax(d, -3), 3))

p <- ggplot(m, aes(r, c, fill = dcl)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_point(data = filter(m, ns), color = "black", size = 0.7) +
  scale_fill_scico(palette = "vik", limits = c(-3, 3), name = "Cohen's d\n(row - col)") +
  coord_equal() +
  labs(x = NULL, y = NULL, title = "Pairwise effect sizes (PSNR, mid noise)",
       caption = "dots = not significant after Bonferroni (statistically interchangeable)") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1),
        panel.grid = element_blank(), plot.title = element_text(face = "bold", size = 10))

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 8, height = 7.2)
cat("saved", out, "\n")
