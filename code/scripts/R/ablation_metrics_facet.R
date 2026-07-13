#!/usr/bin/env Rscript
# ablation_metrics_facet.R -- all arms at a glance across every metric, faceted by metric.
# mean +/- 95% CI per arm at mid noise; arms ordered by PSNR; coloured by family.
suppressMessages({library(readr);library(dplyr);library(tidyr);library(ggplot2)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
csv <- getarg("--csv","results/per_image_metrics.csv"); out <- getarg("--out","ablation_metrics_facet.pdf")

lab <- c(arm_a_l2="A",arm_i_l1="I",arm_j_l1_ssim_ffl="J",arm_q_charb="Q",arm_r_ft_j="R",arm_s_ft_d="S",
 arm_b_nll="B",arm_c_nll_ssim="C",arm_d_nll_ssim_ffl="D",arm_k_nll_l1="K",arm_l_nll_edge_ffl="L",
 arm_m_full_det="M",arm_n_perc="N",arm_e_ppvae="E",arm_f_kl_cyc="F",arm_g_kl_fb="G",arm_h_kl_cyc_fb="H",
 arm_p_best="P",arm_o_prelu="O",dncnn_baseline="DnCNN",ircnn="IRCNN",ffdnet="FFDNet",drunet="DRUNet",swinir="SwinIR")
famv <- c(arm_a_l2="Pixel-norm",arm_i_l1="Pixel-norm",arm_j_l1_ssim_ffl="Pixel-norm",arm_q_charb="Pixel-norm",
 arm_r_ft_j="Fine-tune",arm_s_ft_d="Fine-tune",arm_b_nll="NLL",arm_c_nll_ssim="NLL",arm_d_nll_ssim_ffl="NLL",
 arm_k_nll_l1="NLL",arm_l_nll_edge_ffl="NLL",arm_m_full_det="NLL",arm_n_perc="NLL",arm_e_ppvae="VAE",
 arm_f_kl_cyc="VAE",arm_g_kl_fb="VAE",arm_h_kl_cyc_fb="VAE",arm_p_best="VAE",arm_o_prelu="Failure",
 dncnn_baseline="Baseline",ircnn="Baseline",ffdnet="Baseline",drunet="Baseline",swinir="Baseline")
metrics <- c("psnr","ssim","fsim","lpips")
mlab <- c(psnr="PSNR (dB)", ssim="SSIM", fsim="FSIM", lpips="LPIPS (lower better)")

d <- read_csv(csv, show_col_types=FALSE) |> filter(noise_level=="mid") |>
  select(arm, all_of(metrics)) |> filter(if_all(all_of(metrics), ~ !is.na(.)))
long <- d |> pivot_longer(all_of(metrics), names_to="metric", values_to="v") |>
  group_by(arm, metric) |>
  summarise(m=mean(v), lo=m-1.96*sd(v)/sqrt(n()), hi=m+1.96*sd(v)/sqrt(n()), .groups="drop")
ord <- long |> filter(metric=="psnr") |> arrange(m) |> pull(arm)
long <- long |> mutate(Arm=factor(lab[arm], levels=lab[ord]),
                       Family=factor(famv[arm], levels=c("Pixel-norm","NLL","VAE","Fine-tune","Baseline","Failure")),
                       metric=factor(mlab[metric], levels=unname(mlab[metrics])))

# dot plot (not bars): each metric's x-axis auto-zooms to its data range, so the
# small SSIM/FSIM/LPIPS differences become visible (a truncated bar axis would mislead)
p <- ggplot(long, aes(m, Arm, colour=Family)) +
  geom_errorbarh(aes(xmin=lo, xmax=hi), height=0.32, linewidth=0.4) +
  geom_point(size=2.4) +
  facet_wrap(~metric, scales="free_x", nrow=1) +
  scale_colour_manual(values=c("Pixel-norm"="#4c72b0","NLL"="#dd8452","VAE"="#8172b3",
                               "Fine-tune"="#937860","Baseline"="#55a868","Failure"="#c44e52")) +
  scale_x_continuous(expand=expansion(mult=c(0.06,0.06))) +
  labs(x=NULL, y=NULL, title="All arms across every metric (mid noise)",
       caption="mean +/- 95% CI (n=624); arms ordered by PSNR; each metric axis is zoomed to its own range; higher is better, except LPIPS where lower is better") +
  theme_minimal(base_size=8.5) +
  theme(plot.title=element_text(face="bold"), panel.grid.major.y=element_line(linewidth=0.2, colour="grey92"),
        panel.grid.minor.x=element_blank(),
        panel.border=element_rect(colour="grey35", fill=NA, linewidth=0.5),
        panel.spacing=unit(0.9,"lines"),
        strip.text=element_text(face="bold", colour="grey15"),
        strip.background=element_rect(fill="grey90", colour="grey35", linewidth=0.5),
        legend.position="top", axis.text.y=element_text(size=7))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=12.5, height=6.4)
cat("saved", out, "| arms:", length(unique(long$arm)), "\n")
