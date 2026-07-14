#!/usr/bin/env Rscript
# cohend_matrix_facets.R -- pairwise Cohen's d heatmaps (lower triangle) for ALL FOUR
# metrics in one faceted figure (merges the former Fig 4.6 PSNR matrix with the
# SSIM/FSIM/LPIPS panels). PSNR read from pairwise_stats.csv; the other three from
# pairwise_stats_metrics.csv (LPIPS sign-flipped upstream so positive d = row arm better).
suppressMessages({library(readr);library(dplyr);library(ggplot2);library(scico)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
psnr_csv <- getarg("--psnr","results/pairwise_stats.csv")
met_csv  <- getarg("--metrics","results/pairwise_stats_metrics.csv")
out      <- getarg("--out","cohend_matrix_facets.pdf")

ord <- c("arm_a_l2","arm_i_l1","arm_j_l1_ssim_ffl","arm_k_nll_l1","arm_b_nll","arm_c_nll_ssim",
 "arm_d_nll_ssim_ffl","arm_l_nll_edge_ffl","arm_n_perc","arm_m_full_det","arm_e_ppvae",
 "arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu","dncnn_baseline","ffdnet","ircnn","drunet","swinir")
lab <- c(arm_a_l2="A",arm_i_l1="I",arm_j_l1_ssim_ffl="J",arm_k_nll_l1="K",arm_b_nll="B",arm_c_nll_ssim="C",
 arm_d_nll_ssim_ffl="D",arm_l_nll_edge_ffl="L",arm_n_perc="N",arm_m_full_det="M",arm_e_ppvae="E",
 arm_f_kl_cyc="F",arm_g_kl_fb="G",arm_h_kl_cyc_fb="H",arm_p_best="P",arm_o_prelu="O",
 dncnn_baseline="DnCNN",ffdnet="FFDNet",ircnn="IRCNN",drunet="DRUNet",swinir="SwinIR")
labs <- unname(lab[ord]); idx <- setNames(seq_along(ord), ord)

met <- read_csv(met_csv, show_col_types=FALSE) |> select(metric, arm1, arm2, cohens_d, sig)
psn <- read_csv(psnr_csv, show_col_types=FALSE) |> mutate(metric="psnr") |> select(metric, arm1, arm2, cohens_d, sig)
d <- bind_rows(psn, met) |> filter(arm1 %in% ord, arm2 %in% ord) |>
  mutate(hi=ifelse(idx[arm1]>=idx[arm2],arm1,arm2), lo=ifelse(idx[arm1]>=idx[arm2],arm2,arm1),
         dd=ifelse(idx[arm1]>=idx[arm2],cohens_d,-cohens_d), ns=sig=="n.s.",
         row=factor(lab[hi],levels=rev(labs)), col=factor(lab[lo],levels=labs), dcl=pmin(pmax(dd,-3),3),
         metric=factor(recode(metric, psnr="PSNR", ssim="SSIM", fsim="FSIM",
                              lpips="LPIPS (lower is better; sign-flipped)"),
                       levels=c("PSNR","SSIM","FSIM","LPIPS (lower is better; sign-flipped)")))
p <- ggplot(d, aes(col,row,fill=dcl)) +
  geom_tile(color="white",linewidth=0.15) +
  geom_point(data=filter(d,ns),color="black",size=0.5) +
  scale_fill_scico(palette="vik",limits=c(-3,3),name="Cohen's d\n(row - col)") +
  facet_wrap(~metric, nrow=2) + coord_equal() +
  labs(x=NULL,y=NULL,title="Pairwise effect sizes across all four metrics (mid noise)",
       caption="Lower triangle; dots = not significant after Bonferroni correction (interchangeable). Red = row arm better, blue = worse. PSNR and SSIM are trained metrics; FSIM and LPIPS are held out.") +
  theme_minimal(base_size=8) +
  theme(axis.text.x=element_text(angle=90,vjust=.5,hjust=1,size=6), axis.text.y=element_text(size=6),
        panel.grid=element_blank(), plot.title=element_text(face="bold",size=12),
        plot.caption=element_text(size=7.5, hjust=0),
        strip.text=element_text(face="bold",size=10), legend.position="right",
        panel.spacing=unit(1.1,"lines"))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=11.5, height=11)
cat("saved", out, "\n")
