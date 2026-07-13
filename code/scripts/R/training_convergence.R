#!/usr/bin/env Rscript
# training_convergence.R -- per-arm training loss + validation PSNR, faceted metric x family.
# Shows the well-trained arms plateau (converged) while failure arms visibly break.
# Deps: readr, dplyr, tidyr, ggplot2, purrr
suppressMessages({library(readr);library(dplyr);library(tidyr);library(ggplot2);library(purrr);library(ggrepel)})
args <- commandArgs(trailingOnly=TRUE)
getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
base <- getarg("--logs","/tmp/csd3_pull/trainlogs")
out  <- getarg("--out","training_convergence.pdf")

fam <- tribble(~arm,~lab,~family,
 "arm_a_l2","A","Pixel-norm","arm_i_l1","I","Pixel-norm","arm_j_l1_ssim_ffl","J","Pixel-norm","arm_q_charb","Q","Pixel-norm",
 "arm_b_nll","B","NLL","arm_c_nll_ssim","C","NLL","arm_d_nll_ssim_ffl","D","NLL","arm_k_nll_l1","K","NLL",
 "arm_l_nll_edge_ffl","L","NLL","arm_m_full_det","M","NLL","arm_n_perc","N","NLL","arm_o_prelu","O","NLL",
 "arm_e_ppvae","E","VAE","arm_f_kl_cyc","F","VAE","arm_g_kl_fb","G","VAE","arm_h_kl_cyc_fb","H","VAE","arm_p_best","P","VAE",
 "arm_r_ft_j","R","Fine-tune","arm_s_ft_d","S","Fine-tune",
 "dncnn_baseline","DnCNN","Baselines","ircnn","IRCNN","Baselines","ffdnet","FFDNet","Baselines",
 "drunet","DRUNet","Baselines","swinir","SwinIR","Baselines","nafnet","NAFNet","Baselines","scunet","SCUNet","Baselines")

read_one <- function(arm){
  f <- file.path(base,arm,"train_log.csv"); if(!file.exists(f)) f<-file.path(base,"baselines",arm,"train_log.csv")
  if(!file.exists(f)) return(NULL)
  d <- suppressWarnings(read_csv(f, show_col_types=FALSE, progress=FALSE))
  lc <- if("train_loss"%in%names(d))"train_loss" else if("loss"%in%names(d))"loss" else return(NULL)
  if(!"val_psnr"%in%names(d)) return(NULL)
  tibble(epoch=as.numeric(d$epoch), loss=suppressWarnings(as.numeric(d[[lc]])),
         val_psnr=suppressWarnings(as.numeric(d$val_psnr)))
}
df <- fam %>% mutate(d=map(arm,read_one)) %>% filter(!map_lgl(d,is.null)) %>% unnest(d) %>%
  group_by(arm) %>% filter(min(epoch,na.rm=TRUE) <= 10) %>% mutate(loss_n=(loss-min(loss,na.rm=TRUE))/(max(loss,na.rm=TRUE)-min(loss,na.rm=TRUE))) %>% ungroup()
df$family <- factor(df$family, levels=c("Pixel-norm","NLL","VAE","Fine-tune","Baselines"))

L <- bind_rows(
  df %>% filter(is.finite(loss_n)) %>% transmute(arm,lab,family,epoch,metric="Training loss (per-arm normalised)",y=loss_n),
  df %>% filter(is.finite(val_psnr)) %>% transmute(arm,lab,family,epoch,metric="Validation PSNR (dB)",y=val_psnr))
L$metric <- factor(L$metric, levels=c("Training loss (per-arm normalised)","Validation PSNR (dB)"))
ends <- L %>% group_by(arm,metric) %>% slice_max(epoch,n=1,with_ties=FALSE) %>% ungroup()

p <- ggplot(L, aes(epoch,y,color=lab,group=arm)) +
  geom_line(linewidth=0.55, alpha=0.9) +
  ggrepel::geom_text_repel(data=ends, aes(label=lab), size=2.4, fontface="bold",
    direction="y", hjust=0, nudge_x=6, segment.size=0.25, segment.alpha=0.5,
    min.segment.length=0, box.padding=0.12, point.padding=0.1, max.overlaps=Inf,
    seed=1, show.legend=FALSE) +
  facet_grid(metric~family, scales="free_y", switch="y") +
  scale_x_continuous(expand=expansion(mult=c(0.02,0.22))) +
  labs(x="epoch", y=NULL, title="Training convergence by loss / architecture family",
       caption="Per-arm curves; validation sampled every 5 epochs. Well-trained arms plateau (converged); Arm E (VAE posterior collapse) settles to a low ceiling and Arm O (PReLU) diverges.") +
  guides(color="none") +
  theme_minimal(base_size=9) +
  theme(strip.text=element_text(face="bold",size=8), strip.placement="outside",
        plot.title=element_text(face="bold"), panel.grid.minor=element_blank(),
        panel.spacing=unit(0.5,"lines"))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=13.5, height=6.2)
cat("saved",out,"| arms:",length(unique(L$arm)),"\n")
