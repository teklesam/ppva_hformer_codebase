#!/usr/bin/env Rscript
# subgroup_ggstats.R -- ggstatsplot summary of reconstruction quality by diagnostic
# class (Normal / Bacterial / Viral) for the five Table-4.7 models incl. SCUNet.
# One faceted panel per model; within each, the three classes are compared with a
# Welch ANOVA + Games-Howell pairwise tests, Bonferroni-adjusted (ggstatsplot).
suppressMessages({library(ggstatsplot);library(ggplot2);library(readr);library(dplyr)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
csv    <- getarg("--csv","results/subgroup_perimage.csv")
outdir <- getarg("--outdir",".")

d <- read_csv(csv, show_col_types=FALSE) |>
  mutate(class=factor(class, levels=c("Normal","Bacterial","Viral")),
         model=factor(model, levels=c("SCUNet","A (L2)","D (NLL+SSIM+FFL)","H (PP-VAE)","DnCNN")))

pal <- c(Normal="#4c72b0", Bacterial="#dd8452", Viral="#55a868")

mk <- function(yvar, ylab, better){
  grouped_ggbetweenstats(
    data=d, x=class, y=!!rlang::sym(yvar), grouping.var=model,
    type="parametric", pairwise.comparisons=TRUE, pairwise.display="significant",
    p.adjust.method="bonferroni", centrality.plotting=TRUE,
    point.args=list(alpha=0.14, size=0.8, position=position_jitterdodge(dodge.width=0.6)),
    violin.args=list(width=0.6, alpha=0.25),
    centrality.label.args=list(size=2.6, nudge_x=0.28, segment.linetype=4),
    ggsignif.args=list(textsize=2.4, tip_length=0.01, na.rm=TRUE),
    ggtheme=theme_minimal(base_size=10),
    ggplot.component=list(scale_colour_manual(values=pal),
                          labs(y=paste0(ylab," (",better,")"), x=NULL),
                          theme(plot.subtitle=element_text(size=7.2),
                                plot.title=element_text(size=9, face="bold"),
                                axis.title.y=element_text(size=9), axis.text=element_text(size=8),
                                plot.margin=margin(6, 6, 4, 6))),
    plotgrid.args=list(nrow=2),
    annotation.args=list(title=paste0("Reconstruction ", ylab,
       " by diagnostic class, five representative models (mid noise, Foi)"),
       caption="Welch ANOVA per model; Games-Howell pairwise, Bonferroni-adjusted. Boxes = median/IQR; diamonds = mean.")
  )
}

p_psnr  <- mk("psnr",  "PSNR (dB)", "higher is better")
ggsave(file.path(outdir,"subgroup_psnr_ggstats.pdf"),  p_psnr,  width=13.5, height=10.5)
p_lpips <- mk("lpips", "LPIPS",     "lower is better")
ggsave(file.path(outdir,"subgroup_lpips_ggstats.pdf"), p_lpips, width=13.5, height=10.5)
cat("saved subgroup_psnr_ggstats.pdf and subgroup_lpips_ggstats.pdf\n")
