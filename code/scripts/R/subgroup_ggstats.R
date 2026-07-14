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
    point.args=list(alpha=0.18, size=1, position=position_jitterdodge(dodge.width=0.6)),
    violin.args=list(width=0.6, alpha=0.25),
    ggtheme=theme_minimal(base_size=10),
    ggplot.component=list(scale_colour_manual(values=pal),
                          labs(y=paste0(ylab," (",better,")"), x=NULL)),
    plotgrid.args=list(nrow=1),
    annotation.args=list(title=paste0("Reconstruction ", ylab,
       " by diagnostic class, five representative models (mid noise, Foi)"),
       caption="Welch ANOVA per model; Games-Howell pairwise, Bonferroni-adjusted. Boxes = median/IQR; diamonds = mean.")
  )
}

p_psnr  <- mk("psnr",  "PSNR (dB)", "higher is better")
ggsave(file.path(outdir,"subgroup_psnr_ggstats.pdf"),  p_psnr,  width=17.5, height=6.6)
p_lpips <- mk("lpips", "LPIPS",     "lower is better")
ggsave(file.path(outdir,"subgroup_lpips_ggstats.pdf"), p_lpips, width=17.5, height=6.6)
cat("saved subgroup_psnr_ggstats.pdf and subgroup_lpips_ggstats.pdf\n")
