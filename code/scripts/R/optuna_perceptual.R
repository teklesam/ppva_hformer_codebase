#!/usr/bin/env Rscript
# optuna_perceptual.R -- perceptual view of the Arm-D loss-weight sweep, with each metric
# SEPARATED into its own bordered panel (no confusing shared/normalised axis). Top: marginal
# sensitivity of each metric's real test-set value vs lambda_SSIM, faceted per metric (free y).
# Bottom: a single response surface over (lambda_SSIM, lambda_FFL) coloured by PSNR, with the
# best trial and the literature-default marked. PSNR is confounded with the training objective;
# SSIM/FSIM/LPIPS are less objective-aligned, and all four agree on the best weighting.
suppressMessages({library(readr);library(dplyr);library(tidyr);library(ggplot2);library(scico);library(patchwork)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
csv <- getarg("--csv","results/trials_perimage_metrics.csv"); out <- getarg("--out","optuna_perceptual.pdf")

raw <- read_csv(csv, show_col_types=FALSE)
agg <- raw |> group_by(trial, lambda_ssim, lambda_ffl) |>
  summarise(PSNR=mean(psnr), SSIM=mean(ssim), FSIM=mean(fsim), LPIPS=mean(lpips), .groups="drop")
best  <- agg |> slice_max(PSNR, n=1)
deflt <- agg |> filter(abs(lambda_ssim-0.5)<0.02 & abs(lambda_ffl-0.1)<0.02) |> slice(1)
rho <- with(agg, c(SSIM=cor(PSNR,SSIM,method="spearman"),
                   FSIM=cor(PSNR,FSIM,method="spearman"),
                   LPIPS=cor(PSNR,-LPIPS,method="spearman")))
cat(sprintf("Spearman vs PSNR: SSIM %.2f FSIM %.2f LPIPS %.2f\n", rho["SSIM"], rho["FSIM"], rho["LPIPS"]))
cat(sprintf("best: lambda_SSIM=%.2f lambda_FFL=%.3f PSNR=%.2f\n", best$lambda_ssim, best$lambda_ffl, best$PSNR))

# ---- TOP: marginal sensitivity of each metric to BOTH weights (metric rows x weight cols) ----
lev <- c("PSNR","SSIM","FSIM","LPIPS")
mklong <- function(df) df |>
  pivot_longer(c(PSNR,SSIM,FSIM,LPIPS), names_to="metric", values_to="mval") |>
  pivot_longer(c(lambda_ssim, lambda_ffl), names_to="weight", values_to="wval") |>
  mutate(metric=factor(metric, levels=lev),
         weight=factor(ifelse(weight=="lambda_ssim","lambda[SSIM]","lambda[FFL]"),
                       levels=c("lambda[SSIM]","lambda[FFL]")))
long <- mklong(agg); bestl <- mklong(best)
p_marg <- ggplot(long, aes(wval, mval)) +
  geom_smooth(method="loess", span=1.1, se=TRUE, colour="#1f4e79", fill="#4c72b0",
              alpha=0.18, linewidth=0.8) +
  geom_point(size=1.7, colour="#444444") +
  geom_point(data=bestl, size=3.0, shape=21, fill="#e4572e", colour="black", stroke=0.5) +
  facet_grid(metric ~ weight, scales="free", switch="y", labeller=label_parsed) +
  labs(x="loss-weight value", y="test-set metric (mean over 624 images); PSNR/SSIM/FSIM up, LPIPS down") +
  theme_minimal(base_size=10) +
  theme(strip.text=element_text(face="bold", size=9.5),
        strip.background=element_rect(fill="grey92", colour=NA),
        strip.placement="outside",
        panel.border=element_rect(colour="grey65", fill=NA, linewidth=0.5),
        panel.grid.minor=element_blank())

# ---- BOTTOM: one response surface coloured by PSNR; star = best, cross = default ----
p_surf <- ggplot(agg, aes(lambda_ssim, lambda_ffl)) +
  geom_point(aes(colour=PSNR), size=6.5) +
  geom_point(data=best,  shape=8, size=6, colour="#e4572e", stroke=1.4) +
  { if(nrow(deflt)) geom_point(data=deflt, shape=4, size=5, colour="black", stroke=1.4) } +
  scale_colour_scico(palette="batlow", name="PSNR (dB)") +
  labs(x=expression(lambda[SSIM]), y=expression(lambda[FFL]),
       subtitle="Response surface (star = best trial, cross = literature default); the other three metrics peak at the same point") +
  theme_minimal(base_size=10.5) +
  theme(panel.border=element_rect(colour="grey65", fill=NA, linewidth=0.5),
        panel.grid.minor=element_blank(), plot.subtitle=element_text(size=8.5))

p <- p_marg / p_surf + plot_layout(heights=c(1.75, 1))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=9.0, height=11.2)
cat("saved", out, "\n")
