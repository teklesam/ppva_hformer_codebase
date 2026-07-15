#!/usr/bin/env Rscript
# optuna_perceptual.R -- perceptual view of the Arm-D loss-weight sweep.
# PSNR is confounded with the (SSIM/FFL) training objective; SSIM, FSIM and LPIPS
# give a less objective-aligned read on how loss-weighting shifts image quality.
# Question answered: do the less-confounded metrics agree with PSNR about WHICH
# weights are best? Top: per-trial quality (all four metrics, normalised to a common
# 0-1 "better" axis, LPIPS inverted) vs lambda_SSIM, showing co-movement. Bottom:
# response surfaces over (lambda_SSIM, lambda_FFL), one per metric, best point starred.
suppressMessages({library(readr);library(dplyr);library(tidyr);library(ggplot2);library(scico);library(patchwork)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
csv <- getarg("--csv","results/trials_perimage_metrics.csv"); out <- getarg("--out","optuna_perceptual.pdf")

raw <- read_csv(csv, show_col_types=FALSE)
# per-trial means; LPIPS negated so that higher = better for every metric
agg <- raw |> group_by(trial, lambda_ssim, lambda_ffl) |>
  summarise(PSNR=mean(psnr), SSIM=mean(ssim), FSIM=mean(fsim), LPIPS=mean(lpips), .groups="drop")
q <- agg |> mutate(`LPIPS*`=-LPIPS) |>
  pivot_longer(c(PSNR,SSIM,FSIM,`LPIPS*`), names_to="metric", values_to="value") |>
  group_by(metric) |> mutate(quality=(value-min(value))/(max(value)-min(value))) |> ungroup() |>
  mutate(metric=factor(metric, levels=c("PSNR","SSIM","FSIM","LPIPS*")))

# Spearman rank agreement with PSNR across the 14 trials (LPIPS negated -> higher better)
rho <- agg |> summarise(SSIM=cor(PSNR,SSIM,method="spearman"),
                        FSIM=cor(PSNR,FSIM,method="spearman"),
                        LPIPS=cor(PSNR,-LPIPS,method="spearman"))
best_ssim <- agg$lambda_ssim[which.max(agg$PSNR)]; best_ffl <- agg$lambda_ffl[which.max(agg$PSNR)]
sub <- sprintf("Spearman rank agreement with PSNR across 14 trials: SSIM r_s=%.2f, FSIM r_s=%.2f, LPIPS r_s=%.2f",
               rho$SSIM, rho$FSIM, rho$LPIPS)
wong <- c(PSNR="#0072B2", SSIM="#009E73", FSIM="#D55E00", `LPIPS*`="#CC79A7")

# ---- top: normalised quality vs lambda_SSIM (co-movement) ----
p_top <- ggplot(q, aes(lambda_ssim, quality, colour=metric, fill=metric)) +
  geom_smooth(method="loess", span=1.1, se=FALSE, linewidth=0.9) +
  geom_point(size=2.3, alpha=0.9) +
  scale_colour_manual(values=wong, name=NULL) + scale_fill_manual(values=wong, guide="none") +
  labs(x=expression(lambda[SSIM]), y="normalised quality (0-1, higher better)",
       title="Loss-weight sensitivity agrees across distortion and perceptual metrics",
       subtitle=sub) +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), panel.grid.minor=element_blank(), legend.position="top")

# ---- bottom: response surface per metric, best point starred ----
qbest <- q |> group_by(metric) |> slice_max(quality, n=1) |> ungroup()
p_bot <- ggplot(q, aes(lambda_ssim, lambda_ffl)) +
  geom_point(aes(colour=quality), size=4.6) +
  geom_point(data=qbest, shape=8, size=5, colour="#e4572e", stroke=1.3) +
  scale_colour_scico(palette="batlow", name="quality") +
  facet_wrap(~metric, nrow=1) +
  labs(x=expression(lambda[SSIM]), y=expression(lambda[FFL]),
       subtitle="Response surface per metric; star = best weighting for that metric") +
  theme_minimal(base_size=10.5) +
  theme(panel.grid.minor=element_blank(), strip.text=element_text(face="bold"))

p <- p_top / p_bot + plot_layout(heights=c(1, 0.85))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=9.2, height=8.4)
cat("saved", out, "\n"); cat(sub, "\n")
cat(sprintf("PSNR-best weighting: lambda_SSIM=%.2f lambda_FFL=%.3f\n", best_ssim, best_ffl))
cat("best-weight agreement (which lambda_SSIM,lambda_FFL each metric prefers):\n")
print(qbest |> select(metric, lambda_ssim, lambda_ffl, value))
