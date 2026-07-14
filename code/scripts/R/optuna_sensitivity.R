#!/usr/bin/env Rscript
# optuna_sensitivity.R -- loss-weight sensitivity for Arm D (NLL+SSIM+FFL) from the
# 14-trial Optuna sweep. Top: marginal sensitivity ribbons (validation PSNR vs each
# weight, loess band). Bottom: 2D response surface over (lambda_SSIM, lambda_FFL),
# with the best trial and the literature-default weights marked.
suppressMessages({library(readr);library(dplyr);library(tidyr);library(ggplot2);library(scico);library(patchwork)})
args <- commandArgs(trailingOnly=TRUE); getarg <- function(f,d=NULL){i<-match(f,args); if(is.na(i)) d else args[i+1]}
csv <- getarg("--csv","results/optuna_trials_d.csv"); out <- getarg("--out","optuna_sensitivity.pdf")

d <- read_csv(csv, show_col_types=FALSE) |>
  transmute(number, val_psnr=value,
            lambda_ssim=as.numeric(params_lambda_ssim), lambda_ffl=as.numeric(params_lambda_ffl)) |>
  filter(!is.na(val_psnr))
best <- d |> slice_max(val_psnr, n=1)
# literature default used throughout the thesis: lambda_ssim=0.5, lambda_ffl=0.1
deflt <- d |> filter(abs(lambda_ssim-0.5)<1e-6 & abs(lambda_ffl-0.1)<1e-6) |> slice(1)

# ---- marginal sensitivity ribbons ----
long <- d |> pivot_longer(c(lambda_ssim, lambda_ffl), names_to="param", values_to="weight") |>
  mutate(param=recode(param, lambda_ssim="lambda[SSIM]", lambda_ffl="lambda[FFL]"))
bl  <- best |> pivot_longer(c(lambda_ssim, lambda_ffl), names_to="param", values_to="weight") |>
  mutate(param=recode(param, lambda_ssim="lambda[SSIM]", lambda_ffl="lambda[FFL]"))
p_marg <- ggplot(long, aes(weight, val_psnr)) +
  geom_smooth(method="loess", span=1.0, se=TRUE, fill="#4c72b0", alpha=0.20,
              colour="#1f4e79", linewidth=0.9) +
  geom_point(size=2.4, colour="#333333") +
  geom_point(data=bl, size=3.6, shape=21, fill="#e4572e", colour="black", stroke=0.6) +
  facet_wrap(~param, scales="free_x", labeller=label_parsed) +
  labs(x="loss weight", y="validation PSNR (dB)",
       title="Loss-weight sensitivity for Arm D (NLL + SSIM + FFL)",
       subtitle="14-trial Optuna sweep; shaded ribbon = loess 95% band; orange point = best trial") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), strip.text=element_text(size=12),
        panel.grid.minor=element_blank())

# ---- 2D response surface ----
p_surf <- ggplot(d, aes(lambda_ssim, lambda_ffl)) +
  geom_point(aes(colour=val_psnr), size=6) +
  geom_point(data=best,  shape=8,  size=6, colour="#e4572e", stroke=1.4) +
  { if(nrow(deflt)) geom_point(data=deflt, shape=4, size=5, colour="black", stroke=1.4) } +
  scale_colour_scico(palette="batlow", name="val PSNR (dB)") +
  labs(x=expression(lambda[SSIM]), y=expression(lambda[FFL]),
       title="Response surface over the searched weights",
       subtitle="star = best (0.96, 0.02); cross = literature default (0.5, 0.1)") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), panel.grid.minor=element_blank(),
        legend.position="right")

p <- p_marg / p_surf + plot_layout(heights=c(1, 1.05))
dir.create(dirname(out), showWarnings=FALSE, recursive=TRUE)
ggsave(out, p, width=8.5, height=9.2)
cat("saved", out, " | best:", round(best$val_psnr,3), "dB at ssim", round(best$lambda_ssim,3),
    "ffl", round(best$lambda_ffl,3), "\n")
