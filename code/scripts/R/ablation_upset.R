#!/usr/bin/env Rscript
# ablation_upset.R -- UpSet plot of the ablation design: which loss / architecture
# terms each of the 19 arms activates (a visual of the loss-summary table). Intersection
# bars count the arms sharing a term-combination (e.g. the {NLL,SSIM,FFL,VAE} block is
# the four KL-schedule arms E/F/G/H; {NLL,SSIM,FFL} is shared by D and its fine-tuned S).
#
# Deps: dplyr, ComplexUpset, ggplot2, patchwork
# Usage:
#   Rscript ablation_upset.R --out figures/ablation_upset.pdf

suppressMessages({library(dplyr); library(ComplexUpset); library(ggplot2); library(patchwork)})

args <- commandArgs(trailingOnly = TRUE)
out  <- { i <- match("--out", args); if (is.na(i)) "ablation_upset.pdf" else args[i + 1] }

terms <- c("MSE", "L1", "Charbonnier", "NLL", "SSIM", "FFL", "Edge", "Perceptual", "VAE", "PReLU")
# arm -> active terms (from the loss-summary table); E/F/G/H differ only in KL schedule,
# so they share the {NLL,SSIM,FFL,VAE} combination.
membership <- list(
  A = "MSE", B = "NLL", C = c("NLL","SSIM"), D = c("NLL","SSIM","FFL"),
  E = c("NLL","SSIM","FFL","VAE"), F = c("NLL","SSIM","FFL","VAE"),
  G = c("NLL","SSIM","FFL","VAE"), H = c("NLL","SSIM","FFL","VAE"),
  I = "L1", J = c("L1","SSIM","FFL"), K = c("NLL","L1"), L = c("NLL","Edge","FFL"),
  M = c("NLL","SSIM","Edge","FFL"), N = c("NLL","Perceptual","SSIM","FFL"),
  O = c("NLL","SSIM","FFL","PReLU"), P = c("NLL","SSIM","FFL","Edge","VAE","PReLU"),
  Q = "Charbonnier", R = c("L1","SSIM","FFL"), S = c("NLL","SSIM","FFL")
)
df <- as.data.frame(do.call(rbind, lapply(names(membership), function(a)
        setNames(terms %in% membership[[a]], terms))))
df$Arm <- names(membership)

p <- upset(
  df, intersect = terms, name = "loss / architecture term",
  width_ratio = 0.18, sort_intersections_by = "degree", sort_sets = FALSE,
  base_annotations = list("Arms" = intersection_size(counts = TRUE, text = list(size = 3))),
  set_sizes = upset_set_size() + ylab("arms using term")
) + plot_annotation(
  title = "Ablation design: loss / architecture terms activated by each arm"
)

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 11, height = 6.2)
cat("saved", out, "\n")
