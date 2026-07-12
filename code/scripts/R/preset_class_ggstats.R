#!/usr/bin/env Rscript
# preset_class_ggstats.R -- Fig 4.2: is the degradation class-agnostic?
# grouped_ggbetweenstats: one panel per noise preset (Low/Mid/High), each comparing the
# three diagnostic classes (Normal/Bacterial/Viral) on per-image pre-denoising PSNR, with
# a Welch test, Bonferroni-adjusted pairwise comparisons and Hedges' g effect sizes.
#
# Input CSV produced by: python scripts/preset_stats.py --data <test> --dump results/preset_class_predenoise.csv
# Deps: readr, dplyr, ggstatsplot, ggplot2
# Usage: Rscript preset_class_ggstats.R --csv results/preset_class_predenoise.csv --out figures/preset_class.pdf
suppressMessages({library(readr); library(dplyr); library(ggstatsplot); library(ggplot2)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[i + 1] }
csv <- getarg("--csv", "results/preset_class_predenoise.csv")
out <- getarg("--out", "preset_class.pdf")

df <- read_csv(csv, show_col_types = FALSE) |>
  mutate(preset = factor(preset, levels = c("low","mid","high"),
                         labels = c("Low noise","Mid noise","High noise")),
         class  = factor(class, levels = c("Normal","Bacterial","Viral")))

p <- grouped_ggbetweenstats(
  data = df, x = class, y = psnr, grouping.var = preset,
  type = "parametric", pairwise.display = "all", p.adjust.method = "bonferroni",
  effsize.type = "unbiased", xlab = NULL, ylab = "pre-denoising PSNR (dB)",
  point.args = list(alpha = 0.12, size = 0.7),
  centrality.label.args = list(size = 2.6),
  ggplot.component = theme(axis.text = element_text(size = 8)),
  plotgrid.args = list(nrow = 1),
  annotation.args = list(
    title = "Degradation is class-agnostic at every dose level",
    subtitle = "per-image pre-denoising PSNR by diagnostic class (Normal / Bacterial / Viral)")
)

dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, p, width = 14, height = 6)
cat("saved", out, "\n")
