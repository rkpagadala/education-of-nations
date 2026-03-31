#!/usr/bin/env Rscript
# 01_download.R
# Download all required WCDE v3 data for education analysis.
# Uses SSP2 (scenario=2, medium) for all years including 2020 and 2025.
# Historical years (1960-2015): reconstructed empirical data in v3
# Projection years (2020, 2025): SSP2 medium scenario
#
# Outputs (saved to wcde/data/raw/):
#   prop_both.csv    — educational attainment % by age, both sexes
#   prop_female.csv  — educational attainment % by age, female
#   tfr.csv          — total fertility rate by country-year
#   e0_both.csv      — life expectancy at birth, both sexes
#   e0_female.csv    — life expectancy at birth, female

.libPaths("~/R/libs")
library(wcde)

args <- commandArgs(trailingOnly=FALSE)
file_arg <- grep("--file=", args, value=TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}
OUT <- file.path(dirname(script_dir), "data", "raw")
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)
cat("Output directory:", OUT, "\n")

cat("Downloading prop (education attainment %) — both sexes...\n")
prop_both <- get_wcde(
  indicator   = "prop",
  scenario    = 2,
  pop_age     = "all",
  pop_sex     = "both",
  pop_edu     = "six",
  version     = "wcde-v3"
)
write.csv(prop_both, file.path(OUT, "prop_both.csv"), row.names=FALSE)
cat("  Saved prop_both.csv —", nrow(prop_both), "rows\n")

cat("Downloading prop (education attainment %) — female...\n")
prop_female <- get_wcde(
  indicator   = "prop",
  scenario    = 2,
  pop_age     = "all",
  pop_sex     = "female",
  pop_edu     = "six",
  version     = "wcde-v3"
)
write.csv(prop_female, file.path(OUT, "prop_female.csv"), row.names=FALSE)
cat("  Saved prop_female.csv —", nrow(prop_female), "rows\n")

cat("Downloading TFR...\n")
tfr_data <- get_wcde(
  indicator = "tfr",
  scenario  = 2,
  version   = "wcde-v3"
)
write.csv(tfr_data, file.path(OUT, "tfr.csv"), row.names=FALSE)
cat("  Saved tfr.csv —", nrow(tfr_data), "rows\n")

cat("Downloading life expectancy (both)...\n")
e0_both <- get_wcde(
  indicator = "e0",
  scenario  = 2,
  pop_sex   = "both",
  version   = "wcde-v3"
)
write.csv(e0_both, file.path(OUT, "e0_both.csv"), row.names=FALSE)
cat("  Saved e0_both.csv —", nrow(e0_both), "rows\n")

cat("Downloading life expectancy (female)...\n")
e0_female <- get_wcde(
  indicator = "e0",
  scenario  = 2,
  pop_sex   = "female",
  version   = "wcde-v3"
)
write.csv(e0_female, file.path(OUT, "e0_female.csv"), row.names=FALSE)
cat("  Saved e0_female.csv —", nrow(e0_female), "rows\n")

cat("Downloading pop (population counts in thousands) — both sexes...\n")
pop_both <- get_wcde(
  indicator   = "pop",
  scenario    = 2,
  pop_age     = "all",
  pop_sex     = "both",
  pop_edu     = "six",
  version     = "wcde-v3"
)
write.csv(pop_both, file.path(OUT, "pop_both.csv"), row.names=FALSE)
cat("  Saved pop_both.csv —", nrow(pop_both), "rows\n")

cat("Downloading pop (population counts in thousands) — female (via both, sex disaggregated)...\n")
# pop indicator uses pop_sex="both" which includes female/male breakdown in the data
cat("  Note: pop_both.csv already contains female breakdown by sex column\n")

cat("\nDownload complete. Files in:", OUT, "\n")
cat("Countries in prop_both:", length(unique(prop_both$name)), "\n")
cat("Years:", paste(sort(unique(prop_both$year)), collapse=", "), "\n")
cat("Education levels:", paste(unique(prop_both$education), collapse=", "), "\n")
