# SPDX-License-Identifier: LGPL-3.0-only

snftool_dir <- Sys.getenv("SNFTOOL_DIR")
if (snftool_dir == "") {
  stop("Set SNFTOOL_DIR to the pinned SNFtool checkout")
}

expected_revision <- "64ade299d2cd10e6605063637fed1103564d0ea4"
actual_revision <- system2(
  "git",
  c("-C", snftool_dir, "rev-parse", "HEAD"),
  stdout = TRUE
)
if (!identical(actual_revision, expected_revision)) {
  stop(
    "SNFtool checkout must be pinned to ",
    expected_revision,
    "; found ",
    actual_revision
  )
}
if (
  length(system2(
    "git",
    c("-C", snftool_dir, "status", "--porcelain"),
    stdout = TRUE
  ))
) {
  stop("SNFtool checkout must have a clean worktree")
}

source(file.path(snftool_dir, "R", "dist2.R"))
source(file.path(snftool_dir, "R", "affinityMatrix.R"))
source(file.path(snftool_dir, "R", "internal.R"))
source(file.path(snftool_dir, "R", "SNF.R"))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- sub("^--file=", "", script_arg[[1]])
data_dir <- file.path(dirname(normalizePath(script_path)), "data")

read_features <- function(name) {
  as.matrix(read.csv(file.path(data_dir, name), header = FALSE))
}

write_fixture <- function(value, name) {
  write.table(
    value,
    file.path(data_dir, name),
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

n_neighbors <- 3
scale <- 0.5
features_1 <- read_features("features_1.csv")
features_2 <- read_features("features_2.csv")

euclidean_1 <- sqrt(dist2(features_1, features_1))
euclidean_2 <- sqrt(dist2(features_2, features_2))
squared_euclidean_1 <- dist2(features_1, features_1)

affinity_1 <- affinityMatrix(euclidean_1, n_neighbors, scale)
affinity_2 <- affinityMatrix(euclidean_2, n_neighbors, scale)
affinity_sqeuclidean_1 <- affinityMatrix(
  squared_euclidean_1,
  n_neighbors,
  scale
)

write_fixture(affinity_1, "affinity_1.csv")
write_fixture(affinity_2, "affinity_2.csv")
write_fixture(affinity_sqeuclidean_1, "affinity_sqeuclidean_1.csv")
write_fixture(
  SNF(list(affinity_1, affinity_2), n_neighbors, 1),
  "fused_t1.csv"
)
write_fixture(
  SNF(list(affinity_1, affinity_2), n_neighbors, 20),
  "fused_t20.csv"
)
