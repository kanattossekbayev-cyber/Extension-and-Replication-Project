# ============================================================
# ML Extension (Morozova OAT) — FULL REPRODUCIBLE PIPELINE
# ============================================================

# ----------------------------
# 0) Setup
# ----------------------------
rm(list = ls())

# Set your working directory manually if needed:
# setwd("C:/Users/User/Desktop/Final_project")

cat("Working directory:\n"); print(getwd())
cat("\nFiles in folder (look for smpl_joint_*.Rdata):\n")
print(list.files(pattern = "smpl_joint_.*50K\\.Rdata", ignore.case = TRUE))

# Packages
pkgs <- c("ggplot2", "ranger", "vip", "quantregForest")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

set.seed(42)

# ----------------------------
# 1) Load PSA draws from .Rdata and build psa_all
# ----------------------------
rdata_files <- list.files(pattern = "^smpl_joint_.*50K\\.Rdata$", ignore.case = TRUE)
stopifnot(length(rdata_files) > 0)

psa_list <- lapply(rdata_files, function(f) {
  e <- new.env()
  load(f, envir = e)
  objs <- as.list(e)
  mats <- objs[sapply(objs, is.matrix)]
  if (length(mats) == 0) stop(paste("No matrix found in:", f))
  mats[[1]]
})

max_p <- max(vapply(psa_list, ncol, integer(1)))
common_names <- paste0("V", 1:max_p)

psa_df_list <- lapply(psa_list, function(m) {
  df <- as.data.frame(m)
  colnames(df) <- paste0("V", seq_len(ncol(df)))
  if (ncol(df) < max_p) {
    for (j in (ncol(df)+1):max_p) df[[paste0("V", j)]] <- NA
  }
  df <- df[, common_names, drop = FALSE]
  df
})

psa_all <- do.call(rbind, psa_df_list)

cat("\n✅ psa_all created\n")
cat("Rows:", nrow(psa_all), "| Cols:", ncol(psa_all), "\n")

# ----------------------------
# 2) Define outcomes (NMB)
# ----------------------------
WTP <- 50000
psa_all$delta_cost <- psa_all$V18
psa_all$delta_qaly <- psa_all$V20
psa_all$NMB <- WTP * psa_all$delta_qaly - psa_all$delta_cost

cat("\nNMB summary:\n")
print(summary(psa_all$NMB))

# ----------------------------
# 3) Predictors for ML (drop columns with NA)
# ----------------------------
X_cols <- paste0("V", 1:27)
all_na <- sapply(psa_all[, X_cols], function(x) all(is.na(x)))
X_cols <- X_cols[!all_na]

# Drop columns that contain any NA (needed for quantregForest)
has_any_na <- sapply(psa_all[, X_cols], function(x) any(is.na(x)))
X_cols_clean <- X_cols[!has_any_na]

cat("\nPredictor cols used (no NA):\n")
print(X_cols_clean)

# ----------------------------
# 4) Random Forest (mean NMB) — ranger
# ----------------------------
set.seed(42)
idx_rf <- sample(nrow(psa_all), 50000)
data_rf <- psa_all[idx_rf, ]

rf_nmb <- ranger(
  formula = as.formula(paste("NMB ~", paste(X_cols_clean, collapse = " + "))),
  data = data_rf,
  num.trees = 500,
  max.depth = 6,
  importance = "permutation",
  seed = 42
)

# Save variable importance plot
p_vip <- vip(rf_nmb, num_features = 15)
ggsave("RF_VIP_top15.pdf", plot = p_vip, width = 8, height = 6)

# ----------------------------
# 5) Clustering PSA draws (k-means on PCA)
# ----------------------------
set.seed(42)
idx_cl <- sample(nrow(psa_all), 50000)
df_cl <- psa_all[idx_cl, ]

# Choose a subset of "parameter-like" columns for clustering
cluster_cols <- paste0("V", 7:17)
X_cl <- scale(df_cl[, cluster_cols])

pca <- prcomp(X_cl)
wss <- sapply(2:6, function(k) kmeans(pca$x[, 1:5], centers = k, nstart = 25)$tot.withinss)

# Elbow plot
pdf("Clustering_elbow.pdf", width = 7, height = 5)
plot(2:6, wss, type = "b", xlab = "Number of clusters (K)", ylab = "Within-cluster sum of squares")
dev.off()

# Pick K=3 (your choice)
K <- 3
km <- kmeans(pca$x[, 1:5], centers = K, nstart = 50)
df_cl$cluster <- factor(km$cluster)

# Cluster NMB summary table
cluster_summary <- aggregate(NMB ~ cluster, data = df_cl, function(x)
  c(mean = mean(x), median = median(x), p10 = quantile(x, 0.10), p90 = quantile(x, 0.90))
)

cluster_summary_df <- data.frame(
  cluster    = cluster_summary$cluster,
  NMB_mean   = cluster_summary$NMB[, 1],
  NMB_median = cluster_summary$NMB[, 2],
  NMB_p10    = cluster_summary$NMB[, 3],
  NMB_p90    = cluster_summary$NMB[, 4]
)

write.csv(cluster_summary_df, "cluster_NMB_summary.csv", row.names = FALSE)

# Plot NMB by cluster
p_cluster <- ggplot(df_cl, aes(x = cluster, y = NMB)) +
  geom_boxplot() +
  labs(title = "NMB by system cluster", x = "Cluster", y = "NMB")
ggsave("Cluster_NMB_boxplot.pdf", plot = p_cluster, width = 8, height = 6)

# ----------------------------
# 6) Quantile Random Forest (tail risk)
# ----------------------------
set.seed(42)
idx_qrf <- sample(nrow(psa_all), 15000)
df_qrf <- psa_all[idx_qrf, ]

X <- df_qrf[, X_cols_clean, drop = FALSE]
y <- df_qrf$NMB

qrf <- quantregForest(x = X, y = y, ntree = 200, nodesize = 50, importance = TRUE)

pred_q10 <- predict(qrf, X, what = 0.10)
pred_q90 <- predict(qrf, X, what = 0.90)

# Save q10/q90 histograms
df_pred <- data.frame(
  NMB = c(pred_q10, pred_q90),
  Quantile = rep(c("q10 (worst)", "q90 (best)"), each = length(pred_q10))
)

p_qtails <- ggplot(df_pred, aes(x = NMB)) +
  geom_histogram(bins = 40) +
  facet_wrap(~Quantile, scales = "free_y") +
  labs(title = "Predicted NMB: q10 vs q90", x = "Predicted NMB", y = "Count")

ggsave("QRF_q10_q90_hist.pdf", plot = p_qtails, width = 9, height = 5)

# Importance table + plot
imp_sorted <- sort(importance(qrf), decreasing = TRUE)
names(imp_sorted) <- colnames(X)

imp_top <- data.frame(
  Variable = names(imp_sorted)[1:min(10, length(imp_sorted))],
  Importance = as.numeric(imp_sorted[1:min(10, length(imp_sorted))])
)

write.csv(imp_top, "QRF_importance_top10.csv", row.names = FALSE)

p_imp <- ggplot(imp_top, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() + coord_flip() +
  labs(title = "Top 10 variable importance (QRF)", x = "Variable", y = "Importance")

ggsave("QRF_importance_top10.pdf", plot = p_imp, width = 8, height = 6)

# ----------------------------
# 7) QRF heatmap surface for V12 x V8 (median prediction)
# ----------------------------
# Note: requires V8 and V12 present in X
stopifnot(all(c("V8", "V12") %in% colnames(X)))

v12_seq <- seq(quantile(X$V12, 0.05), quantile(X$V12, 0.95), length.out = 25)
v8_seq  <- seq(quantile(X$V8,  0.05), quantile(X$V8,  0.95), length.out = 25)

grid <- expand.grid(V12 = v12_seq, V8 = v8_seq)
base <- as.data.frame(lapply(X, median))

newdata <- base[rep(1, nrow(grid)), ]
newdata$V12 <- grid$V12
newdata$V8  <- grid$V8

grid$NMB_q50 <- as.numeric(predict(qrf, newdata, what = 0.50))

p_heat <- ggplot(grid, aes(x = V12, y = V8, fill = NMB_q50)) +
  geom_tile() +
  labs(title = "QRF median NMB surface: V12 x V8", x = "V12", y = "V8", fill = "Predicted NMB (q50)")

ggsave("QRF_heatmap_V12_V8.pdf", plot = p_heat, width = 8, height = 6)

cat("\n✅ DONE. Outputs saved as PDF/CSV in the project folder.\n")
