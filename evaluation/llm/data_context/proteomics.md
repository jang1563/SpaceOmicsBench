# Proteomics Data

## Plasma Proteomics
- **Samples**: 21 (4 crew × ~5-6 timepoints, plasma tissue)
- **Proteins measured**: 2,845
- **DE analysis**: 1,686 proteins with differential expression statistics (logFC, AveExpr, t-statistic, B-statistic, adj_pval)

## EVP (Extracellular Vesicle Particle) Proteomics
- **DE analysis**: 496 proteins with DE statistics
- **Overlap with plasma**: 380 proteins measured in both biofluids

## Benchmark Tasks
- **C1**: Proteomics phase classification (3-class: pre_flight/post_flight/recovery)
  - N=21 samples, per-fold PCA reduction (2,845 proteins → 10 components)
  - Note: C1 uses 10 PCA components; G1 multi-modal uses 8 per modality
  - Metric: macro_f1, LOCO evaluation (4-fold), Random baseline: 0.170
  - Best baseline: MLP=0.517 | LogReg=0.512 | RF=0.464 | XGBoost=0.355 | LightGBM=0.228
  - Note: LightGBM collapses to majority (0.228) on small-N LOCO
- **C2**: Cross-biofluid protein DE concordance (frontier difficulty)
  - Predict EVP significance from plasma DE features
  - N=380 overlapping proteins, metric=AUROC, Random baseline: 0.529
  - Best baseline: LightGBM=0.565 | RF=0.555 | XGBoost=0.533 | MLP=0.524 | LogReg=0.500

## Key Observations
- p >> n problem: 2,845 proteins vs 21 samples requires PCA dimensionality reduction
- C1 achieves reasonable classification despite tiny N, suggesting strong proteomic shifts during spaceflight
- C2 is frontier difficulty — plasma and EVP protein changes show weak concordance, reflecting different biological compartments
- EVP cargo represents active cellular secretion, while plasma includes both secreted and leaked proteins
