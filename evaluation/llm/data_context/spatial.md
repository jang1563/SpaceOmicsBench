# Spatial Transcriptomics Data

## Dataset
- **Source**: Inspiration4 skin biopsies
- **Genes**: 18,677 measured across skin tissue layers
- **Layers analyzed**: outer_epidermis, inner_epidermis, epidermis (combined), outer_dermis

## All-Skin DE Analysis (Feature Source)
- Baseline DE statistics computed across entire skin tissue
- Features per gene: baseMean, log2FoldChange, lfcSE (standard error)
- These all-skin features serve as input for layer-specific prediction

## Layer-Specific DE (Labels)
- Separate DE analysis per skin layer
- Significance threshold: adjusted p-value < 0.05
- Positive counts: outer_epidermis ~35, epidermis ~40, inner_epidermis ~11, outer_dermis ~18

## Benchmark Tasks
- **E1**: Cross-layer DE prediction (outer_epidermis) — Advanced
  - N=18,677, positives=35 (0.19%), AUPRC, Random: 0.008
  - Best: LogReg=0.017 | RF=0.015 | XGBoost=0.010 | LightGBM=0.005 | MLP=0.003
- **E4**: Cross-layer DE prediction (epidermis) — Advanced
  - N=18,677, positives=~40 (0.21%), AUPRC, Random: 0.003
  - Best: LogReg=0.022 | LightGBM=0.009 | XGBoost=0.006 | MLP=0.003 | RF=0.002
- **E2**: Cross-layer DE (inner_epidermis) — Supplementary/Frontier
  - Extreme imbalance (~11 positives), AUPRC, Random: 0.001
  - Best: RF=0.050 | LogReg=0.031 | XGBoost=0.020 | MLP=0.011 | LightGBM=0.005
- **E3**: Cross-layer DE (outer_dermis) — Supplementary/Frontier
  - Extreme imbalance (~18 positives), AUPRC, Random: 0.002
  - Best: RF=0.223 | LogReg=0.172 | MLP=0.168 | XGBoost=0.160 | LightGBM=0.088

## Key Observations
- Extreme class imbalance (0.1-0.2% positive rate) necessitates AUPRC metric
- All-skin DE features have limited power for predicting layer-specific effects
- Skin is directly exposed to space radiation; layer-specific responses may reflect depth-dependent radiation exposure
- E2/E3 are supplementary due to metric instability from very few positives
