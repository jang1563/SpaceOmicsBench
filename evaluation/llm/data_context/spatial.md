# Spatial Transcriptomics Data

## Dataset
- **Source**: Inspiration4 skin biopsies
- **Genes**: 18,677 measured across skin tissue layers
- **Layers analyzed**: outer_epidermis, inner_epidermis, epidermis (combined), outer_dermis, vasculature

## All-Skin DE Analysis (Feature Source)
- Baseline DE statistics computed across entire skin tissue
- Features per gene: baseMean, log2FoldChange, lfcSE (standard error)
- These all-skin features serve as input for layer-specific prediction

## Layer-Specific DE (Labels)
- Separate DE analysis per skin layer
- Significance threshold: adjusted p-value < 0.05
- Positive counts: outer_epidermis ~35, epidermis ~40, inner_epidermis ~15, outer_dermis ~18

## Benchmark Tasks
- **E1**: Cross-layer DE prediction (outer_epidermis) — Advanced
  - N=18,677, positives=35 (0.19%), AUPRC, Best: LogReg=0.017
- **E4**: Cross-layer DE prediction (epidermis) — Advanced
  - N=18,677, positives=~40 (0.21%), AUPRC, Best: LogReg=0.023
- **E2**: Cross-layer DE (inner_epidermis) — Supplementary/Frontier
  - Extreme imbalance (~15 positives), metric instability expected
- **E3**: Cross-layer DE (outer_dermis) — Supplementary/Frontier
  - Extreme imbalance (~18 positives), Best: RF=0.223

## Key Observations
- Extreme class imbalance (0.1-0.2% positive rate) necessitates AUPRC metric
- All-skin DE features have limited power for predicting layer-specific effects
- Skin is directly exposed to space radiation; layer-specific responses may reflect depth-dependent radiation exposure
- E2/E3 are supplementary due to metric instability from very few positives
