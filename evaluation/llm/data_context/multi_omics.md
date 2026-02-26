# Multi-Omics Integration and Cross-Tissue Analysis

## G1: Multi-Modal Phase Classification
- **Modalities fused**: Clinical (CBC + CMP) + Proteomics (PCA) + Metabolomics (PCA)
- **Matched samples**: 21 (only timepoints where all 3 modalities were collected)
- **Feature construction**:
  - Clinical: ~40 raw CBC + CMP features
  - Proteomics: PCA of 2,845 proteins → 8 components
  - Metabolomics: PCA of metabolite matrix → 8 components
  - Total: ~56 fused features (PCA applied per-fold to prevent data leakage)
- **Task**: 3-class phase classification (pre_flight / post_flight / recovery)
- **Evaluation**: LOCO (4-fold), macro_f1
- **Results**: LogReg=0.517 | XGBoost=0.328 | MLP=0.285 | RF=0.254 | LightGBM=0.228, Random: 0.253

## H1: Cross-Tissue Gene Conservation (PBMC → Skin)
- **Question**: Do PBMC DE patterns predict skin DE?
- **N**: 731 genes tested in both PBMC and skin
- **Features**: 9 I4 PBMC cell type log2FC values (CD4_T, CD8_T, other_T, B, NK, CD14_Mono, CD16_Mono, DC, other)
- **Target**: Whether gene is also significantly DE in skin (binary)
- **Metric**: AUPRC
- **Results**: LightGBM=0.284 | RF=0.266 | XGBoost=0.213 | LogReg=0.176 | MLP=0.062, Random: 0.060

## Key Observations
- Multi-modal fusion (G1) improves over single-modality approaches, but small N (21) limits gains
- PCA is necessary for proteomics and metabolomics (p >> n) but loses interpretability
- LOCO evaluation with N=21 means each fold tests on ~5 samples from one crew
- Cross-tissue conservation (H1) shows moderate signal: some genes respond systemically to spaceflight across both blood and skin
- LightGBM best for H1 (0.284), suggesting nonlinear relationships between PBMC cell type effects and skin DE
