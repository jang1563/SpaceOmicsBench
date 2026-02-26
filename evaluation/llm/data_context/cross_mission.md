# Cross-Mission Comparison Data

## Missions Compared
| Property | Inspiration4 (I4) | NASA Twins Study |
|----------|-------------------|-----------------|
| Duration | 3 days | 340 days |
| Orbit | LEO (~585 km) | ISS (~408 km) |
| Crew | 4 civilians | 1 astronaut (twin control) |
| Year | 2021 | 2015-2016 |
| Blood analysis | cfRNA (cell-free), PBMC (single-cell) | Blood cell scRNA-seq (CD4, CD8, CD19, PBMC, LD) |

## NASA Twins Study Details
- **Subjects**: Scott Kelly (space) vs Mark Kelly (ground twin)
- **Experiment types**: Multivariate (PolyA+ and Ribodepleted together), PolyA+, Ribodepleted
- **Cell types**: CD4, CD8, CD19 (B cells), PBMC, LD (lymphocyte-depleted)
- **Contrasts**: 21 different coefficients including In-flight vs Pre-flight, Post-flight vs Pre-flight, In-flight vs Post-flight, GLM models for time effects
- **Total DEG entries**: 115,493 rows across all cell types, contrasts, and experiment types
- **Unique genes tested**: 19,446

## Cross-Mission Pathway Conservation (Task I2)
- **I4 PBMC GSEA**: 452 unique pathways enriched across 9 cell types (CD4_T, CD8_T, other_T, B, NK, CD14_Mono, CD16_Mono, DC, other)
- **Twins GSEA**: 152 unique pathways significant in CD4, CD8, CD19
- **Conserved**: 146 pathways found in both missions (32.3% of I4 pathways)
- **Key conserved pathways**: HALLMARK_OXIDATIVE_PHOSPHORYLATION, HALLMARK_MYC_TARGETS_V1, HALLMARK_UV_RESPONSE_DN
- **Features**: mean NES, std NES, mean ES, mean/min padj, number of cell types, pathway size, direction consistency
- **N=452**, metric=AUROC, Random: 0.504
- **Best**: LightGBM=0.735 | XGBoost=0.716 | RF=0.706 | LogReg=0.586 | MLP=0.580

## Cross-Mission Gene DE Conservation (Task I3)
- **Shared gene universe**: 15,540 genes (intersection of Twins 19,446 and I4 cfRNA 26,845)
- **Conserved DE genes**: 814 (5.2%) — significant in both Twins blood cells and I4 cfRNA
- **Twins features per gene**: mean absolute log2FC, max absolute log2FC, mean base expression, mean lfcSE, mean absolute Wald statistic, number of cell types DE, number of contrasts DE, total DEG entries, direction consistency
- **I4 target**: cfRNA ANOVA FDR < 0.05
- **N=15,540**, metric=AUPRC, Random: 0.059
- **Best**: MLP=0.090 | LogReg=0.090 | LightGBM=0.086 | RF=0.081 | XGBoost=0.081

## Biological Significance
- Despite 100-fold difference in mission duration (3 days vs 340 days), 32.3% pathway overlap is remarkable
- Suggests core spaceflight stress pathways that activate regardless of duration
- Oxidative phosphorylation disruption is a hallmark of mitochondrial stress in microgravity
- MYC targets relate to cell proliferation changes observed in both missions
- The 5.2% gene-level conservation is lower than pathway-level, suggesting pathways capture broader biological themes while individual gene responses are more context-dependent
