# Cell-Free RNA Transcriptomics

## I4 cfRNA Dataset
- **Genes measured**: 26,845 across 4 crew, 7 timepoints
- **Analysis**: 3-group differential expression (ANOVA + pairwise edge tests)
- **Groups**: pre-flight vs in-flight vs post-flight
- **DRR genes**: 466 differentially-regulated response genes identified (1.7% of total)

## Feature Set (29 features per gene)
- **Effect-size features**: fold-changes and mean differences for each pairwise comparison (pre vs flight, pre vs post, flight vs post)
- **Distribution features**: group means (normalized, raw, transformed), experiment-level range, IQR, fold-change, difference across all samples
- **No p-values included** (removed to prevent leakage)

## JAXA CFE Study
- 6 astronauts, >120 days ISS
- Cell-free RNA epigenome analysis
- Provides independent validation context for spaceflight transcriptomic changes

## Benchmark Tasks
- **B1**: Spaceflight-responsive gene ranking (binary: DRR vs non-DRR)
  - N=26,845, positives=466 (1.7%), metric=AUPRC, Random baseline: 0.020
  - Feature 80/20 split (5 reps), seed=42
  - Best baseline: LightGBM=0.922 | XGBoost=0.911 | RF=0.884 | MLP=0.854 | LogReg=0.533
- **B2**: Coregulated gene cluster prediction (multilabel, 16 clusters)
  - N=466, metric=micro_f1
  - Best: LogReg=0.154 | RF=0.131 | MLP=0.000, Random: 0.083

## B1 Feature Ablation Study
| Variant | LogReg | RF | MLP | XGBoost | LightGBM |
|---------|--------|----|-----|---------|----------|
| All 29 features | 0.533 | 0.884 | 0.854 | 0.911 | **0.922** |
| Effect-only (fold-change/diff) | 0.248 | 0.813 | 0.741 | 0.780 | 0.801 |
| No-effect (distribution only) | 0.527 | 0.863 | 0.847 | **0.899** | 0.884 |

**Key finding**: Distribution-based features carry most predictive signal. LightGBM achieves overall best (0.922) but XGBoost has best no-effect score (0.899), suggesting XGBoost better exploits distribution-only features while LightGBM benefits more from the combined feature set.
