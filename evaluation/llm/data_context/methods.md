# SpaceOmicsBench — Benchmark Design and Methodology

## Evaluation Strategies

Three evaluation strategies are used, matched to data structure:

### LOCO (Leave-One-Crew-Out)
- **Folds**: 4 (one per crew member C001-C004)
- **Training set**: ~21-24 samples (3 crew), **Test set**: ~4-7 samples (1 crew)
- **Tasks**: A1, A2, C1, F1, F2, F3 (as LOTO), F4, F5, G1
- **Rationale**: Tests generalization to unseen individuals; stringent but high-variance due to N=4 crews
- **Key limitation**: With only 4 folds and ~5-7 test samples per fold, per-fold performance estimates are highly variable

### LOTO (Leave-One-Timepoint-Out)
- **Folds**: 7 (one per timepoint: L-92, L-44, L-3, R+1, R+45, R+82, R+194)
- **Tasks**: F3 only (human vs environmental microbiome classification)
- **Rationale**: Environmental samples are collected per-timepoint, not per-crew

### Feature 80/20 Split
- **Method**: Stratified random 80/20 train/test split, repeated 5 times (seed=42)
- **Tasks**: B1, B2, C2, D1, E1, E2, E3, E4, H1, I1, I2, I3
- **Rationale**: Used for gene/protein/metabolite-level classification where samples are features (not crew members)
- **N ranges**: 380 (C2) to 26,845 (B1, I1)

## Difficulty Tiers

Tiers reflect best model performance relative to random baseline:

| Tier | Criteria | Count | Tasks |
|------|----------|-------|-------|
| Calibration | Best >> random, expected solvable | 1 | F3 |
| Standard | Best moderately above random | 5 | A1, A2, C1, F1, F4 |
| Advanced | Best meaningfully above random but challenging | 9 | B1, B2, D1, E1, E4, G1, H1, I2, I3 |
| Frontier | Best near random, largely unsolved | 6 | C2, F2, F5, I1, E2*, E3* |

*E2, E3 are supplementary tasks (not counted in main 19).

## Normalized Composite Score

### Formula
For each task: `normalized = max(0, (score - random_baseline) / (1 - random_baseline))`

Floor at 0 prevents negative normalization.

### Category Averaging
Tasks are grouped into **11 categories**, each weighted equally (1/11):

| Category | Tasks | Description |
|----------|-------|-------------|
| A_clinical | A1, A2 | Clinical blood panel classification |
| B_cfrna | B1 | cfRNA gene ranking (B2 excluded — multilabel) |
| C_proteomics | C1, C2 | Proteomics classification + concordance |
| D_metabolomics | D1 | Metabolite response prediction |
| E_spatial | E1, E4 | Spatial transcriptomics cross-layer DE |
| F_bodysite | F1, F4 | Microbiome body site classification |
| F_phase | F2, F5 | Microbiome flight phase detection |
| F_source | F3 | Microbiome human vs environmental |
| G_multimodal | G1 | Multi-modal fusion classification |
| H_crosstissue | H1 | Cross-tissue gene conservation |
| I_crossmission | I1, I2, I3 | Cross-mission (I4 ↔ Twins) tasks |

Within each category, task scores are averaged. Then the 11 category scores are averaged equally.

### Composite Results (5 learned models)
| Model | Composite | Best-on-N-tasks |
|-------|-----------|-----------------|
| RF | 0.258 | 2 |
| XGBoost | 0.250 | 3 |
| LightGBM | 0.238 | 8 |
| LogReg | 0.201 | 5 |
| MLP | 0.133 | 1 |

**Key paradox**: LightGBM wins 8/19 individual tasks but ranks 3rd in composite because it **collapses to majority-baseline on small-N LOCO tasks** (A1=0.200, A2=0.200, C1=0.228, G1=0.228), producing category-level zeros for A_clinical (0.000) and G_multimodal (0.000). RF avoids this by maintaining above-random performance across all categories.

## Per-Model Results (All 19 Main Tasks)

| Task | Metric | N | Random | LogReg | RF | MLP | XGBoost | LightGBM |
|------|--------|---|--------|--------|----|-----|---------|----------|
| A1 | macro_f1 | 28 | 0.214 | **0.546** | 0.294 | 0.310 | 0.332 | 0.200 |
| A2 | macro_f1 | 28 | 0.214 | **0.493** | 0.374 | 0.331 | 0.353 | 0.200 |
| B1 | AUPRC | 26845 | 0.020 | 0.533 | 0.884 | 0.854 | 0.911 | **0.922** |
| C1 | macro_f1 | 21 | 0.170 | 0.512 | 0.464 | **0.517** | 0.355 | 0.228 |
| C2 | AUROC | 380 | 0.529 | 0.500 | 0.555 | 0.524 | 0.533 | **0.565** |
| D1 | AUROC | 433 | 0.481 | 0.561 | **0.676** | 0.557 | 0.617 | 0.638 |
| E1 | AUPRC | 18677 | 0.008 | **0.017** | 0.015 | 0.003 | 0.010 | 0.005 |
| E4 | AUPRC | 18677 | 0.003 | **0.022** | 0.002 | 0.003 | 0.006 | 0.009 |
| F1 | macro_f1 | 275 | 0.112 | 0.147 | 0.199 | 0.108 | 0.193 | **0.200** |
| F2 | macro_f1 | 275 | 0.205 | 0.236 | 0.238 | 0.204 | 0.263 | **0.280** |
| F3 | AUROC | 314 | 0.402 | 0.574 | **0.841** | 0.320 | 0.838 | 0.838 |
| F4 | macro_f1 | 275 | 0.112 | **0.163** | 0.151 | 0.096 | 0.134 | 0.160 |
| F5 | macro_f1 | 275 | 0.205 | 0.240 | 0.254 | 0.229 | 0.300 | **0.304** |
| G1 | macro_f1 | 21 | 0.253 | **0.517** | 0.254 | 0.285 | 0.328 | 0.228 |
| H1 | AUPRC | 731 | 0.060 | 0.176 | 0.266 | 0.062 | 0.213 | **0.284** |
| I1 | AUPRC | 26845 | 0.003 | 0.003 | 0.005 | 0.003 | 0.005 | **0.006** |
| I2 | AUROC | 452 | 0.504 | 0.586 | 0.706 | 0.580 | 0.716 | **0.735** |
| I3 | AUPRC | 15540 | 0.059 | 0.090 | 0.081 | **0.090** | 0.081 | 0.086 |

Bold = best per task. Majority baseline omitted for clarity (generally at or below random).

## Gradient Boosting Small-N Collapse

LightGBM and XGBoost collapse to majority-baseline performance on LOCO tasks with N≤28:

| Task | N (train) | LogReg | RF | XGBoost | LightGBM | Majority |
|------|-----------|--------|----|---------|----------|----------|
| A1 | ~21 | 0.546 | 0.294 | 0.332 | 0.200 | 0.200 |
| A2 | ~21 | 0.493 | 0.374 | 0.353 | 0.200 | 0.200 |
| C1 | ~16 | 0.512 | 0.464 | 0.355 | 0.228 | 0.228 |
| G1 | ~16 | 0.517 | 0.254 | 0.328 | 0.228 | 0.228 |

LightGBM collapses more severely than XGBoost because its leaf-wise (best-first) tree growth strategy is more aggressive and prone to overfitting with very few training samples. XGBoost's level-wise (depth-first) approach is more conservative.

## B1 Feature Ablation (7 Models)

| Variant | LogReg | RF | MLP | XGBoost | LightGBM |
|---------|--------|----|-----|---------|----------|
| All 29 features | 0.533 | 0.884 | 0.854 | 0.911 | **0.922** |
| Effect-only (fold-change/diff) | 0.248 | 0.813 | 0.741 | 0.780 | 0.801 |
| No-effect (distribution only) | 0.527 | 0.863 | 0.847 | **0.899** | 0.884 |

**Key finding**: Distribution features carry most predictive signal. LightGBM achieves overall best (0.922) but XGBoost has best no-effect score (0.899), suggesting XGBoost better exploits distribution-only features while LightGBM benefits more from the combined feature set.

## Per-Category Composite Scores

| Category | LogReg | RF | MLP | XGBoost | LightGBM |
|----------|--------|----|-----|---------|----------|
| A_clinical | 0.389 | 0.152 | 0.135 | 0.163 | **0.000** |
| B_cfrna | 0.523 | 0.882 | 0.851 | 0.910 | **0.921** |
| C_proteomics | 0.206 | 0.205 | 0.209 | 0.117 | 0.074 |
| D_metabolomics | 0.154 | 0.375 | 0.147 | 0.262 | 0.302 |
| E_spatial | 0.014 | 0.004 | 0.000 | 0.002 | 0.003 |
| F_bodysite | 0.048 | 0.071 | 0.000 | 0.058 | 0.076 |
| F_phase | 0.042 | 0.052 | 0.015 | 0.096 | 0.110 |
| F_source | 0.288 | 0.735 | 0.000 | 0.728 | 0.730 |
| G_multimodal | 0.353 | 0.001 | 0.042 | 0.100 | **0.000** |
| H_crosstissue | 0.123 | 0.219 | 0.002 | 0.162 | 0.239 |
| I_crossmission | 0.066 | 0.144 | 0.062 | 0.151 | 0.165 |
