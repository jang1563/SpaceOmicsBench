# SpaceOmicsBench

A multi-omics AI benchmark for spaceflight biomedical data, featuring **21 ML tasks** across **9 modalities** and a **100-question LLM evaluation** framework. Data from the SpaceX Inspiration4 (I4) civilian astronaut mission, NASA Twins Study, and JAXA Cell-Free Epigenome (CFE) study.

All benchmark tables are derived from OSDR public releases and/or published supplementary tables. Any human sequence-level or restricted files are excluded from the open track; a controlled-access track may require an approved OSDR Data Access Request (DAR).

## Overview

| | |
|---|---|
| **ML Tasks** | 21 tasks (19 main + 2 supplementary) |
| **LLM Evaluation** | 100 questions, 5-dimension Claude-as-judge scoring |
| **Modalities** | Clinical, cfRNA, Proteomics, Metabolomics, Spatial Transcriptomics, Microbiome, Multi-modal, Cross-tissue, Cross-mission |
| **Difficulty Tiers** | Calibration (1) / Standard (5) / Advanced (9) / Frontier (6) |
| **Missions** | Inspiration4 (4 crew, 3 days LEO), NASA Twins (340 days ISS), JAXA CFE (6 astronauts, >120 days ISS) |
| **Evaluation** | Leave-One-Crew-Out, Leave-One-Timepoint-Out, 80/20 feature splits (5 reps) |
| **Baselines** | Random, Majority, LogReg, RF, MLP, XGBoost, LightGBM |

## Quick Start

### 1. Setup

```bash
git clone https://github.com/jang1563/SpaceOmicsBench.git
cd SpaceOmicsBench

# Create conda environment
conda create -n spaceomics python=3.11 -y
conda activate spaceomics
pip install -r requirements.txt

# Optional: LLM evaluation dependencies
pip install -r requirements-llm.txt
```

### 2. Run Baselines

```bash
python baselines/run_baselines.py
```

This runs all 7 baseline models on all 21 tasks and outputs:
- Per-task metrics (primary + secondary)
- B1 feature ablation study
- Normalized composite scores
- Results saved to `baselines/baseline_results.json`

### 3. Evaluate Your Model

```bash
# Dry run — verify all tasks and splits load correctly
python evaluation/eval_harness.py --dry-run

# Evaluate predictions
python evaluation/eval_harness.py --task all --predictions your_results/ --output results.json
```

Prediction file format: one JSON per task in the predictions directory. See `evaluation/eval_harness.py` header for format details.

### 4. Interactive Demo

Open `demo.html` in a browser for an interactive visualization of benchmark results, task descriptions, and baseline comparisons.

## Task Catalog

### Category A: Clinical Biomarkers

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| A1 | Flight Phase Classification (Blood Panel) | 3-class | Standard | 28 | macro_f1 | LOCO (4-fold) |
| A2 | Flight Phase Classification (Immune Markers) | 3-class | Standard | 28 | macro_f1 | LOCO (4-fold) |

### Category B: Cell-Free RNA

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| B1 | Spaceflight-Responsive Gene Ranking | Binary | Advanced | 26,845 | AUPRC | Feature 80/20 (5-rep) |
| B2 | Coregulated Gene Cluster Prediction | Multilabel (16) | Advanced | 466 | micro_f1 | Feature 80/20 (5-rep) |

### Category C: Proteomics

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| C1 | Proteomics Phase Classification | 3-class | Standard | 21 | macro_f1 | LOCO (4-fold) |
| C2 | Cross-Biofluid Protein DE Concordance | Binary | Frontier | 380 | AUROC | Feature 80/20 (5-rep) |

### Category D: Metabolomics

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| D1 | Metabolite Spaceflight Response Prediction | Binary | Advanced | 433 | AUROC | Feature 80/20 (5-rep) |

### Category E: Spatial Transcriptomics

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| E1 | Cross-Layer DE (outer_epidermis) | Binary | Advanced | 18,677 | AUPRC | Feature 80/20 (5-rep) |
| E4 | Cross-Layer DE (epidermis) | Binary | Advanced | 18,677 | AUPRC | Feature 80/20 (5-rep) |
| E2* | Cross-Layer DE (inner_epidermis) | Binary | Frontier | 18,677 | AUPRC | Feature 80/20 (5-rep) |
| E3* | Cross-Layer DE (outer_dermis) | Binary | Frontier | 18,677 | AUPRC | Feature 80/20 (5-rep) |

*\* Supplementary: extreme class imbalance (11-18 positives out of 18,677), metric instability expected.*

### Category F: Microbiome

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| F1 | Body Site Classification (Taxonomy) | 10-class | Standard | 275 | macro_f1 | LOCO (4-fold) |
| F2 | Flight Phase Detection (Taxonomy) | 4-class | Frontier | 275 | macro_f1 | LOCO (4-fold) |
| F3 | Human vs Environmental Classification | Binary | Calibration | 314 | AUROC | LOTO (7-fold) |
| F4 | Body Site Classification (Pathways) | 10-class | Standard | 275 | macro_f1 | LOCO (4-fold) |
| F5 | Flight Phase Detection (Pathways) | 4-class | Frontier | 275 | macro_f1 | LOCO (4-fold) |

### Category G: Multi-Modal Integration

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| G1 | Multi-Modal Phase Classification | 3-class | Advanced | 21 | macro_f1 | LOCO (4-fold) |

*Fuses clinical biomarkers + PCA(proteomics) + PCA(metabolomics).*

### Category H: Cross-Tissue

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| H1 | Cross-Tissue Gene Conservation | Binary | Advanced | 731 | AUPRC | Feature 80/20 (5-rep) |

### Category I: Cross-Mission (NASA Twins × I4)

| ID | Task | Type | Tier | N | Metric | Split |
|----|------|------|------|---|--------|-------|
| I1 | Hemoglobin Gene DE Prediction | Binary | Frontier | 26,845 | AUPRC | Feature 80/20 (5-rep) |
| I2 | Cross-Mission Pathway Conservation | Binary | Advanced | 452 | AUROC | Feature 80/20 (5-rep) |
| I3 | Cross-Mission Gene DE Conservation | Binary | Advanced | 15,540 | AUPRC | Feature 80/20 (5-rep) |

*Uses NASA Twins Study (340-day ISS, N=1 astronaut with twin control) to predict Inspiration4 patterns. I1 tests whether Twins fold-changes identify hemoglobin pathway genes. I2/I3 test cross-mission conservation at pathway and gene levels.*

## Difficulty Tiers

| Tier | Description | Baseline Behavior |
|------|-------------|-------------------|
| **Calibration** | Easy validation tasks | Best baseline AUROC > 0.8 |
| **Standard** | Learnable with standard methods | Best baseline clearly above random |
| **Advanced** | Challenging, meaningful signal exists | Some baselines above random |
| **Frontier** | At the boundary of learnability | Near-random baseline performance |

## Evaluation

### Metrics

- **Classification (multi-class)**: macro F1, accuracy, per-class F1
- **Binary classification**: AUROC, AUPRC, F1
- **Multilabel**: micro F1, macro F1, Hamming loss
- **Direction concordance**: for cross-biofluid tasks (C2)

### Normalized Composite Score

Individual task scores are normalized against the random baseline to handle metric scale differences:

```
normalized_score = (model_score - random_baseline) / (1.0 - random_baseline)
```

Category scores are averaged within each category, then the composite is the mean across all categories:

```
composite = mean(category_averages)
```

### Split Strategies

| Strategy | Used By | Description |
|----------|---------|-------------|
| LOCO | A1, A2, C1, F1-F5, G1 | Leave-One-Crew-Out (4 folds for I4 crew) |
| LOTO | F3 | Leave-One-Timepoint-Out (7 folds) |
| Feature 80/20 | B1, B2, C2, D1, E1-E4, H1, I1-I3 | Stratified 80/20 (5 repetitions, seed=42) |

## Baseline Results

Tables below are generated from `baselines/baseline_results.json`.
To refresh them after re-running baselines:
`python scripts/generate_readme_tables.py --update-readme README.md`

<!-- BEGIN BASELINE_TABLE -->

| Task | Tier | Metric | Random | Majority | LogReg | RF | MLP | XGBoost | LightGBM |
|------|------|--------|--------|----------|--------|----|-----|---------|----------|
| A1 | Standard | macro_f1 | 0.214 | 0.200 | **0.546** | 0.294 | 0.310 | 0.332 | 0.200 |
| A2 | Standard | macro_f1 | 0.214 | 0.200 | **0.493** | 0.374 | 0.331 | 0.353 | 0.200 |
| B1 | Advanced | AUPRC | 0.020 | 0.017 | 0.533 | 0.885 | 0.854 | 0.912 | **0.922** |
| B2 | Advanced | micro_f1 | 0.083 | 0.000 | **0.154** | 0.131 | 0.000 | — | — |
| C1 | Standard | macro_f1 | 0.170 | 0.228 | 0.512 | 0.464 | **0.517** | 0.355 | 0.228 |
| C2 | Frontier | AUROC | 0.529 | 0.500 | 0.500 | 0.555 | 0.524 | 0.533 | **0.565** |
| D1 | Advanced | AUROC | 0.481 | 0.500 | 0.561 | **0.676** | 0.557 | 0.617 | 0.638 |
| E1 | Advanced | AUPRC | 0.008 | 0.002 | **0.017** | 0.015 | 0.003 | 0.010 | 0.005 |
| E4 | Advanced | AUPRC | 0.003 | 0.002 | **0.023** | 0.002 | 0.003 | 0.006 | 0.009 |
| F1 | Standard | macro_f1 | 0.112 | 0.018 | 0.147 | 0.199 | 0.108 | 0.193 | **0.200** |
| F2 | Frontier | macro_f1 | 0.205 | 0.111 | 0.236 | 0.238 | 0.204 | 0.263 | **0.280** |
| F3 | Calibration | AUROC | 0.402 | 0.500 | 0.574 | **0.841** | 0.320 | 0.838 | 0.838 |
| F4 | Standard | macro_f1 | 0.112 | 0.018 | **0.163** | 0.151 | 0.096 | 0.134 | 0.160 |
| F5 | Frontier | macro_f1 | 0.205 | 0.111 | 0.240 | 0.254 | 0.229 | 0.300 | **0.304** |
| G1 | Advanced | macro_f1 | 0.253 | 0.228 | **0.517** | 0.254 | 0.285 | 0.328 | 0.228 |
| H1 | Advanced | AUPRC | 0.060 | 0.048 | 0.176 | 0.266 | 0.062 | 0.213 | **0.284** |
| I1 | Frontier | AUPRC | 0.003 | 0.002 | 0.003 | 0.005 | 0.003 | 0.005 | **0.006** |
| I2 | Advanced | AUROC | 0.504 | 0.500 | 0.586 | 0.706 | 0.580 | 0.716 | **0.735** |
| I3 | Advanced | AUPRC | 0.059 | 0.052 | **0.090** | 0.081 | 0.090 | 0.081 | 0.086 |

<!-- END BASELINE_TABLE -->

**Bold** = best performing baseline per task. — = not applicable (multilabel task).

### Normalized Composite Scores

<!-- BEGIN COMPOSITE_TABLE -->

| Model | Composite | Best Categories |
|-------|-----------|-----------------|
| RF | **0.258** | B_cfrna (0.882), F_source (0.735), D_metabolomics (0.375) |
| XGBoost | 0.250 | B_cfrna (0.910), F_source (0.728), D_metabolomics (0.262) |
| LightGBM | 0.238 | B_cfrna (0.921), F_source (0.730), D_metabolomics (0.302) |
| LogReg | 0.201 | B_cfrna (0.523), A_clinical (0.389), G_multimodal (0.353) |
| MLP | 0.133 | B_cfrna (0.851), C_proteomics (0.209), D_metabolomics (0.147) |

<!-- END COMPOSITE_TABLE -->

### B1 Feature Ablation

The B1 task includes effect-size features (fold-changes, differences) alongside distribution features. Ablation reveals:

| Variant | Features | LogReg | RF | MLP | XGBoost | LightGBM |
|---------|----------|--------|----|-----|---------|----------|
| B1 (all) | All 29 features | 0.533 | 0.885 | 0.854 | 0.912 | **0.922** |
| B1 (effect-only) | Only fold-change/diff | 0.248 | 0.813 | 0.741 | 0.780 | 0.801 |
| B1 (no-effect) | Exclude fold-change/diff | 0.527 | 0.863 | 0.847 | **0.899** | 0.884 |

Distribution-based features (means, ranges, IQRs) carry most of the predictive signal, confirming the task tests genuine biological pattern recognition rather than simple effect-size thresholding. Gradient boosting methods (XGBoost, LightGBM) achieve the highest B1 scores, with LightGBM reaching AUPRC=0.922.

## LLM Evaluation

SpaceOmicsBench includes a question-based evaluation framework for assessing LLM understanding of spaceflight multi-omics data.

### Question Bank

100 questions across 9 modalities and 4 difficulty levels:

| Modality | Easy | Medium | Hard | Expert | Total |
|----------|------|--------|------|--------|-------|
| Clinical | 3 | 3 | 3 | 1 | 10 |
| Transcriptomics | 2 | 3 | 3 | 2 | 10 |
| Proteomics | 2 | 3 | 3 | 2 | 10 |
| Metabolomics | 2 | 3 | 3 | 2 | 10 |
| Spatial | 1 | 4 | 3 | 2 | 10 |
| Microbiome | 2 | 4 | 3 | 1 | 10 |
| Cross-Mission | 2 | 5 | 6 | 5 | 18 |
| Multi-Omics | 1 | 3 | 5 | 3 | 12 |
| Methods | 2 | 4 | 2 | 2 | 10 |
| **Total** | **17** | **32** | **31** | **20** | **100** |

Question types: factual, interpretation, reasoning, counterfactual, experimental design, cross-mission comparison.

### 5-Dimension Scoring (Claude-as-Judge)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Factual Accuracy | 0.25 | Are stated facts correct? |
| Reasoning Quality | 0.25 | Is scientific logic sound? |
| Completeness | 0.20 | Are key factors addressed? |
| Uncertainty Calibration | 0.15 | Appropriate hedging for small-N data? |
| Domain Integration | 0.15 | Cross-omics/mission knowledge? |

### Running LLM Evaluation

```bash
# Run evaluation (Claude)
export ANTHROPIC_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model claude-sonnet-4-20250514 --sample 10

# Run evaluation (OpenAI)
export OPENAI_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model gpt-4o --full

# Score responses with Claude-as-judge
python evaluation/llm/score_responses.py results/eval_*.json

# Score with GPT-4o as judge (cross-judge verification)
python evaluation/llm/score_responses.py results/eval_*.json --judge-backend openai --judge-model gpt-4o

# Generate comparison report
python evaluation/llm/generate_report.py results/scored_*.json --compare
```

**LLM reproducibility notes**:
- Default generation settings: `temperature=0.3`, `max_tokens=2000`
- No fixed random seed is set by default; expect small variability across runs
- For model comparisons, report the exact model name, temperature, and max_tokens

### LLM Evaluation Results

**2×2 Cross-Judge Verification** — Both Claude and GPT-4o as judges, scoring responses from both models:

| | Claude-as-Judge | GPT-4o-as-Judge |
|---|---|---|
| **Claude Sonnet 4** | **4.55** / 5.00 | 4.76 / 5.00 |
| **GPT-4o** | 3.64 / 5.00 | 4.36 / 5.00 |

Both judges consistently rank Claude Sonnet above GPT-4o. GPT-4o-as-judge is uniformly more lenient (+0.51 same-vendor differential) but does not reverse the ranking.

**Claude Sonnet 4 — Dimension Breakdown** (Claude-as-Judge):

| Dimension | Weight | Score |
|-----------|--------|-------|
| Factual Accuracy | 0.25 | 4.60 |
| Reasoning Quality | 0.25 | 4.83 |
| Completeness | 0.20 | 4.75 |
| Uncertainty Calibration | 0.15 | 3.90 |
| Domain Integration | 0.15 | 4.38 |

**GPT-4o** — largest gaps vs Claude in Completeness (−1.18) and Domain Integration (−1.21), reflecting less thorough coverage of cross-omics connections and key reasoning points.

## Directory Structure

```
SpaceOmicsBench/
├── README.md
├── demo.html                        # Interactive benchmark visualization
├── data/
│   └── processed/                   # Benchmark data (CSV)
│       ├── clinical_cbc.csv         # Clinical CBC features
│       ├── cfrna_3group_de_noleak.csv # cfRNA gene features (no leakage)
│       ├── cross_mission_*.csv      # I-series cross-mission data
│       └── ...
├── tasks/                           # Task definitions (JSON)
│   ├── A1.json ... H1.json         # 19 main + 2 supplementary
│   └── I1.json, I2.json, I3.json   # Cross-mission tasks
├── splits/                          # Train/test split indices (JSON)
│   ├── loco_clinical.json
│   ├── feature_split_B1.json
│   ├── feature_split_I1.json
│   └── ...
├── evaluation/
│   ├── eval_harness.py              # ML evaluation harness
│   ├── metrics.py                   # Metric implementations
│   └── llm/                         # LLM evaluation framework
│       ├── question_bank.json       # 100 questions
│       ├── annotation_schema.json   # 5-dimension scoring schema
│       ├── data_context/            # 12 markdown context files
│       ├── run_llm_evaluation.py    # Run LLM on questions
│       ├── score_responses.py       # Claude-as-judge scoring
│       └── generate_report.py       # Report generation
├── baselines/
│   ├── run_baselines.py             # Baseline runner
│   └── baseline_results.json        # Precomputed results
├── scripts/                         # Data preprocessing scripts
│   └── preprocess_cross_mission.py  # I-series data preprocessing
└── docs/                            # Additional documentation
```

## Data Provenance

All data originates from:

1. **NASA Open Science Data Repository (OSDR)** — GeneLab-processed omics files from OSD-569 to OSD-687 (Inspiration4) and OSD-530 (JAXA CFE)
2. **Published supplementary data** — Processed results from peer-reviewed publications on the Inspiration4 and JAXA missions

See `docs/CITATIONS.bib` for the complete list of source publications.

## Data Redistribution & Access

This repository includes only **publicly shareable processed/summary tables**. Raw sequencing data and controlled-access human data are **not redistributed**. For any controlled-access material (e.g., human sequence-level data), users must obtain access directly via the original source (e.g., OSDR DAR, dbGaP/LSDA).

## Results Policy

The `results/` directory is intended for example outputs and local experiments. If you publish benchmark results, treat `baselines/baseline_results.json` as the canonical baseline reference and provide your own model results separately.

## Provenance Table

For a consolidated source/track/license view, see `docs/PROVENANCE.md`.

## Adding a New Model

1. Read task definitions from `tasks/` to understand input/output specifications
2. Load data from `data/processed/` and splits from `splits/`
3. Generate predictions in the required JSON format
4. Run evaluation: `python evaluation/eval_harness.py --task all --predictions your_results/`

Each task JSON specifies:
- `data_files`: which CSV(s) to load
- `input_spec`: feature description and count
- `output_spec`: target type, classes, and class distribution
- `evaluation`: primary and secondary metrics
- `split`: which split file to use

## Task Schema Validation

Task JSON files are expected to follow the schema in `docs/task_schema.json`. This can be used for local validation or CI checks if you extend the benchmark.

## License

- **Benchmark code**: MIT License
- **Data**: Subject to original source licenses and terms (OSDR terms/Science Information Policy; publisher supplementary licenses). The open track includes only publicly accessible processed/summary tables; controlled-access data is excluded unless explicitly obtained via approved DAR.
