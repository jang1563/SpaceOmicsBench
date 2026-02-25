# SpaceOmicsBench

A multi-omics AI benchmark for spaceflight biomedical data, featuring **21 ML tasks** across **9 modalities** and a **60-question LLM evaluation** framework. Data from the SpaceX Inspiration4 (I4) civilian astronaut mission, NASA Twins Study, and JAXA Cell-Free Epigenome (CFE) study.

All data is derived from publicly accessible sources (NASA Open Science Data Repository and published supplementary data).

## Overview

| | |
|---|---|
| **ML Tasks** | 21 tasks (19 main + 2 supplementary) |
| **LLM Evaluation** | 60 questions, 5-dimension Claude-as-judge scoring |
| **Modalities** | Clinical, cfRNA, Proteomics, Metabolomics, Spatial Transcriptomics, Microbiome, Multi-modal, Cross-tissue, Cross-mission |
| **Difficulty Tiers** | Calibration (1) / Standard (5) / Advanced (8) / Frontier (7) |
| **Missions** | Inspiration4 (4 crew, 3 days LEO), NASA Twins (340 days ISS), JAXA CFE (6 astronauts, >120 days ISS) |
| **Evaluation** | Leave-One-Crew-Out, Leave-One-Timepoint-Out, Stratified 80/20 feature splits |
| **Baselines** | Random, Majority, LogReg, RF, MLP |

## Quick Start

### 1. Setup

```bash
git clone https://github.com/jang1563/SpaceOmicsBench.git
cd SpaceOmicsBench

# Create conda environment
conda create -n spaceomics python=3.11 -y
conda activate spaceomics
pip install numpy pandas scikit-learn
```

### 2. Run Baselines

```bash
python baselines/run_baselines.py
```

This runs all 5 baseline models on all 21 tasks and outputs:
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
| Feature 80/20 | B1, B2, C2, D1, E1-E4, H1, I1-I3 | Stratified random 80/20 (5 repetitions, seed=42) |

## Baseline Results

| Task | Tier | Metric | Random | Majority | LogReg | RF | MLP |
|------|------|--------|--------|----------|--------|----|-----|
| A1 | Standard | macro_f1 | 0.214 | 0.200 | **0.546** | 0.326 | 0.310 |
| A2 | Standard | macro_f1 | 0.214 | 0.200 | **0.493** | 0.374 | 0.331 |
| B1 | Advanced | AUPRC | 0.020 | 0.017 | 0.533 | **0.885** | 0.839 |
| B2 | Advanced | micro_f1 | 0.083 | 0.000 | **0.154** | 0.131 | 0.000 |
| C1 | Standard | macro_f1 | 0.170 | 0.228 | **0.597** | 0.413 | 0.474 |
| C2 | Frontier | AUROC | 0.529 | 0.500 | 0.500 | **0.555** | 0.524 |
| D1 | Advanced | AUROC | 0.481 | 0.500 | 0.561 | **0.676** | 0.557 |
| E1 | Advanced | AUPRC | 0.008 | 0.002 | **0.017** | 0.015 | 0.003 |
| E4 | Advanced | AUPRC | 0.003 | 0.002 | **0.023** | 0.002 | 0.003 |
| F1 | Standard | macro_f1 | 0.112 | 0.018 | 0.147 | **0.199** | 0.108 |
| F2 | Frontier | macro_f1 | 0.205 | 0.111 | 0.236 | **0.238** | 0.204 |
| F3 | Calibration | AUROC | 0.402 | 0.500 | 0.574 | **0.841** | 0.320 |
| F4 | Standard | macro_f1 | 0.112 | 0.018 | **0.163** | 0.151 | 0.096 |
| F5 | Frontier | macro_f1 | 0.205 | 0.111 | 0.240 | **0.254** | 0.229 |
| G1 | Advanced | macro_f1 | 0.253 | 0.228 | **0.481** | 0.349 | 0.461 |
| H1 | Advanced | AUPRC | 0.060 | 0.048 | 0.176 | **0.266** | 0.062 |
| I1 | Frontier | AUPRC | 0.003 | 0.002 | 0.003 | **0.005** | 0.002 |
| I2 | Advanced | AUROC | 0.482 | 0.500 | 0.682 | **0.706** | 0.592 |
| I3 | Advanced | AUPRC | 0.050 | 0.052 | **0.090** | 0.072 | 0.056 |

**Bold** = best performing baseline per task.

### Normalized Composite Scores

| Model | Composite | Best Categories |
|-------|-----------|-----------------|
| RF | **0.269** | B_cfrna (0.882), F_source (0.735), D_metabolomics (0.375) |
| LogReg | 0.201 | B_cfrna (0.523), A_clinical (0.389), G_multimodal (0.304) |
| MLP | 0.151 | B_cfrna (0.836), G_multimodal (0.278), C_proteomics (0.183) |

### B1 Feature Ablation

The B1 task includes effect-size features (fold-changes, differences) alongside distribution features. Ablation reveals:

| Variant | Features | LogReg | RF | MLP |
|---------|----------|--------|----|-----|
| B1 (all) | All 29 features | 0.533 | 0.885 | 0.839 |
| B1 (effect-only) | Only fold-change/diff features | 0.248 | 0.813 | 0.756 |
| B1 (no-effect) | Exclude fold-change/diff features | 0.527 | 0.863 | 0.865 |

Distribution-based features (means, ranges, IQRs) carry most of the predictive signal, confirming the task tests genuine biological pattern recognition rather than simple effect-size thresholding.

## LLM Evaluation

SpaceOmicsBench includes a question-based evaluation framework for assessing LLM understanding of spaceflight multi-omics data.

### Question Bank

60 questions across 9 modalities and 4 difficulty levels:

| Modality | Easy | Medium | Hard | Expert | Total |
|----------|------|--------|------|--------|-------|
| Clinical | 2 | 2 | 2 | 1 | 7 |
| Transcriptomics | 2 | 2 | 2 | 1 | 7 |
| Proteomics | 1 | 2 | 2 | 1 | 6 |
| Metabolomics | 1 | 2 | 2 | 1 | 6 |
| Spatial | 0 | 2 | 1 | 1 | 4 |
| Microbiome | 1 | 2 | 1 | 0 | 4 |
| Cross-Mission | 2 | 5 | 5 | 4 | 16 |
| Multi-Omics | 1 | 2 | 3 | 2 | 8 |
| Methods | 1 | 1 | 0 | 0 | 2 |
| **Total** | **11** | **20** | **18** | **11** | **60** |

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

# Generate comparison report
python evaluation/llm/generate_report.py results/scored_*.json --compare
```

## Directory Structure

```
SpaceOmicsBench/
├── README.md
├── demo.html                        # Interactive benchmark visualization
├── data/
│   └── processed/                   # Benchmark data (CSV)
│       ├── gt_clinical_cbc.csv      # Clinical CBC features
│       ├── gt_cfrna_features.csv    # cfRNA gene features
│       ├── cross_mission_*.csv      # I-series cross-mission data
│       └── ...
├── tasks/                           # Task definitions (JSON)
│   ├── A1.json ... H1.json         # 16 main + 2 supplementary
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
│       ├── question_bank.json       # 60 questions
│       ├── annotation_schema.json   # 5-dimension scoring schema
│       ├── data_context/            # 11 markdown context files
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

## License

- **Benchmark code**: MIT License
- **Data**: Subject to original source licenses (NASA OSDR: public domain; published supplementary: CC-BY 4.0)
