# SpaceOmicsBench

[![CI](https://github.com/jang1563/SpaceOmicsBench/actions/workflows/ci.yml/badge.svg)](https://github.com/jang1563/SpaceOmicsBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A multi-omics AI benchmark for spaceflight biomedical data, featuring **21 ML tasks** across **9 modalities** and a **100-question LLM evaluation** framework. Data from the SpaceX Inspiration4 (I4) civilian astronaut mission, NASA Twins Study, and JAXA Cell-Free Epigenome (CFE) study.

All benchmark tables are derived from OSDR public releases and/or published supplementary tables. Any human sequence-level or restricted files are excluded from the open track; a controlled-access track may require an approved OSDR Data Access Request (DAR).

## Overview

| | |
|---|---|
| **ML Tasks** | 21 tasks (19 main + 2 supplementary) |
| **LLM Evaluation** | 100 questions, 5-dimension Claude-as-judge scoring, 9 models evaluated |
| **Modalities** | Clinical, cfRNA, Proteomics, Metabolomics, Spatial Transcriptomics, Microbiome, Multi-modal, Cross-tissue, Cross-mission |
| **Difficulty Tiers** | Calibration (1) / Standard (5) / Advanced (9) / Frontier (6) |
| **Missions** | Inspiration4 (4 crew, 3 days LEO), NASA Twins (340 days ISS), JAXA CFE (6 astronauts, >120 days ISS) |
| **Evaluation** | Leave-One-Crew-Out, Leave-One-Timepoint-Out, 80/20 feature splits (5 reps) |
| **ML Baselines** | Random, Majority, LogReg, RF, MLP, XGBoost, LightGBM |
| **LLM Evaluated** | Claude Sonnet 4.6, Haiku 4.5, Sonnet 4, GPT-4o, GPT-4o Mini, DeepSeek-V3, Gemini 2.5 Flash, Llama-3.3-70B |

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
# ── Proprietary models ────────────────────────────────────────────────────
# Claude
export ANTHROPIC_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model claude-sonnet-4-6 --sample 10

# OpenAI
export OPENAI_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model gpt-4o --full

# ── Open-source via API (OpenAI-compatible) ───────────────────────────────
# Groq (Llama 3.3 70B — free tier, fast)
export GROQ_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model llama-3.3-70b-versatile \
    --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --full

# Together.ai (Llama 3.3 70B — serverless, no dedicated endpoint needed)
export TOGETHER_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --base-url https://api.together.xyz/v1 --api-key-env TOGETHER_API_KEY --full

# DeepSeek-V3
export DEEPSEEK_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model deepseek-chat \
    --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY --full

# Gemini 2.5 Flash (via OpenAI-compatible endpoint; billing must be active)
export GEMINI_API_KEY="your-key"
python evaluation/llm/run_llm_evaluation.py --model models/gemini-2.5-flash \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --api-key-env GEMINI_API_KEY --full

# ── Open-source via Ollama (fully local, Apple Silicon supported) ─────────
# Pull model first: ollama pull llama3.3
python evaluation/llm/run_llm_evaluation.py --model llama3.3:70b \
    --base-url http://localhost:11434/v1 --full

# ── HuggingFace local (Apple Silicon MPS auto-detected) ──────────────────
python evaluation/llm/run_llm_evaluation.py \
    --model meta-llama/Llama-3.3-70B-Instruct --sample 10

# ── Scoring (Claude / GPT-4o / open-source as judge) ─────────────────────
# Claude as judge (default)
python evaluation/llm/score_responses.py results/eval_*.json

# GPT-4o as judge
python evaluation/llm/score_responses.py results/eval_*.json \
    --judge-backend openai --judge-model gpt-4o

# Open-source as judge via Groq
python evaluation/llm/score_responses.py results/eval_*.json \
    --judge-backend compatible --judge-model llama-3.3-70b-versatile \
    --judge-base-url https://api.groq.com/openai/v1 \
    --judge-api-key-env GROQ_API_KEY

# ── Generate comparison report ────────────────────────────────────────────
python evaluation/llm/generate_report.py results/scored_*.json --compare
```

**LLM reproducibility notes**:
- Default generation settings: `temperature=0.3`, `max_tokens=2000`
- No fixed random seed is set by default; expect small variability across runs
- For model comparisons, report the exact model name, temperature, and max_tokens

**Reproducing the results table** — commands used to generate the numbers in the Results section below:

```bash
# Step 1: Run full evaluation (100 questions)
python evaluation/llm/run_llm_evaluation.py --model claude-sonnet-4-6 --full
python evaluation/llm/run_llm_evaluation.py --model gpt-4o --full

# Step 2: Score with Sonnet 4.6 as judge (primary judge for the table)
python evaluation/llm/score_responses.py results/eval_claude-sonnet-4-6_*.json \
    --judge-model claude-sonnet-4-6 --judge-backend anthropic \
    --output results/scored_eval_claude-sonnet-4-6_judged-by-sonnet46.json

python evaluation/llm/score_responses.py results/eval_claude-sonnet-4-20250514_*.json \
    --judge-model claude-sonnet-4-6 --judge-backend anthropic \
    --output results/scored_eval_claude-sonnet-4_judged-by-sonnet46.json

python evaluation/llm/score_responses.py results/eval_gpt-4o_*.json \
    --judge-model claude-sonnet-4-6 --judge-backend anthropic \
    --output results/scored_eval_gpt-4o_judged-by-sonnet46.json

# Step 3: Score with Sonnet 4 as judge (for cross-judge column)
python evaluation/llm/score_responses.py results/eval_claude-sonnet-4-6_*.json \
    --judge-model claude-sonnet-4-20250514 --judge-backend anthropic \
    --output results/scored_eval_claude-sonnet-4-6_judged-by-sonnet4.json

# Step 4: Score with GPT-4o as judge (for cross-judge column)
python evaluation/llm/score_responses.py results/eval_claude-sonnet-4-20250514_*.json \
    --judge-backend openai --judge-model gpt-4o \
    --output results/scored_eval_claude-sonnet-4_judged-by-gpt4o.json

python evaluation/llm/score_responses.py results/eval_gpt-4o_*.json \
    --judge-backend openai --judge-model gpt-4o \
    --output results/scored_eval_gpt-4o_judged-by-gpt4o.json
```

A summary of the scored outputs used to generate the table is in `docs/samples/llm_eval_summary.json`.

**Supported backends summary:**

| Backend flag | Covers | API key env |
|---|---|---|
| `anthropic` (default) | Claude models | `ANTHROPIC_API_KEY` |
| `openai` | GPT-4o, o1/o3 | `OPENAI_API_KEY` |
| `compatible` + `--base-url` | Groq, Together, DeepSeek, Mistral, OpenRouter, Ollama | via `--api-key-env` |
| `huggingface` | Local models (CUDA / Apple MPS / CPU) | — |

**For publication-quality benchmarking**, aim to evaluate at minimum:
- 2 proprietary frontier models (Claude + GPT-4o)
- 2–3 open-source flagship models (Llama 3.3 70B + DeepSeek R1 + Qwen 2.5 72B)
- 1 biomedical-specialized model (BioMistral, Meditron, etc.)
- Cross-judge verification with ≥ 2 judges (inter-rater reliability)

### LLM Evaluation Results

**9-Model Ranking** (Judge: Claude Sonnet 4.6, 100 questions, 1–5 scale):

🔒 = proprietary API &nbsp; 🔓 = open-weights

| Rank | Model | Score | Easy | Med | Hard | Expert | Factual | Reasoning | Complete | Uncert | Domain |
|------|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 🔒 **Claude Sonnet 4.6** | **4.60** | 4.58 | 4.49 | 4.66 | 4.68 | 4.59 | 4.96 | 4.73 | 4.09 | 4.33 |
| 2 | 🔒 **Claude Haiku 4.5** | **4.39** | 4.35 | 4.45 | 4.38 | 4.37 | 4.35 | 4.84 | 4.53 | 3.82 | 4.12 |
| 3 | 🔓 **DeepSeek-V3** | **4.31** | 4.11 | 4.22 | 4.42 | 4.46 | 4.34 | 4.72 | 4.37 | 3.72 | 4.09 |
| 4 | 🔒 **Claude Sonnet 4** | **4.03** | 4.05 | 4.05 | 4.00 | 4.01 | 4.26 | 4.47 | 4.07 | 3.14 | 3.74 |
| 5 | 🔒 **Gemini 2.5 Flash** | **3.43** | 3.68 | 3.91 | 3.25 | 2.75 | 4.13 | 3.75 | 3.22 | 2.62 | 2.83 |
| 6 | 🔓 **Llama-3.3-70B** (Together) | **3.30** | 3.57 | 3.33 | 3.29 | 3.06 | 3.98 | 3.49 | 3.22 | 2.65 | 2.63 |
| 7 | 🔒 **GPT-4o** | **3.30** | 3.24 | 3.40 | 3.28 | 3.22 | 3.98 | 3.60 | 3.13 | 2.57 | 2.63 |
| 8 | 🔒 **GPT-4o Mini** | **3.30** | 3.37 | 3.40 | 3.29 | 3.10 | 3.91 | 3.51 | 3.19 | 2.75 | 2.63 |
| 9 | 🔓 **Llama-3.3-70B** (Groq) | **3.29** | 3.48 | 3.35 | 3.27 | 3.06 | 4.02 | 3.50 | 3.18 | 2.58 | 2.58 |

Key findings:
- **Claude models dominate the top tier**; Haiku 4.5 notably outperforms Sonnet 4 (+0.36) despite being a smaller model
- **DeepSeek-V3 (#3, 4.31)** is the strongest open-weights model, surpassing Claude Sonnet 4 and all GPT/Gemini variants — particularly strong on Hard and Expert questions
- **Gemini 2.5 Flash** excels on Easy/Medium (+3.68–3.91) but collapses on Expert questions (2.75), suggesting surface-level competence without deep spaceflight domain knowledge
- **GPT-4o ≈ GPT-4o Mini ≈ Llama-3.3-70B** (all ~3.30) — no meaningful scaling advantage in this tier for this specialized domain
- **Uncertainty Calibration** is the weakest dimension across all models; small-N spaceflight data requires careful hedging that all models underperform on
- **Novel insight flags**: DeepSeek-V3 (62 flagged) and Claude models (44–92 flagged) generate novel cross-modal reasoning; GPT/Llama variants generate none

**Cross-Judge Verification** — Sonnet 4, Sonnet 4.6, and GPT-4o were additionally scored by Sonnet 4 and GPT-4o judges for bias analysis:

| Respondent | Sonnet 4 Judge | Sonnet 4.6 Judge | GPT-4o Judge |
|-----------|:-:|:-:|:-:|
| **Claude Sonnet 4.6** | 4.73 | **4.60** | — |
| **Claude Sonnet 4** | 4.55 | 4.03 | 4.76 |
| **GPT-4o** | 3.64 | 3.30 | 4.36 |

Sonnet 4.6 is the strictest judge (scores 0.3–0.5 lower); GPT-4o as judge inflates scores by ~0.2–0.7 but does not change ranking order.

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
│   ├── signature_query.py           # Compare new DE data vs. benchmark signatures
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
├── scripts/                         # Preprocessing and utility scripts
│   ├── preprocess_cross_mission.py  # I-series data preprocessing
│   ├── generate_readme_tables.py    # Regenerate baseline tables in README
│   ├── validate_tasks.py            # Validate all task JSON files against schema
│   └── generate_tasks_and_splits.py # [LEGACY] original task/split generator
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

## Signature Query: Compare New Data Against Benchmark

If you have DE results from a new spaceflight mission, compare them against SpaceOmicsBench
reference signatures to identify biological overlap with known spaceflight responses.

**Input:** any CSV with `gene`, `log2FC` (or `logFC`/`log2FoldChange`), and `padj` columns.

```bash
python evaluation/signature_query.py --input my_de_results.csv
```

**Options:**
```
--padj-threshold FLOAT   Adjusted p-value cutoff for DE (default: 0.05)
--fc-threshold   FLOAT   Minimum |log2FC| for DE (default: 0, disabled)
--signatures     IDs...  Subset of signatures to query (default: all 8)
--output-dir     DIR     Output directory for JSON + Markdown report (default: results/)
```

**Available reference signatures:**

| Signature ID | Description | N |
|---|---|---|
| `I4_cfRNA_DRR` | Inspiration4 cfRNA spaceflight-responsive genes (JAXA IHSP) | 466 |
| `I4_Plasma_Proteomics` | Inspiration4 plasma DE proteins | 57 |
| `I4_PBMC_CD4T` | Inspiration4 CD4+ T cell DE genes | 736 |
| `I4_PBMC_CD8T` | Inspiration4 CD8+ T cell DE genes | 661 |
| `I4_PBMC_CD14Mono` | Inspiration4 CD14+ Monocyte DE genes | 709 |
| `GeneLab_Mouse` | GeneLab rodent spaceflight DE genes (conserved with I4) | 134 |
| `JAXA_cfRNA` | JAXA cfRNA DE genes | 36 |
| `CrossMission_Conserved` | Cross-mission conserved spaceflight genes | 814 |

**Metrics computed:** Jaccard index, overlap coefficient, hypergeometric enrichment (FDR-corrected),
direction concordance, and Spearman correlation of log2FC values.

**Note:** This is an exploratory overlap tool. I4 had n=4 crew; reference signatures reflect
one specific mission cohort. See `docs/extension_plan.md` for planned ingestion pipeline.

## Task Schema Validation

Task JSON files are expected to follow the schema in `docs/task_schema.json`. This can be used for local validation or CI checks if you extend the benchmark.

## License

- **Benchmark code**: MIT License
- **Data**: Subject to original source licenses and terms (OSDR terms/Science Information Policy; publisher supplementary licenses). The open track includes only publicly accessible processed/summary tables; controlled-access data is excluded unless explicitly obtained via approved DAR.
