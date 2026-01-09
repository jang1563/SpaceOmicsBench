# SpaceOmicsBench

**A Multi-Modal Benchmark for Evaluating Large Language Models on Spaceflight Biomedical Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Abstract

SpaceOmicsBench is a comprehensive benchmark suite designed to evaluate Large Language Model (LLM) capabilities in interpreting, reasoning about, and integrating multi-omics data from commercial spaceflight missions. The benchmark comprises **115 expert-curated questions** spanning clinical biomarkers, transcriptomics, and metabolomics data from the **Inspiration4**, **Polaris Dawn**, and **Fram2** missions—representing the first systematic evaluation framework for LLMs in space medicine and multi-omics integration.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
  - [Mission Profiles](#mission-profiles)
  - [Data Modalities](#data-modalities)
  - [Key Scientific Findings](#key-scientific-findings)
- [Benchmark Design](#benchmark-design)
  - [Question Taxonomy](#question-taxonomy)
  - [Evaluation Dimensions](#evaluation-dimensions)
- [Experimental Results](#experimental-results)
  - [Model Comparison](#model-comparison)
  - [Performance by Difficulty](#performance-by-difficulty)
  - [Performance by Modality](#performance-by-modality)
  - [Error Analysis](#error-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The rapid expansion of commercial spaceflight has generated unprecedented volumes of biomedical data requiring sophisticated analytical approaches. Large Language Models (LLMs) represent a promising tool for integrating and interpreting complex multi-omics datasets, yet no standardized benchmark exists for evaluating their capabilities in this domain.

SpaceOmicsBench addresses this gap by providing:

1. **Curated Multi-Omics Dataset**: Integrated clinical, transcriptomic, and metabolomic data from three commercial space missions
2. **Expert-Designed Questions**: 115 questions spanning four difficulty levels and multiple reasoning categories
3. **Rigorous Evaluation Framework**: Five-dimensional scoring rubric with LLM-as-Judge methodology
4. **Reproducible Benchmarking Pipeline**: End-to-end evaluation and scoring infrastructure

---

## Dataset Description

### Mission Profiles

| Mission | Duration | Crew Size | Orbit | Key Features |
|---------|----------|-----------|-------|--------------|
| **Inspiration4** | 3 days | 4 | LEO (585 km) | First all-civilian orbital mission |
| **Polaris Dawn** | 5 days | 4 | LEO (1,400 km) | First commercial EVA; highest Earth orbit since Apollo |
| **Fram2** | 5 days | 4 | LEO (Polar) | First polar orbit human mission |

**Total Participants**: 12 crew members
**Sampling Timepoints**: Pre-flight (L-92 to L-3), Post-flight (R+1), Recovery (R+39 to R+82)

### Data Modalities

| Modality | Features | Samples | Description |
|----------|----------|---------|-------------|
| **Clinical** | 34 biomarkers | 37 observations | Complete blood count (CBC), comprehensive metabolic panel (CMP), derived indices (NLR, PLR) |
| **Transcriptomics** | 504 genes | 24 samples | Cell-free RNA sequencing (cfRNA-seq) with differential expression analysis |
| **Metabolomics** | 199 metabolites | Cross-mission | Targeted metabolite profiling with pathway enrichment analysis |

### Key Scientific Findings

The dataset captures several reproducible spaceflight-induced physiological changes:

#### Cross-Modal Conservation
- **Metabolomics correlation**: r = 0.40 (Pearson) across missions
- **Transcriptomics correlation**: r = -0.02 (near-zero, indicating gene-specific responses)
- **Interpretation**: Metabolic responses are highly conserved; transcriptional responses are pathway-specific

#### Hematological Signatures
- **Space Anemia Signature**: 75% concordance in erythroid gene downregulation (EPB42, ANK1, HBB, GYPA)
- **Immune Dysregulation**: Lymphocyte reduction (I4: -26%, PD: -40%, F2: -12%)
- **Stress Response**: NLR elevation (I4: +8%, PD: +52%)

#### Metabolic Perturbations
- **Stress Hormones**: Corticosterone elevated (I4: +0.54, PD: +0.60 log₂FC)
- **Dietary Metabolites**: Caffeine and methylxanthines consistently reduced
- **Recovery Kinetics**: Complete transcriptomic normalization by R+39 (0 DEGs)

---

## Benchmark Design

### Question Taxonomy

#### Distribution by Difficulty

| Difficulty | Count | Percentage | Description | Example Topics |
|------------|-------|------------|-------------|----------------|
| **Easy** | 18 | 16% | Basic data interpretation | Calculating percentages, identifying trends |
| **Medium** | 42 | 37% | Multi-step reasoning | Comparing mission responses, mechanism identification |
| **Hard** | 39 | 34% | Cross-modal integration | Linking clinical-omics findings, temporal analysis |
| **Expert** | 16 | 14% | Novel hypothesis generation | Counterfactual reasoning, predictive modeling |

#### Distribution by Modality

| Modality | Questions | Percentage | Coverage |
|----------|-----------|------------|----------|
| Clinical | 50 | 43% | CBC/CMP interpretation, mission comparisons |
| Transcriptomics | 40 | 35% | Differential expression, pathway analysis |
| Metabolomics | 25 | 22% | Cross-mission correlation, biomarker discovery |
| **Total** | **115** | 100% | |

#### Question Categories

| Category | Description |
|----------|-------------|
| **Factual** | Direct data retrieval and calculation |
| **Comparative** | Cross-mission or cross-modality comparisons |
| **Mechanistic** | Biological mechanism identification |
| **Predictive** | Hypothesis generation and extrapolation |
| **Counterfactual** | Reasoning about alternative scenarios |
| **Data Quality** | Assessment of limitations and uncertainty |

### Evaluation Dimensions

Each response is scored on five dimensions using a 1-5 Likert scale:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Factual Accuracy** | Core | Correctness of stated facts, data citations, and numerical values |
| **Reasoning Quality** | Core | Soundness of scientific logic, causal inference, and mechanistic explanations |
| **Completeness** | Core | Coverage of all question aspects and relevant considerations |
| **Uncertainty Calibration** | Important | Appropriate acknowledgment of limitations, sample sizes, and confidence levels |
| **Domain Integration** | Important | Cross-omics connections, synthesis across modalities, and holistic interpretation |

**Scoring Methodology**: LLM-as-Judge (Claude Sonnet 4) with structured rubrics and flag detection for hallucinations, factual errors, and harmful content.

---

## Experimental Results

### Model Comparison

Full benchmark evaluation comparing Claude Sonnet 4 and Claude Opus 4 on all 115 questions:

#### Overall Performance

| Metric | Claude Sonnet 4 | Claude Opus 4 | Δ |
|--------|-----------------|---------------|---|
| **Overall Score** | 4.34 / 5.00 | **4.45 / 5.00** | +0.11 |
| Factual Accuracy | 4.22 | **4.32** | +0.10 |
| Reasoning Quality | 4.72 | **4.73** | +0.01 |
| Completeness | 4.65 | **4.76** | +0.11 |
| Uncertainty Calibration | 3.66 | **3.89** | +0.23 |
| Domain Integration | 4.43 | **4.55** | +0.12 |

#### Operational Metrics

| Metric | Claude Sonnet 4 | Claude Opus 4 | Ratio |
|--------|-----------------|---------------|-------|
| **Average Response Time** | **14.8 s** | 18.8 s | 0.79× |
| **Total API Cost** | **$1.22** | $6.22 | 0.20× |
| Tokens per Response | 556 | 572 | 1.03× |
| Success Rate | 100% | 100% | — |

### Performance by Difficulty

| Difficulty | n | Sonnet | Opus | Winner |
|------------|---|--------|------|--------|
| Easy | 18 | **4.33** | 4.28 | Sonnet |
| Medium | 42 | 4.30 | **4.50** | Opus |
| Hard | 39 | 4.25 | **4.47** | Opus |
| Expert | 16 | 4.36 | **4.47** | Opus |

**Key Finding**: Opus demonstrates increasing advantage on harder questions, suggesting superior complex reasoning capabilities.

### Performance by Modality

| Modality | n | Sonnet | Opus | Winner |
|----------|---|--------|------|--------|
| Clinical | 50 | 4.19 | **4.30** | Opus |
| Transcriptomics | 40 | 4.30 | **4.54** | Opus |
| Metabolomics | 25 | 4.50 | **4.61** | Opus |

**Key Finding**: Both models perform best on metabolomics questions; Opus shows consistent advantage across all modalities.

### Error Analysis

#### Quality Flags

| Flag Type | Sonnet | Opus | Description |
|-----------|--------|------|-------------|
| Hallucination | 13.0% (15/115) | 14.8% (17/115) | Fabricated data or unsupported claims |
| Factual Error | 17.4% (20/115) | 17.4% (20/115) | Incorrect statements about provided data |
| Harmful Content | 0% | 0% | Dangerous medical advice |

#### Common Error Patterns

1. **Unsupported Mechanistic Claims**: Both models occasionally assert biological mechanisms without sufficient data support
2. **Numerical Approximation**: Minor rounding or calculation errors in complex multi-step problems
3. **Overconfident Extrapolation**: Extending findings beyond what small sample sizes (n=4) support

### Model Selection Guidelines

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Production/Research | **Opus** | +2.6% quality, better uncertainty calibration |
| Rapid Prototyping | **Sonnet** | 5× cost reduction, 27% faster |
| Easy Questions | Sonnet | Comparable performance at lower cost |
| Complex Integration | **Opus** | Superior cross-modal reasoning |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jang1563/SpaceOmicsBench.git
cd SpaceOmicsBench

# Install dependencies
pip install -r requirements.txt

# Configure API access
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Requirements

- Python 3.8+
- `anthropic>=0.18.0`
- `pandas>=1.5.0`
- `tqdm>=4.65.0`

---

## Usage

### Running Evaluations

```bash
# Quick test (10 questions, ~2 minutes)
python run_evaluation.py --sample 10

# Full evaluation (115 questions, ~30 minutes)
python run_evaluation.py --full

# Specific model
python run_evaluation.py --full --model claude-opus-4-20250514

# Specific modality
python run_evaluation.py --full --modality transcriptomics

# Retry failed questions (for API rate limits)
python run_evaluation.py --retry results/evaluation_results_XXXXXX.json
```

### Scoring Responses

```bash
# LLM-as-Judge scoring
python score_responses.py results/evaluation_results_XXXXXX.json
```

### Generating Reports

```bash
# HTML comparison report
python generate_report.py
```

Output: `results/sonnet_vs_opus_comparison.html`

### Available Models

| Model ID | Description | Cost (Input/Output per 1M tokens) |
|----------|-------------|-----------------------------------|
| `claude-sonnet-4-20250514` | Claude Sonnet 4 (default) | $3 / $15 |
| `claude-opus-4-20250514` | Claude Opus 4 | $15 / $75 |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet | $3 / $15 |
| `claude-3-opus-20240229` | Claude 3 Opus | $15 / $75 |

---

## Repository Structure

```
SpaceOmicsBench/
├── README.md                 # This documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
│
├── run_evaluation.py         # Main evaluation script
├── score_responses.py        # LLM-as-Judge scoring
├── generate_report.py        # HTML report generator
│
├── data/                     # Multi-omics datasets
│   ├── clinical_feature_matrix.csv       # 34 biomarkers × 37 observations
│   ├── cfrna_feature_matrix.csv          # 504 genes × 24 samples
│   ├── metabolomics_*.csv                # Metabolite data and annotations
│   ├── gene_annotations.csv              # Gene metadata and pathways
│   └── summary_statistics.csv            # Aggregate statistics
│
├── tasks/                    # Benchmark questions
│   ├── question_bank.json               # Clinical questions (n=50)
│   ├── transcriptomics_questions_v2.json # Transcriptomics questions (n=40)
│   └── metabolomics_questions.json      # Metabolomics questions (n=25)
│
└── results/                  # Evaluation outputs
    ├── evaluation_results_*.json        # Raw model responses
    ├── scored_results_*.json            # Scored responses with metrics
    └── sonnet_vs_opus_comparison.html   # Visual comparison report
```

---

## Citation

If you use SpaceOmicsBench in your research, please cite:

```bibtex
@software{spaceomicsbench2024,
  title     = {{SpaceOmicsBench}: A Multi-Modal Benchmark for Evaluating Large
               Language Models on Spaceflight Biomedical Data},
  author    = {Jang, Kirubin},
  year      = {2024},
  version   = {2.1},
  url       = {https://github.com/jang1563/SpaceOmicsBench},
  note      = {115 questions across clinical, transcriptomics, and metabolomics
               modalities from Inspiration4, Polaris Dawn, and Fram2 missions}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Mission Data**: Biomedical data from Inspiration4, Polaris Dawn, and Fram2 commercial space missions
- **Infrastructure**: Anthropic Claude API for LLM evaluation and scoring
- **Mission Partners**: SpaceX and mission crews for enabling commercial spaceflight research

---

## Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/jang1563/SpaceOmicsBench/issues).
