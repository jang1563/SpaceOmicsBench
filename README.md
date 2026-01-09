# SpaceOmicsBench

A comprehensive benchmark suite for evaluating Large Language Models (LLMs) on spaceflight biomedical multi-omics data from commercial space missions.

## Overview

SpaceOmicsBench v2.1 provides a rigorous evaluation framework for assessing LLM capabilities in understanding, reasoning, and integrating findings across multiple omics layers in the context of human spaceflight physiology.

## Dataset

### Missions

| Mission | Duration | Crew | Key Features |
|---------|----------|------|--------------|
| **Inspiration4** | 3 days | 4 | First all-civilian orbital mission |
| **Polaris Dawn** | 5 days | 4 | First commercial EVA (spacewalk) |

### Data Modalities

| Modality | Features | Description |
|----------|----------|-------------|
| **Clinical** | 34 biomarkers | CBC/CMP blood panels |
| **Transcriptomics** | 5,346 genes | Cell-free RNA sequencing (cfRNA-seq) |
| **Metabolomics** | 199 metabolites | Targeted metabolite profiling |

### Key Scientific Findings in Dataset

- **Cross-Mission Correlation**: Metabolomics shows 20x higher correlation (r=0.40) than transcriptomics (r=-0.02)
- **Space Anemia Signature**: Erythroid genes show 75% concordance across missions (EPB42, ANK1, HBB)
- **Immune Response**: NLR increased 52% in Polaris Dawn, lymphocytes decreased 40%
- **Stress Response**: Corticosterone elevated (+0.57 log2FC), caffeine decreased (-2.0 log2FC)
- **Temporal Recovery**: Complete transcriptomic normalization by R+39 (0 DEGs)

## Benchmark Questions

### Distribution

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 18 | Basic data interpretation |
| Medium | 42 | Multi-step reasoning |
| Hard | 39 | Cross-modal integration |
| Expert | 16 | Novel hypothesis generation |

| Modality | Questions |
|----------|-----------|
| Clinical | 50 |
| Transcriptomics | 40 |
| Metabolomics | 25 |
| **Total** | **115** |

---

## Model Comparison Results

### Overall Performance

| Metric | Claude Sonnet 4 | Claude Opus 4 | Winner |
|--------|-----------------|---------------|--------|
| **Overall Score** | 4.34/5 | **4.45/5** | Opus (+0.11) |
| **Factual Accuracy** | 4.22/5 | **4.32/5** | Opus |
| **Reasoning Quality** | 4.72/5 | 4.73/5 | Tie |
| **Completeness** | 4.65/5 | **4.76/5** | Opus |
| **Uncertainty Calibration** | 3.66/5 | **3.89/5** | Opus (+0.23) |
| **Domain Integration** | 4.43/5 | **4.55/5** | Opus |

### Performance Metrics

| Metric | Claude Sonnet 4 | Claude Opus 4 | Winner |
|--------|-----------------|---------------|--------|
| **Avg Response Time** | **14.8s** | 18.8s | Sonnet (27% faster) |
| **Total Cost** | **$1.22** | $6.22 | Sonnet (5.1x cheaper) |
| **Tokens/Response** | 556 | 554 | Tie |
| **Success Rate** | 100% | 100% | Tie |

### Quality Scores by Difficulty

| Difficulty | Sonnet | Opus | Winner |
|------------|--------|------|--------|
| Easy | **4.33** | 4.28 | Sonnet |
| Medium | 4.30 | **4.50** | Opus |
| Hard | 4.36 | **4.47** | Opus |
| Expert | 4.36 | **4.47** | Opus |

### Quality Scores by Modality

| Modality | Sonnet | Opus | Winner |
|----------|--------|------|--------|
| Clinical | 4.19 | **4.30** | Opus |
| Transcriptomics | 4.42 | **4.54** | Opus |
| Metabolomics | 4.50 | **4.61** | Opus |

### Quality Flags (Issues Detected)

| Flag Type | Sonnet | Opus |
|-----------|--------|------|
| Hallucination | 13.2% | 14.8% |
| Factual Error | 17.5% | 17.4% |
| Harmful Content | 0% | 0% |

### Recommendation

- **Choose Sonnet** for: Cost-sensitive applications, rapid iteration, initial prototyping
- **Choose Opus** for: Maximum quality, uncertainty-aware responses, production deployments

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jang1563/SpaceOmicsBench.git
cd SpaceOmicsBench

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

## Usage

### Run Evaluation

```bash
# Quick test (10 questions)
python run_evaluation.py --sample 10

# Full evaluation (115 questions)
python run_evaluation.py --full

# Specific model
python run_evaluation.py --full --model claude-opus-4-20250514

# Specific modality
python run_evaluation.py --full --modality transcriptomics

# Retry failed questions
python run_evaluation.py --retry results/evaluation_results_XXXXXX.json
```

### Score Responses (LLM-as-Judge)

```bash
python score_responses.py results/evaluation_results_XXXXXX.json
```

### Generate Comparison Report

```bash
python generate_report.py
```

Creates `results/sonnet_vs_opus_comparison.html` with interactive visualizations.

## Project Structure

```
SpaceOmicsBench/
├── README.md
├── LICENSE
├── requirements.txt
├── run_evaluation.py       # Main evaluation script
├── score_responses.py      # LLM-as-Judge scoring
├── generate_report.py      # HTML report generator
├── data/                   # Multi-omics datasets
│   ├── clinical_*.csv
│   ├── cfrna_*.csv
│   ├── metabolomics_*.csv
│   └── *.json              # Configuration files
├── tasks/                  # Question banks
│   ├── question_bank.json           # Clinical (50 questions)
│   ├── transcriptomics_questions_v2.json  # Transcriptomics (40 questions)
│   └── metabolomics_questions.json  # Metabolomics (25 questions)
└── results/                # Evaluation outputs
    ├── evaluation_results_*.json
    ├── scored_results_*.json
    └── sonnet_vs_opus_comparison.html
```

## Scoring Dimensions

| Dimension | Description | Weight |
|-----------|-------------|--------|
| **Factual Accuracy** | Correctness of stated facts and data citations | Core |
| **Reasoning Quality** | Soundness of scientific logic and inference | Core |
| **Completeness** | Coverage of all question aspects | Core |
| **Uncertainty Calibration** | Appropriate acknowledgment of limitations | Important |
| **Domain Integration** | Cross-omics connections and synthesis | Important |

## Available Models

| Model ID | Description | Cost (Input/Output per 1M) |
|----------|-------------|---------------------------|
| `claude-sonnet-4-20250514` | Claude Sonnet 4 (default) | $3 / $15 |
| `claude-opus-4-20250514` | Claude Opus 4 | $15 / $75 |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet | $3 / $15 |
| `claude-3-opus-20240229` | Claude 3 Opus | $15 / $75 |

## Citation

```bibtex
@software{spaceomicsbench2024,
  title = {SpaceOmicsBench: A Benchmark for LLM Evaluation on Spaceflight Biomedical Multi-Omics Data},
  author = {Jang, Kirubin},
  year = {2024},
  version = {2.1},
  url = {https://github.com/jang1563/SpaceOmicsBench}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Biomedical data from Inspiration4 and Polaris Dawn commercial space missions
- Anthropic Claude API for LLM evaluation and scoring
- SpaceX and mission crews for enabling commercial spaceflight research
