# SpaceOmicsBench

A benchmark suite for evaluating Large Language Models (LLMs) on spaceflight biomedical multi-omics data.

## Overview

SpaceOmicsBench v2.1 provides a comprehensive evaluation framework for assessing LLM capabilities in understanding and reasoning about spaceflight biomedical data from commercial space missions.

### Dataset

- **Missions**: Inspiration4 (3-day), Polaris Dawn (5-day with EVA)
- **Modalities**:
  - Clinical (CBC/CMP): 34 blood biomarkers
  - Transcriptomics (cfRNA-seq): 5,346 genes
  - Metabolomics: 199 metabolites
- **Questions**: 115 curated questions across 4 difficulty levels

### Key Findings from Benchmark

| Metric | Sonnet 4 | Opus 4 |
|--------|----------|--------|
| Overall Score | 4.34/5 | 4.45/5 |
| Speed | 14.8s | 18.8s |
| Cost (115 questions) | $1.22 | $6.22 |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SpaceOmicsBench.git
cd SpaceOmicsBench

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

## Quick Start

### Run a Quick Test (10 questions)
```bash
python run_evaluation.py --sample 10
```

### Run Full Evaluation (115 questions)
```bash
python run_evaluation.py --full
```

### Run with Specific Model
```bash
python run_evaluation.py --full --model claude-opus-4-20250514
```

### Run Specific Modality
```bash
python run_evaluation.py --full --modality transcriptomics
```

### Retry Failed Questions
```bash
python run_evaluation.py --retry results/evaluation_results_XXXXXX.json
```

## Scoring Responses

Score LLM responses using Claude as a judge:

```bash
python score_responses.py results/evaluation_results_XXXXXX.json
```

### Scoring Dimensions (1-5 scale)
1. **Factual Accuracy**: Are stated facts correct?
2. **Reasoning Quality**: Is scientific reasoning sound?
3. **Completeness**: Does it address all aspects?
4. **Uncertainty Calibration**: Does it acknowledge limitations?
5. **Domain Integration**: Does it connect findings across omics?

## Generate Comparison Report

Generate an HTML comparison report between models:

```bash
python generate_report.py
```

This creates `results/sonnet_vs_opus_comparison.html` with:
- Quality scores comparison
- Performance metrics
- Cost analysis
- Detailed breakdowns by difficulty and modality

## Project Structure

```
SpaceOmicsBench/
├── README.md
├── requirements.txt
├── run_evaluation.py      # Main evaluation script
├── score_responses.py     # LLM-as-judge scoring
├── generate_report.py     # HTML report generator
├── data/                  # Raw omics data files
│   ├── clinical/
│   ├── transcriptomics/
│   └── metabolomics/
├── tasks/                 # Question banks
│   ├── question_bank.json
│   ├── transcriptomics_questions_v2.json
│   └── metabolomics_questions.json
└── results/               # Evaluation outputs
    ├── evaluation_results_*.json
    ├── scored_results_*.json
    └── sonnet_vs_opus_comparison.html
```

## Available Models

| Model ID | Description |
|----------|-------------|
| `claude-sonnet-4-20250514` | Claude Sonnet 4 (default) |
| `claude-opus-4-20250514` | Claude Opus 4 |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet |
| `claude-3-opus-20240229` | Claude 3 Opus |

## API Costs (Approximate)

| Model | Input (per 1M) | Output (per 1M) | Full Benchmark |
|-------|----------------|-----------------|----------------|
| Sonnet 4 | $3 | $15 | ~$1.22 |
| Opus 4 | $15 | $75 | ~$6.22 |

## Citation

If you use SpaceOmicsBench in your research, please cite:

```bibtex
@software{spaceomicsbench2024,
  title = {SpaceOmicsBench: A Benchmark for LLM Evaluation on Spaceflight Biomedical Data},
  year = {2024},
  version = {2.1}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data from Inspiration4 and Polaris Dawn commercial space missions
- Anthropic Claude API for LLM evaluation
