# Cross-Judge Verification Analysis

## 2x2 Score Matrix

| | Claude Judge | GPT-4o Judge |
|:---|:---:|:---:|
| Claude Sonnet | 4.55 | 4.76 |
| GPT-4o | 3.64 | 4.36 |
| Gap (Claude - GPT-4o) | +0.91 | +0.40 |

## Key Findings

### 1. Both judges consistently rank Claude > GPT-4o
- Claude judge: 4.55 vs 3.64 (+0.91)
- GPT-4o judge: 4.76 vs 4.36 (+0.40)
- No reversal on any modality

### 2. GPT-4o judge is uniformly more lenient
- +0.21 more generous to Claude responses
- +0.71 more generous to GPT-4o responses
- Mild same-vendor leniency: +0.51 differential toward GPT-4o

### 3. Dimension-level disagreements

| Dimension | Claude resp (GJ-CJ) | GPT-4o resp (GJ-CJ) |
|:---|:---:|:---:|
| factual_accuracy | +0.24 | +0.72 |
| reasoning_quality | +0.09 | +0.52 |
| completeness | +0.09 | +0.73 |
| uncertainty_calibration | +0.64 | +1.20 |
| domain_integration | +0.07 | +0.52 |

Largest disagreement: uncertainty_calibration (+1.20 for GPT-4o responses)

### 4. Flag disagreements

| Flag | C×CJ | C×GJ | G×CJ | G×GJ |
|:---|:---:|:---:|:---:|:---:|
| factual_error | 16 | 7 | 18 | 3 |
| hallucination | 5 | 1 | 4 | 0 |
| novel_insight | 87 | 31 | 18 | 4 |

Claude judge detects 3.4x more factual errors than GPT-4o judge.

### 5. Top judge disagreement questions

| QID | Model | CJ | GJ | Gap | Issue |
|:---|:---|:---:|:---:|:---:|:---|
| Q92 | Claude | 2.15 | 4.85 | +2.70 | H1 task interpretation |
| Q11 | GPT-4o | 2.25 | 4.70 | +2.45 | B2 classification type |
| Q41 | GPT-4o | 2.75 | 5.00 | +2.25 | scRNA-seq vs bulk error |
| Q34 | GPT-4o | 1.85 | 3.70 | +1.85 | Normalization methodology |
| Q58 | Claude | 3.35 | 5.00 | +1.65 | Hallucination detection |

### 6. Modality-level judge agreement

| Modality | Avg |Judge Gap| |
|:---|:---:|
| Microbiome | 0.66 (worst) |
| Multi_omics | 0.64 |
| Transcriptomics | 0.53 |
| Methods | 0.49 |
| Cross_mission | 0.44 |
| Clinical | 0.41 |
| Spatial | 0.39 |
| Metabolomics | 0.37 |
| Proteomics | 0.36 (best) |

### 7. Low-scoring questions (< 3.0 from any judge)
13 questions total. Most problematic:
- Q34 (microbiome/hard): Both models low — normalization method ambiguity
- Q92 (multi_omics/expert): Extreme judge disagreement — H1 task interpretation
- Q12 (transcriptomics/hard): Both models struggle — B1 ablation crossover complexity

## Conclusions

1. **Claude judge is preferred as primary** — stricter, more discriminating, better factual error detection
2. **GPT-4o judge validates ranking** — no same-vendor bias reversal
3. **Benchmark improvements needed** — Q34 normalization, Q92 H1 clarity, uncertainty_calibration rubric
