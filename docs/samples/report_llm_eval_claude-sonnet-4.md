# SpaceOmicsBench v2 - LLM Evaluation Report

**Model:** claude-sonnet-4-20250514
**Date:** 2026-02-25
**Questions scored:** 100/100

---

## Overall Scores

| Dimension | Weight | Score |
|-----------|--------|-------|
| Factual Accuracy | 0.25 | 4.60/5 |
| Reasoning Quality | 0.25 | 4.83/5 |
| Completeness | 0.20 | 4.75/5 |
| Uncertainty Calibration | 0.15 | 3.90/5 |
| Domain Integration | 0.15 | 4.38/5 |
| **Weighted Total** | **1.00** | **4.55/5** |

## Scores by Difficulty

| Difficulty | N | Mean | Min | Max |
|------------|---|------|-----|-----|
| Easy | 17 | 4.29 | 2.30 | 4.85 |
| Medium | 32 | 4.63 | 3.35 | 5.00 |
| Hard | 31 | 4.60 | 2.05 | 5.00 |
| Expert | 20 | 4.57 | 2.15 | 5.00 |

## Scores by Modality

| Modality | N | Mean Score |
|----------|---|------------|
| Clinical | 10 | 4.62 |
| Transcriptomics | 10 | 4.30 |
| Proteomics | 10 | 4.70 |
| Metabolomics | 10 | 4.64 |
| Spatial | 10 | 4.46 |
| Microbiome | 10 | 4.18 |
| Cross Mission | 18 | 4.84 |
| Multi Omics | 12 | 4.40 |
| Methods | 10 | 4.62 |

## Quality Flags

| Flag | Count |
|------|-------|
| Hallucination | 5 |
| Factual Error | 16 |
| Harmful Recommendation | 0 |
| Exceeds Data Scope | 4 |
| Novel Insight | 87 |

## Per-Question Results

| ID | Modality | Difficulty | Score | Flags |
|----|----------|------------|-------|-------|
| Q01 | clinical | easy | 3.95 | - |
| Q02 | clinical | easy | 4.70 | novel_insight |
| Q03 | clinical | medium | 4.85 | novel_insight |
| Q04 | clinical | medium | 4.70 | novel_insight |
| Q05 | clinical | hard | 4.60 | novel_insight |
| Q06 | clinical | hard | 4.85 | novel_insight |
| Q07 | clinical | expert | 4.85 | novel_insight |
| Q08 | transcriptomics | easy | 4.30 | novel_insight |
| Q09 | transcriptomics | easy | 4.10 | novel_insight |
| Q10 | transcriptomics | medium | 4.55 | novel_insight |
| Q100 | methods | expert | 4.70 | novel_insight |
| Q11 | transcriptomics | medium | 4.70 | novel_insight |
| Q12 | transcriptomics | hard | 3.00 | factual_error, novel_insight |
| Q13 | transcriptomics | hard | 5.00 | novel_insight |
| Q14 | transcriptomics | expert | 4.45 | novel_insight |
| Q15 | proteomics | easy | 4.70 | novel_insight |
| Q16 | proteomics | medium | 4.70 | novel_insight |
| Q17 | proteomics | medium | 4.70 | novel_insight |
| Q18 | proteomics | hard | 4.85 | novel_insight |
| Q19 | proteomics | hard | 4.85 | novel_insight |
| Q20 | proteomics | expert | 4.85 | novel_insight |
| Q21 | metabolomics | easy | 4.45 | novel_insight |
| Q22 | metabolomics | medium | 4.45 | factual_error, novel_insight |
| Q23 | metabolomics | medium | 4.70 | novel_insight |
| Q24 | metabolomics | hard | 4.85 | novel_insight |
| Q25 | metabolomics | hard | 4.55 | novel_insight |
| Q26 | metabolomics | expert | 4.85 | novel_insight |
| Q27 | spatial | medium | 4.45 | novel_insight |
| Q28 | spatial | medium | 4.25 | novel_insight |
| Q29 | spatial | hard | 4.70 | exceeds_data_scope, novel_insight |
| Q30 | spatial | expert | 4.85 | novel_insight |
| Q31 | microbiome | easy | 2.30 | hallucination, factual_error, exceeds_data_scope |
| Q32 | microbiome | medium | 4.85 | novel_insight |
| Q33 | microbiome | medium | 5.00 | novel_insight |
| Q34 | microbiome | hard | 2.05 | factual_error |
| Q35 | cross_mission | easy | 4.85 | novel_insight |
| Q36 | cross_mission | easy | 4.85 | novel_insight |
| Q37 | cross_mission | medium | 5.00 | novel_insight |
| Q38 | cross_mission | medium | 4.85 | novel_insight |
| Q39 | cross_mission | medium | 4.70 | novel_insight |
| Q40 | cross_mission | medium | 4.85 | novel_insight |
| Q41 | cross_mission | medium | 4.85 | novel_insight |
| Q42 | cross_mission | hard | 4.70 | novel_insight |
| Q43 | cross_mission | hard | 4.85 | novel_insight |
| Q44 | cross_mission | hard | 4.85 | novel_insight |
| Q45 | cross_mission | hard | 4.85 | novel_insight |
| Q46 | cross_mission | hard | 5.00 | novel_insight |
| Q47 | cross_mission | expert | 5.00 | novel_insight |
| Q48 | cross_mission | expert | 4.85 | novel_insight |
| Q49 | cross_mission | expert | 4.85 | novel_insight |
| Q50 | cross_mission | expert | 4.85 | novel_insight |
| Q51 | multi_omics | easy | 4.70 | - |
| Q52 | multi_omics | medium | 4.25 | factual_error, novel_insight |
| Q53 | multi_omics | medium | 4.85 | novel_insight |
| Q54 | multi_omics | hard | 4.85 | factual_error, novel_insight |
| Q55 | multi_omics | hard | 4.60 | novel_insight |
| Q56 | multi_omics | hard | 5.00 | novel_insight |
| Q57 | multi_omics | expert | 4.60 | novel_insight |
| Q58 | multi_omics | expert | 3.35 | hallucination, factual_error, novel_insight |
| Q59 | methods | easy | 4.85 | novel_insight |
| Q60 | methods | medium | 4.85 | novel_insight |
| Q61 | clinical | easy | 4.15 | - |
| Q62 | clinical | medium | 4.70 | novel_insight |
| Q63 | clinical | hard | 4.85 | novel_insight |
| Q64 | transcriptomics | medium | 3.35 | hallucination, factual_error, exceeds_data_scope |
| Q65 | transcriptomics | hard | 4.85 | novel_insight |
| Q66 | transcriptomics | expert | 4.70 | novel_insight |
| Q67 | proteomics | easy | 4.85 | novel_insight |
| Q68 | proteomics | medium | 4.35 | - |
| Q69 | proteomics | hard | 4.25 | novel_insight |
| Q70 | proteomics | expert | 4.85 | novel_insight |
| Q71 | metabolomics | easy | 4.55 | novel_insight |
| Q72 | metabolomics | medium | 4.45 | factual_error, novel_insight |
| Q73 | metabolomics | hard | 4.70 | novel_insight |
| Q74 | metabolomics | expert | 4.85 | novel_insight |
| Q75 | spatial | easy | 3.15 | hallucination, factual_error |
| Q76 | spatial | medium | 4.45 | factual_error |
| Q77 | spatial | medium | 4.85 | novel_insight |
| Q78 | spatial | hard | 4.70 | novel_insight |
| Q79 | spatial | hard | 4.75 | factual_error, novel_insight |
| Q80 | spatial | expert | 4.40 | factual_error, novel_insight |
| Q81 | microbiome | easy | 3.95 | - |
| Q82 | microbiome | medium | 4.85 | novel_insight |
| Q83 | microbiome | medium | 4.85 | novel_insight |
| Q84 | microbiome | hard | 4.25 | factual_error, novel_insight |
| Q85 | microbiome | hard | 4.85 | novel_insight |
| Q86 | microbiome | expert | 4.85 | novel_insight |
| Q87 | cross_mission | hard | 4.50 | novel_insight |
| Q88 | cross_mission | expert | 4.85 | novel_insight |
| Q89 | multi_omics | medium | 4.70 | novel_insight |
| Q90 | multi_omics | hard | 4.70 | novel_insight |
| Q91 | multi_omics | hard | 5.00 | novel_insight |
| Q92 | multi_omics | expert | 2.15 | factual_error |
| Q93 | methods | easy | 4.55 | - |
| Q94 | methods | medium | 4.70 | novel_insight |
| Q95 | methods | medium | 3.85 | hallucination, factual_error, exceeds_data_scope |
| Q96 | methods | medium | 4.85 | novel_insight |
| Q97 | methods | hard | 4.70 | novel_insight |
| Q98 | methods | hard | 4.55 | novel_insight |
| Q99 | methods | expert | 4.60 | novel_insight |
