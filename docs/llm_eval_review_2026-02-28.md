# LLM Evaluation Review — Benchmark Update Feasibility

**Date**: 2026-02-28
**Scope**: 9-Model Leaderboard (Judge: Claude Sonnet 4.6) 결과 검토
**목적**: 기존 벤치마크를 보존하면서, 새로운 LLM 평가 결과를 v2.1 업데이트 버전으로 공식화할 수 있는지 판단

---

## 1. 현재 Leaderboard 요약

| Rank | Model | Score | Easy | Med | Hard | Expert |
|------|-------|:-----:|:----:|:---:|:----:|:------:|
| 1 | Claude Sonnet 4.6 | **4.60** | 4.58 | 4.49 | 4.66 | 4.69 |
| 2 | Claude Haiku 4.5 | **4.39** | 4.35 | 4.45 | 4.38 | 4.37 |
| 3 | DeepSeek-V3 | **4.31** | 4.11 | 4.22 | 4.42 | 4.46 |
| 4 | Claude Sonnet 4 | **4.03** | 4.05 | 4.06 | 4.00 | 4.01 |
| 5 | Gemini 2.5 Flash | **3.43** | 3.68 | 3.91 | 3.25 | 2.75 |
| 6 | Llama-3.3-70B (Together) | **3.30** | 3.57 | 3.33 | 3.29 | 3.06 |
| 7 | GPT-4o | **3.30** | 3.24 | 3.40 | 3.28 | 3.22 |
| 8 | GPT-4o Mini | **3.30** | 3.37 | 3.40 | 3.29 | 3.10 |
| 9 | Llama-3.3-70B (Groq) | **3.29** | 3.48 | 3.35 | 3.27 | 3.06 |

**Tier 구조**:
- Tier 1 (4.3+): Sonnet 4.6, Haiku 4.5, DeepSeek-V3
- Tier 2 (~4.0): Sonnet 4
- Tier 3 (~3.3): Gemini, GPT-4o, GPT-4o Mini, Llama-70B x2

---

## 2. Per-Dimension 분석

| Model | Factual | Reasoning | Complete | Uncertainty | Domain |
|-------|:-------:|:---------:|:--------:|:-----------:|:------:|
| Sonnet 4.6 | 4.59 | **4.96** | **4.73** | **4.09** | **4.33** |
| Haiku 4.5 | 4.35 | 4.84 | 4.53 | 3.82 | 4.12 |
| DeepSeek-V3 | 4.34 | 4.72 | 4.37 | 3.72 | 4.09 |
| Sonnet 4 | 4.26 | 4.47 | 4.07 | 3.14 | 3.74 |
| Gemini 2.5 Flash | 4.13 | 3.75 | 3.22 | 2.62 | 2.83 |
| Llama-70B (Together) | 3.98 | 3.49 | 3.22 | 2.65 | 2.63 |
| GPT-4o | 3.98 | 3.60 | 3.13 | 2.57 | 2.63 |
| GPT-4o Mini | 3.91 | 3.51 | 3.19 | 2.75 | 2.63 |
| Llama-70B (Groq) | 4.02 | 3.50 | 3.18 | 2.58 | 2.58 |

**관찰**:
- 가장 큰 차별화 요소: domain_integration (range 1.75), reasoning_quality (range 1.47)
- 가장 작은 차별화: factual_accuracy (range 0.68) — 가장 객관적인 차원에서 차이가 작음
- Uncertainty calibration이 **모든 모델에서 최저** — small-N spaceflight 데이터의 본질적 어려움

---

## 3. Per-Modality 분석

| Model | Clinical | Trans | Prot | Metab | Spatial | Micro | Cross-M | Multi-O | Methods |
|-------|:--------:|:-----:|:----:|:-----:|:-------:|:-----:|:-------:|:-------:|:-------:|
| Sonnet 4.6 | **4.76** | 4.33 | 4.47 | 4.65 | 4.34 | 4.66 | **4.87** | 4.59 | 4.51 |
| Haiku 4.5 | 4.50 | 4.06 | 4.29 | **4.68** | 3.94 | 4.42 | 4.56 | 4.48 | 4.44 |
| DeepSeek-V3 | 4.51 | 4.13 | 4.30 | 4.36 | 3.89 | 4.33 | 4.56 | 4.11 | 4.46 |
| Sonnet 4 | 4.28 | 3.67 | 3.91 | 4.04 | 3.52 | 4.03 | 4.32 | 3.94 | 4.35 |
| Gemini | 3.74 | 3.04 | 3.46 | 3.41 | 3.01 | 3.42 | 3.83 | 3.02 | 3.73 |
| Llama-Tg | 3.50 | 3.04 | 3.28 | 3.30 | 3.06 | 3.41 | 3.54 | 3.31 | 3.11 |
| GPT-4o | 3.38 | 3.02 | 3.44 | 3.26 | 2.91 | 3.38 | 3.53 | 3.33 | 3.29 |
| GPT-4o-mini | 3.60 | 3.06 | 3.13 | 3.24 | 3.00 | 3.14 | 3.58 | 3.34 | 3.39 |
| Llama-Gq | 3.57 | 2.93 | 3.39 | 3.32 | 2.97 | 3.43 | 3.56 | 3.16 | 3.10 |

- **Cross-mission**: 모든 모델에서 가장 강함 (Twins Study 문헌이 풍부)
- **Spatial**: 모든 모델에서 가장 약함 (spatial transcriptomics는 상대적으로 신기술)

---

## 4. Flag 분석

| Model | Hallucination | Factual Error | Novel Insight | Exceeds Scope |
|-------|:------------:|:-------------:|:-------------:|:-------------:|
| Sonnet 4.6 | 11 | 10 | **92** | 12 |
| Haiku 4.5 | 15 | 18 | 79 | 17 |
| DeepSeek-V3 | 11 | 14 | 62 | 8 |
| Sonnet 4 | 10 | 18 | 44 | 6 |
| Gemini 2.5 Flash | 9 | 10 | 5 | 0 |
| Llama-70B (Together) | 5 | 14 | **0** | 1 |
| GPT-4o | 3 | 12 | **0** | 0 |
| GPT-4o Mini | 9 | 21 | **0** | 1 |
| Llama-70B (Groq) | 5 | 13 | **0** | 2 |

**Novel insight flag 분포 (92 vs 0)**: 극단적이나, weighted score에 직접 반영되지 않음. 응답 길이와 강하게 상관 (Claude/DeepSeek 평균 617-1326 토큰 vs GPT/Llama 379-490 토큰). 보고 시 주석 필요.

---

## 5. 반드시 수정해야 할 문제 (MUST-FIX)

### 5-A. Gemini 2.5 Flash: Truncation으로 인한 부당한 점수

**원인**: `max_tokens=2000`으로 설정했지만, Gemini의 thinking token이 이 예산을 잠식. ~1920 토큰을 사고에 소비하면 가시적 응답은 ~80 토큰에서 잘림.

| 구분 | 개수 | 평균 점수 |
|------|:----:|:---------:|
| Thinking-truncated (77-80 tok) | 14 | **1.52** |
| 자연스러운 짧은 답변 | 5 | 3.07 |
| 정상 응답 (>80 tok) | 81 | **3.79** |
| 전체 | 100 | 3.43 |

**Truncation 난이도별 분포**:
- Easy: 0/17 (0%)
- Medium: 0/32 (0%)
- Hard: 7/31 (23%)
- Expert: 7/20 (35%)

**예상 공정 점수**: 3.43 → **~3.72** (약 +0.29 인위적 감소)

**조치**: `max_tokens`를 8192+ 로 설정하여 재실행 필요. 또는 thinking 비활성화.

### 5-B. 문제 질문 3개: ground_truth_key_facts 오류

#### Q27 (spatial/medium) — 9개 모델 중 8개 hallucination flag

- **문제**: ground truth에 "E4: even lower positive rate"로 기재
- **실제 data_context** (`spatial.md`): E4 positive rate = 0.21%, E1 = 0.19% → E4가 **더 높음**
- **결과**: context를 정확히 읽고 positive count (35, 40, 11, 18)를 인용한 모델이 오히려 hallucination으로 처벌됨
- **수정**: ground_truth_key_facts를 data_context와 일치시키기

#### Q28 (spatial/medium) — **9개 모델 전부** factual_error flag

- **문제**: ground truth: `inner_epidermis ~15` vs data_context: `inner_epidermis ~11` — 내부 불일치
- **결과**: data_context를 정확히 읽은 모델 (11이라고 답한)이 factual error 처리됨
- **수정**: ground_truth_key_facts의 ~15를 ~11로 수정 (data_context가 모델에 제공되는 것이므로)

#### Q64 (transcriptomics/medium) — **9개 모델 전부** factual_error, 8개 hallucination

- **문제**: ground truth가 RF ablation 값만 수록 (All=0.884, No-effect=0.863, Effect-only=0.813)
- **실제 data_context** (`transcriptomics.md`): 5개 모델 전체 ablation 테이블 포함 (XGBoost no-effect=0.899, LightGBM=0.922 등)
- **결과**: XGBoost 0.899 등을 context에서 정확히 인용하면 "hallucination" 처리됨
- **수정**: ground_truth_key_facts에 전체 ablation 테이블 값 추가

**공통 패턴**: 세 질문 모두 **ground truth의 오류/불완전성**을 측정하고 있으며, 모델 능력을 측정하지 못하고 있음. Context를 정확히 읽은 모델이 penalize되는 역설적 상황.

### 5-C. Haiku Q10 스코어링 실패

- JSON 파싱 에러로 99/100만 채점됨
- 의도된 점수 3.35 반영 필요

---

## 6. Judge Bias 분석 — 문제 없음 (PASS)

### Self-Judge 비교 (동일 Sonnet 4.6 응답)

| 지표 | Sonnet 4.6 Judge (self) | Sonnet 4 Judge (external) |
|------|:-----------------------:|:-------------------------:|
| Weighted score | **4.60** | **4.73** |
| Hallucination flags | 11 | 1 |
| Factual error flags | 10 | 3 |
| Exceeds scope flags | 12 | 0 |
| Novel insight flags | 92 | 95 |

- Self-judge가 **-0.13점 더 엄격** (p < 0.001, paired t-test)
- 51개 질문에서 self-judge가 더 낮게 채점, 17개만 더 높게
- Self-judge가 hallucination을 **11배 더 많이** 자기에게 부여
- **Anti-self-favoring bias** — 자기 편향의 정반대

### Cross-Judge Matrix

| Respondent | Sonnet 4 Judge | Sonnet 4.6 Judge | GPT-4o Judge |
|-----------|:-:|:-:|:-:|
| Claude Sonnet 4.6 | 4.73 | **4.60** | — |
| Claude Sonnet 4 | 4.55 | 4.03 | 4.76 |
| GPT-4o | 3.64 | 3.30 | **4.36** |

- **Sonnet 4.6**: 가장 엄격한 judge (평균 3.92)
- **Sonnet 4**: 중간 (평균 4.31)
- **GPT-4o**: 가장 관대한 judge (평균 4.56)
- GPT-4o 자체 채점: **+1.06 인플레이션** (4.36 vs Claude judge 3.30) — 진짜 편향 우려는 여기

**결론**: Claude-as-judge 방법론은 방어 가능. 동일 모델 간 편향 없음. 순위 변동 없음.

---

## 7. DeepSeek-V3 3위 — 정당성 검증 (PASS)

| 근거 | 상세 |
|------|------|
| **난이도 반비례** | easy 4.11 → expert 4.46 (Sonnet 4.6과 동일 패턴, 깊은 추론 능력) |
| **완벽 점수** | 10개 (Haiku와 동일), 다양한 modality에 걸침 |
| **Flag 프로필** | hallucination 11 (Sonnet 4.6과 동일), novel_insight 62 |
| **Sonnet 4.6을 이긴 질문** | Q64: 4.00 vs 3.15, Q96: 4.45 vs 4.10, Q07: 5.00 vs 4.85 |
| **약점도 정상** | spatial 최저 (3.89), Q27/Q28 spatial 질문에서 모두와 동일하게 고전 |

---

## 8. Llama-3.3-70B 재현성 검증 (PASS)

동일 모델을 Groq/Together 두 provider로 실행:
- Overall: Groq 3.290 vs Together 3.304 (차이 0.013)
- Per-question 평균 절대 차이: 0.285
- Flag 일치율: 90.5% (181/200)
- **추론 랜덤성으로 인한 ~0.3점 노이즈** — 순위 해석 시 참고

---

## 9. Score Distribution

| Model | 1.0-2.0 | 2.0-3.0 | 3.0-3.5 | 3.5-4.0 | 4.0-4.5 | 4.5-5.0 |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Sonnet 4.6 | 0 | 0 | 1 | 5 | 26 | **68** |
| Haiku 4.5 | 0 | 0 | 2 | 18 | 34 | 45 |
| DeepSeek-V3 | 0 | 0 | 6 | 16 | 41 | 37 |
| Sonnet 4 | 0 | 2 | 9 | 33 | 36 | 20 |
| Gemini | **14** | 10 | 11 | 31 | 26 | 8 |
| Llama-Tg | 1 | 18 | **47** | 26 | 7 | 1 |
| GPT-4o | 0 | 22 | **39** | 33 | 6 | 0 |
| GPT-4o-mini | 0 | 30 | 29 | 34 | 6 | 1 |
| Llama-Gq | 0 | 26 | 36 | 26 | 12 | 0 |

- Sonnet 4.6: **ceiling effect** (68/100 질문이 4.5-5.0) — 향후 7점 척도 고려
- Gemini: 14개 질문이 1.0-2.0 → truncation 문제의 직접적 증거
- 하위 4개 모델: 3.0-3.5에 집중 (변별력 양호)

---

## 10. 개선 권장사항

| 우선순위 | 항목 | 설명 | 영향 |
|:--------:|------|------|------|
| **높음** | Gemini 재평가 | max_tokens 8192+로 재실행 | 순위 5위 → 4~5위 |
| **높음** | Q27/Q28/Q64 수정 | ground_truth_key_facts를 data_context와 일치 | 전 모델 hallucination/error flag 감소 |
| **중간** | Haiku Q10 보완 | JSON 파싱 에러 수정, 점수 3.35 반영 | 99→100 Q 완성 |
| **중간** | Novel insight 주석 | 92 vs 0 분포에 대한 해석 주석 추가 | 편향 오해 방지 |
| **낮음** | 7점 척도 | Ceiling effect 해소 (v3용) | 상위 모델 변별력 향상 |

---

## 11. 최종 판단

### 현재 상태로 v2.1 공식화: **보류** (MUST-FIX 3가지 수정 후 진행)

**수정 순서**:
1. Q27, Q28, Q64의 `ground_truth_key_facts` 수정 (`question_bank.json`)
2. Gemini 2.5 Flash `max_tokens` 확대 후 재실행 (`run_llm_evaluation.py`)
3. Haiku Q10 점수 보완
4. 수정된 ground truth로 전체 9개 모델 재채점 (`score_responses.py`)
5. 재채점 결과 반영하여 README 및 `llm_eval_summary.json` 업데이트

### 예상 결과
- **전체 순위 구조는 대체로 유지** (Q27/Q28/Q64 수정은 모든 모델에 동일 영향)
- Gemini는 5위 유지 또는 Sonnet 4에 근접 가능 (~3.72 추정)
- 전반적 hallucination/factual_error flag 수 감소 → 결과 신뢰도 향상
- **기존 벤치마크 리스트** (21 ML tasks, 100 questions, scoring schema)는 **변경 없이 보존**

---

## 부록: 검토에 사용된 파일

| 파일 | 용도 |
|------|------|
| `evaluation/llm/question_bank.json` | 100개 질문 원본 |
| `evaluation/llm/annotation_schema.json` | 5차원 채점 스키마 |
| `evaluation/llm/data_context/spatial.md` | Q27/Q28 검증 |
| `evaluation/llm/data_context/transcriptomics.md` | Q64 검증 |
| `results/scored_eval_*_judged-by-sonnet46.json` (9개) | 모델별 채점 결과 |
| `results/scored_eval_claude-sonnet-4-6_20260227_013541.json` | Cross-judge (Sonnet 4 judge) |
| `results/raw/eval_models_gemini-2.5-flash_*.json` | Gemini truncation 원본 검증 |
| `docs/samples/llm_eval_summary.json` | Cross-judge matrix |
