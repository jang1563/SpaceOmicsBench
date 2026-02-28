# SpaceOmicsBench v3 — Design Plan

**Created:** 2026-02-28
**Status:** Planning — pending OSDR data verification

---

## 1. v2 현황 및 약점 분석

### v2 강점
- 실제 우주비행 오믹스 데이터 (I4 미션 기반, 4 크루)
- 17개 ML 태스크, 100개 LLM 평가 질문
- 7개 오믹스 모달리티 커버리지
- Claude-as-Judge 자동 평가 파이프라인 구현

### v2 핵심 약점

| 약점 | 원인 | 심각도 |
|------|------|--------|
| 샘플 수 부족 (n=4 crew, n=28 임상) | I4 미션 자체 제약 | 높음 |
| 통계 신뢰도 부족 (LLM 100Q) | 질문 수 부족, 1회 평가 | 높음 |
| 단일 미션 편향 (I4 only) | 데이터 소스 제한 | 중간 |
| Foundation model 평가 없음 | Track 부재 | 중간 |
| HuggingFace 통합 없음 | 배포 인프라 미구축 | 낮음 |

---

## 2. v3 핵심 목표

1. **다중 미션 데이터** 통합 (Axiom, Polaris, SOMA 포함)
2. **LLM 평가 100Q → 500Q** 확장 + 난이도 분층 강화
3. **Foundation Model Track** 신설 (Track C)
4. **HuggingFace Dataset + Leaderboard** 배포
5. **재현성**: Docker + GitHub Actions

---

## 3. 데이터 확장 계획

### 3.1 후보 미션 및 데이터 가용성 (2026-02-28 OSDR 조사 기준)

| 미션 | 기간 | OSD ID | 오믹스 | N | 가용성 | v3 포함 여부 |
|------|------|--------|--------|---|--------|------------|
| I4 (현재) | 2021-2022 | OSD-569~687 | 전체 7종 | n=4 crew | Open | **v2에 포함** |
| JAXA CFE | 2019-2020 | OSD-530 | cfRNA (그룹-level) | n=4 | Open | **v2에 포함** |
| Twins Study | 2015-2016 | OSD-53 | 마이크로어레이 | n=1 twin pair | Open | 형식 불일치 |
| Axiom-1 (AX-1) | 2022 | 미등록 | — | — | **OSDR 없음** | 대기 |
| Axiom-2 (AX-2) | 2023 | 미등록 | — | — | **OSDR 없음** | 대기 |
| Polaris Dawn | 2024-09 | 미등록 | — | — | **처리 중 추정** | 대기 |

**OSDR 조사 결과 (2026-02-28)**:
- SOMA Nature 2024 논문의 2,911 샘플 = **I4 전용**. Axiom/Polaris는 미래 계획.
- OSD 688~900 스캔: 인간 우주비행 관련 GeneLab DE 파일 없음.
- NASA Twins Study (OSD-53): 마이크로어레이 형식, GLbulkRNAseq 파이프라인 미적용.
- **결론**: 현시점에서 I4/JAXA 외 신규 미션 데이터 자동 수집 불가.

### 3.2 현실적 v3 데이터 전략

**단기 (2026 Q2-Q3, 현재 가능)**:
1. **I4 추가 분석** (v2 데이터 재활용):
   - 시간 연속 패턴 분석 태스크 (7 timepoint trajectory)
   - 미탐구 모달리티 교차 예측 태스크
2. **GEO 마우스 데이터** (GSE213808): 스크립트 자동 수집 가능
3. **LLM 100Q → 500Q**: 기존 데이터로 질문 확장 (가장 실현 가능)

**중기 (Axiom/Polaris OSDR 등록 후)**:
- `ingest_osdr.py` 직접 적용 (GeneLab 표준 형식이면 자동화)
- 신규 미션 크루 n=4~10 추가 → sample-level 태스크 확장

**v3 타깃 코호트 (현실적)**:
- Feature-level 태스크: I4 기존 (B1-E4 확장) + 마우스 추가 (GEO)
- Sample-level 태스크: I4 n=4 유지 (F-series) + 추후 확장
- LLM Track B: 500Q (주요 성과물)

---

## 4. v3 3-Track 구조

### Track A: ML Benchmark (확장)

v2 17개 태스크 유지 + 신규 Category I 확장:

| 추가 태스크 | 설명 | 데이터 |
|------------|------|-------|
| I1_expand | cfRNA cross-mission DEG prediction | SOMA multi-mission cfRNA |
| I2_expand | Proteomics cross-mission DE | SOMA proteomics |
| I3_microbiome | Microbiome cross-mission body site | Axiom/Polaris 있는 경우 |
| I4_foundation | Foundation model embedding → phase clf | 임베딩 기반 (Track C 연계) |

**n 증가 전략**: SOMA 통합 시 feature-level n이 동일하지만 크루 수가 I4 n=4 → n=23으로 증가
→ Sample-level 태스크도 신뢰도 향상 가능

### Track B: LLM Evaluation (확장)

| 항목 | v2 | v3 |
|------|-----|-----|
| 총 질문 수 | 100 | 500 |
| 난이도 분포 | 25/25/25/25 | 100/150/150/100 (easy/medium/hard/expert) |
| 미션 커버리지 | I4, Twins, JAXA | + Axiom, Polaris, SOMA |
| 평가 반복 | 1회 | 3회 (통계적 신뢰도) |
| 평가자 모델 | claude-sonnet-4-6 | 다중 judge (claude + gemini) |
| 자동 HF 업로드 | 없음 | 있음 |

**500Q 분류 계획:**

```
Track B 질문 구조 (500Q):
├── 우주 생물학 기반 (100Q): 방사선, 미세중력, 격리 메커니즘
├── I4 데이터 해석 (120Q): 기존 v2 100Q + 20Q 추가
├── 신규 미션 해석 (120Q): Axiom/Polaris/SOMA 데이터 기반
├── 교차 미션 비교 (100Q): 미션 간 패턴 비교, 메타분석
└── 임상 응용 (60Q): 대응 조치, 건강 모니터링, 위험 평가
```

### Track C: Foundation Model Evaluation (신규)

**목적**: 생물의학 기반모델이 우주 오믹스에 얼마나 잘 전이되는가?

**평가 태스크:**

| 태스크 | 입력 | 출력 | 목표 |
|--------|------|------|------|
| Zero-shot DE prediction | Gene sequence + expression | Flight DE status | 사전학습 지식 활용 |
| Gene embedding quality | Gene embeddings (ESM2 등) | Spaceflight signature overlap | 임베딩 공간 평가 |
| Cross-modal transfer | cfRNA embedding → protein prediction | Cross-modal concordance | 모달리티 전이 |
| Few-shot adaptation | 5-shot spaceflight examples → new mission | New mission DE | 소수 적응 능력 |

**평가 모델 후보 (초기):**
- ESM2 (단백질 임베딩)
- Enformer / Borzoi (유전체)
- GeneFormer / scGPT (단세포 RNA)
- BioGPT / BioBERT (텍스트 기반)
- OpenBioLLM (의학 특화 LLM)

---

## 5. 인프라 계획

### 5.1 HuggingFace 통합

```
HuggingFace Hub:
├── Dataset: spaceomicsbench/v3-tasks
│   ├── train/test splits (Parquet)
│   ├── task configs (JSON)
│   └── README (데이터카드)
├── Dataset: spaceomicsbench/v3-llm-questions
│   ├── questions_500.jsonl
│   └── reference_answers_500.jsonl
└── Leaderboard Space: spaceomicsbench/leaderboard
    ├── ML Track A results
    ├── LLM Track B results
    └── Foundation Model Track C results
```

### 5.2 재현성

```
재현성 레이어:
├── Docker: spaceomicsbench/v3:latest
│   ├── conda env (spaceomics)
│   ├── 모든 평가 스크립트
│   └── 기준선 모델 (RandomForest, LogReg, XGBoost)
├── GitHub Actions
│   ├── 새 제출 → 자동 평가 → HF 업로드
│   └── 주간 리더보드 갱신
└── Zenodo DOI
    └── 데이터 스냅샷 고정 버전
```

### 5.3 디렉토리 구조

```
SpaceOmicsBench/
├── v2_public/              ← FROZEN (현재)
├── v3/                     ← 신규
│   ├── data/
│   │   ├── processed/      # gitignored
│   │   └── splits/         # tracked
│   ├── tasks/
│   │   ├── track_a/        # ML tasks (JSON)
│   │   ├── track_b/        # LLM questions (JSONL)
│   │   └── track_c/        # Foundation model tasks (JSON)
│   ├── evaluation/
│   │   ├── score_ml.py
│   │   ├── score_llm.py
│   │   └── score_foundation.py
│   ├── baselines/
│   │   └── run_baselines.py
│   ├── scripts/
│   │   ├── preprocess/
│   │   └── build_questions.py
│   ├── results/
│   │   └── leaderboard/
│   ├── docs/
│   └── README.md
└── missions/               ← 기존 (extension)
```

---

## 6. 로드맵 (2026)

### Phase 1: 데이터 검증 및 확장 (Q1 2026, 현재)

- [ ] OSDR Axiom-1/2 데이터 직접 검색 및 OSD ID 확인
- [ ] Polaris Dawn 공개 여부 확인
- [ ] SOMA 논문 보조 데이터에서 샘플 메타데이터 추출
- [ ] `ingest_osdr.py`로 Axiom/Polaris DE 데이터 수집 테스트
- [ ] v2 100Q → 추가 질문 초안 작성 (목표: 200Q → 500Q)

### Phase 2: v3 태스크 설계 (Q2 2026)

- [ ] Track A: 신규 Category I 태스크 정의 (v2 패턴 따름)
- [ ] Track B: 500Q 작성 및 검증 (참조 답변 포함)
- [ ] Track C: Foundation model 평가 태스크 프로토타입
- [ ] HuggingFace 데이터카드 작성

### Phase 3: 구현 및 기준선 (Q3 2026)

- [ ] 전처리 스크립트 (Phase 1 v2 패턴 확장)
- [ ] 기준선 결과 생성 (RandomForest, Logistic Regression 등)
- [ ] 초기 Foundation model 평가 (ESM2, GeneFormer)
- [ ] Docker 이미지 빌드

### Phase 4: 배포 및 논문 (Q4 2026)

- [ ] HuggingFace 공개 배포
- [ ] GitHub Actions 리더보드 자동화
- [ ] arXiv 사전 공개 (Nature Scientific Data 투고 목표)
- [ ] 커뮤니티 공개 (bioRxiv, Twitter/X)

---

## 7. 단기 액션 (즉시 실행 가능)

우선순위 순:

1. **LLM 100Q → 200Q 확장** *(즉시 가능)*:
   - 기존 100Q 패턴 유지 + 교차 미션 비교/생물학적 메커니즘 질문 추가
   - Track B 확장의 가장 빠른 성과물
2. **GEO 마우스 데이터 수집** *(즉시 가능)*:
   - GSE213808 (쥐 우주비행 RNA-seq) → ingest_osdr.py 호환 형식으로 변환
   - Category I (cross-species prediction) 태스크 추가
3. **HuggingFace 데이터카드 작성** *(즉시 가능)*:
   - v2 기존 데이터셋을 HF에 업로드 (tasks/, splits/ 형식 변환)
4. **Foundation Model Track C 프로토타입** *(즉시 가능)*:
   - ESM2로 spaceflight DE 예측 1개 태스크 구현
   - GeneFormer로 phase classification 시도
5. **Polaris Dawn OSDR 모니터링**: 2026 Q3~Q4 등록 예상, 주기적 확인

~~1. OSDR Axiom 검색~~ → 완료. **결론: 현시점 데이터 없음** (external_data_roadmap.md 참조)

---

## 8. 학술 가치 평가

### v3가 v2 대비 개선하는 학술 기여

| 기준 | v2 | v3 |
|------|-----|-----|
| 코호트 크기 | n=4 crew | n=23 crew (예상) |
| 미션 수 | 3 (I4/Twins/JAXA) | 6+ (+ Axiom-1/2, Polaris) |
| 태스크 수 | 17 ML + 100 LLM | 30+ ML + 500 LLM + Foundation |
| 통계 신뢰도 | 낮음 | 높음 (반복 평가, CI) |
| 배포 접근성 | GitHub만 | HuggingFace + GitHub + Docker |
| 타깃 저널 | Workshop/arXiv | Nature Scientific Data |

### 주요 학술 포지셔닝

1. **"우주 오믹스 최초 표준 벤치마크"**: 아직 이 분야에 공식 벤치마크 없음
2. **LLM-for-Science 평가 도구**: LLM 도메인 지식 평가 프레임워크
3. **Foundation model 전이 학습 연구**: 지구 데이터 사전학습 → 우주 환경 전이

---

## 9. 위험 요소 및 대응

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| Axiom/Polaris 데이터 미공개 | 중간 | 높음 | SOMA 통합만으로도 Track A/B 가능 |
| Foundation model API 비용 | 높음 | 중간 | 오픈소스 모델 우선 (HuggingFace inference) |
| n=23으로 여전히 통계적 한계 | 높음 | 중간 | Feature-level 태스크에 집중 (n=수천~수만 유전자) |
| HuggingFace 자동화 복잡도 | 중간 | 낮음 | Phase 4로 미루고 v3 초기는 GitHub만 |

---

*다음 단계: OSDR Axiom 검색 결과 → 이 문서 §3.1 업데이트*
