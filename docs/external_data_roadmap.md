# External Data Collection Roadmap

**Created:** 2026-02-28
**Status:** Planning — revisit before starting data collection

---

## 1. 공개 데이터 현황 (Open Access)

### 자동 수집 가능 (스크립트화)

| 데이터 | OSD ID | 형식 | 비고 |
|---|---|---|---|
| I4 전체 (OSD-569~687) | 다수 | 커스텀 xlsx | **v2에 이미 포함** |
| JAXA CFE cfRNA | OSD-530 | 커스텀 xlsx | **v2에 이미 포함** |
| GeneLab 표준 처리 데이터 | 미션별 | `differential_expression_GLbulkRNAseq.csv` | **Module 1 타깃** |
| GEO 마우스 데이터 | GSE213808 | CSV/txt | 스크립트 가능 |

### 수동 작업 필요

| 데이터 | 이유 |
|---|---|
| Nature 논문 supplementary (I4 papers P1-P10) | 페이지별 다운로드, 형식 불균일 |
| SOMA Browser 원시 데이터 | JS 렌더링, 직접 접속 필요 |
| Axiom-1/2, Polaris Dawn | OSDR 등록 여부/형식 직접 확인 필요 |
| NASA Twins 개인 데이터 | CONTROLLED — dbGaP/LSDA 신청 필요 |

---

## 2. 핵심 제약사항

- **I4는 GeneLab 표준 파이프라인이 아님**: 커스텀 파이프라인 → `GLbulkRNAseq.csv` 없음
- **신규 미션이 GeneLab 표준 파이프라인 사용 시** → `ingest_osdr.py` 자동화 직접 가능
- **신규 미션이 커스텀 형식 사용 시** → 수동 파싱 레이어 필요
- **OSDR 검색 UI**: JavaScript 렌더링 → WebFetch로 API 조회 불가, 직접 브라우징 필요

---

## 3. 별도 벤치마크 아키텍처 (권고)

```
SpaceOmicsBench/
├── v2_public/              ← 현재 (FROZEN — 건드리지 않음)
│   ├── tasks/              # 21 ML tasks
│   ├── evaluation/
│   └── README.md
│
└── missions/               ← 신규 (별도 벤치마크)
    ├── README.md
    ├── schema/             # task 스키마 (v2 호환)
    ├── scripts/
    │   └── ingest_osdr.py  # Module 1 (OSDR → 벤치마크 형식)
    ├── axiom_2/
    │   ├── data/processed/
    │   ├── tasks/
    │   └── results/
    └── polaris_dawn/
        └── ...
```

별도 레포 전환 기준: 미션 수 ≥ 3개 또는 태스크 수 ≥ 10개 시 고려

---

## 4. 수동 vs 자동 분류

| 자동화 가능 | 수동 필요 |
|---|---|
| OSDR API 파일 목록 조회 | 논문 supplementary 다운로드 |
| GeneLab 표준 CSV 파싱 | 새 미션 형식 확인 및 매핑 |
| Jaccard/방향성/Spearman 계산 | Task feasibility 결정 (N 충분?) |
| Hypergeometric 검증 | Gene alias → HGNC 변환 검수 |
| Markdown/JSON 리포트 생성 | 생물학적 해석 검토 |
| baseline_results 업데이트 | IRB/데이터 접근 신청 |

---

## 5. 진행 전 확인 사항 (TODO)

- [ ] OSDR에서 Axiom-1/2 직접 검색: 어떤 OSD ID? GeneLab 표준 처리 파일 있는가?
- [ ] Polaris Dawn 데이터 공개 여부 확인 (2025년 10월 비행)
- [ ] SOMA 원시 데이터 다운로드 방법 확인 (soma.weill.cornell.edu)
- [ ] missions/ 디렉토리 생성 전 새 미션 데이터 샘플 1개 수동 처리로 파이프라인 검증

---

## 6. 참고 데이터 접근 방법 (검증된 방법)

```python
import requests

# OSDR Biodata API — 파일 목록 조회
BASE = "https://visualization.osdr.nasa.gov/biodata/api"
url = f"{BASE}/v2/dataset/OSD-530/files/"
files = requests.get(url).json()["OSD-530"]["files"]

# GEODE 다운로드 (S3 리다이렉트)
url = "https://osdr.nasa.gov/geode-py/ws/studies/OSD-530/download"
params = {"source": "datamanager", "file": "FILENAME.xlsx"}
resp = requests.get(url, params=params, allow_redirects=True, stream=True)

# AWS S3 직접 (인증 불필요)
# aws s3 sync s3://nasa-osdr/OSD-NNN/ ./data/ --no-sign-request \
#     --exclude "*" --include "*.xlsx" --include "*.csv"
```
