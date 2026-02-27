# SpaceOmicsBench v2 - Error/Hallucination Review

Date: 2026-02-27  
Scope: `SpaceOmicsBench/v2_public` 전체 점검 (코드, 문서, 데이터/결과 추적 정책)

## Summary

- 핵심 실행 경로는 정상 동작했다.
- 문서/운영 측면에서 "재현 근거 부족" 또는 "오해 유발 가능성"이 있는 지점이 확인되었다.
- 즉시 수정이 필요한 치명적 런타임 버그는 발견하지 못했다.

## Findings (Severity Ordered)

### 1) High - README의 LLM 성능 수치 재현 근거 부족

- `README.md`에 고정된 LLM 비교 점수(예: 4.55/4.76 등)가 기재되어 있음.
- 그러나 `results/`는 git에서 무시되고(`results/README.md`만 추적), 원 점수 산출 JSON/리포트는 저장소에서 추적되지 않음.
- 현재 저장소에 추적되는 샘플은 `docs/samples/result_sample_eval.json` 1개(원 eval 샘플)라 README의 비교 표를 직접 재현하기 어려움.

Risk:
- 외부 리뷰어가 점수 신뢰성을 검증하기 어렵다.
- 문서가 결과를 "단정"하는 인상으로 보일 수 있다.

### 2) Medium - `generate_tasks_and_splits.py` 설명과 실제 동작 간 괴리

- 파일 상단 설명은 최신 v2 생성기처럼 보이지만, 실제로는 legacy 안전장치가 걸려 있음(`--allow-legacy-write` 필요).
- 실수 실행 방지는 잘 되어 있으나, 파일 설명만 보면 유지보수자가 혼동할 수 있음.

Risk:
- 신규 기여자가 해당 스크립트를 공식 생성기로 오해할 수 있다.

### 3) Low - README의 `scripts/` 구조 설명이 최신 운영 스크립트를 충분히 반영하지 않음

- README의 디렉터리 설명은 전처리 스크립트 중심으로 보이며,
  현재 운영 핵심 스크립트(`generate_readme_tables.py`, `validate_tasks.py`)가 눈에 띄게 소개되지 않음.

Risk:
- 실무 사용자가 자동화/검증 스크립트 존재를 놓칠 수 있다.

## Functional Validation Performed

다음 검증은 `spaceomics` conda 환경에서 수행:

1. Task schema validation  
Command:
`conda run -n spaceomics python scripts/validate_tasks.py`  
Result:
`Validated 21 task files against task_schema.json`

2. Evaluation harness dry run  
Command:
`conda run -n spaceomics python evaluation/eval_harness.py --dry-run`  
Result:
21개 태스크 모두 OK, `All 21 tasks verified successfully.`

3. Syntax check  
Command:
`python -m py_compile ...` (핵심 Python 스크립트 대상)  
Result:
성공 (문법 오류 없음)

## Practicality Assessment

- 벤치마크 실행/평가 파이프라인은 현재 실사용 가능 상태.
- CI의 스키마/README 동기화 검증 추가로 운영 안정성은 개선됨.
- 남은 실용 리스크는 "LLM 결과 수치의 추적/재현 정책"에 집중되어 있음.

## Recommended Actions

1. README의 LLM 결과 섹션에 재현 경로 명시:
   - 점수 산출 커맨드
   - 입력 파일 패턴
   - judge 모델/버전
2. `docs/samples/`에 최소 1개 scored 파일(또는 집계 요약 JSON) 추가:
   - README 수치와 연결 가능한 근거 파일 제공
3. `generate_tasks_and_splits.py` 상단 docstring에 `LEGACY` 문구 명시:
   - 오해 가능성 제거
4. README의 Directory Structure에 운영 스크립트 2개 노출:
   - `scripts/generate_readme_tables.py`
   - `scripts/validate_tasks.py`

