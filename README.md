# LAMBO

LAMBO는 `Loong` 데이터셋 위에서 동작하는 long-context, anchor-based, iterative document navigation 실험 코드입니다.  
이 저장소의 핵심 구현은 `script/anchor` 아래에 있으며, `golden retrieve`처럼 미리 정답 chunk를 주는 대신 문서별 anchor를 만들고, summary만 본 뒤 좌표를 선택해 실제 paragraph/table chunk를 점진적으로 확인하는 방식으로 설계되어 있습니다.

## 핵심 컨셉

이 구현은 문서마다 아래 루프를 따릅니다.

1. `Anchor Agent`
   - 입력: `question + instruction + one whole document`
   - 역할: 문서를 의미적으로 이어지는 구간 단위의 anchor 좌표로 나누고, 각 anchor summary를 생성
   - 중요한 점: 문서 하나당 한 번의 anchor 생성 호출을 사용하며, 미리 rule-based로 chunk를 잘라서 anchor를 만드는 방식이 아닙니다.

2. `Search R1`
   - 입력: `question + instruction + doc title + anchor summaries`
   - 역할: 아직 원문 paragraph는 보지 않고, 어떤 anchor 좌표를 다음에 확인해야 하는지 결정
   - 출력 태그:
     - `<think>...</think>`
     - `<search>{"next_anchor_id": ...}</search>`

3. `Refine Extractor`
   - 입력: 선택된 단일 anchor chunk
   - 역할: 해당 chunk만 보고 gold evidence를 원자 단위로 추출
   - 출력 태그:
     - `<think>...</think>`
     - `<retrieve>{"items": [...], "enough": ..., "followup_query": ...}</retrieve>`

4. Iterative loop
   - `enough=false`이면 다시 `<think> -> <search> -> <info> -> <think> -> <retrieve>`를 반복
   - `enough=true`이면 해당 문서에 대한 탐색 종료

중간 trace는 아래 형식으로 저장됩니다.

```text
<think>이 문서에서 무엇을 봐야 하는지 판단</think>
<search>{"next_anchor_id":"DOC1_A5", ...}</search>
<info>{"anchor_id":"DOC1_A5", "text":"...", ...}</info>
<think>이 chunk가 무엇을 주는지 판단</think>
<retrieve>{"items":[...], "enough":true, "followup_query":""}</retrieve>
```

## 디렉토리 구조

```text
.
├── Agentic-R/                # Search-R1 / Agentic-R 참고 코드
├── Loong/                    # 원본 benchmark 및 평가 코드
├── logs/                     # 실험 산출물
└── script/anchor/            # LAMBO 구현
```

`script/anchor`의 주요 파일은 아래와 같습니다.

- `manifest.py`: set1 고정 10개 인덱스 manifest 생성
- `anchor_agent.py`: whole-doc anchor 생성 + summary 생성
- `search_r1.py`: summary 기반 iterative anchor 선택
- `refine_extractor.py`: 선택된 chunk만 보고 evidence 추출
- `answer_writer.py`: 문서별 evidence를 최종 정답 형식으로 집계
- `evaluate_structured.py`: structured EM / pair-F1 계산
- `run_lambo_set1.py`: end-to-end 실행 엔트리포인트
- `run_loong_judge_local.py`: Loong judge 프롬프트를 로컬 Qwen으로 실행

## 데이터와 환경

기본 입력 데이터는 아래 파일만 사용합니다.

- `Loong/data/loong_process.jsonl`

문서 입력은 외부 `doc_path`가 아니라 각 샘플의 `docs` 필드만 사용합니다.

GitHub 업로드 시에는 `Loong/data`를 포함하지 않습니다.  
즉, 저장소에는 코드와 로그만 올리고, 데이터셋 파일(`loong.jsonl`, `loong_process.jsonl`, `Loong/data/doc/*`)은 로컬에서만 유지합니다.

기본 실행 환경은 아래를 가정합니다.

- Python venv: `/workspace/StructRAG/venv`
- 모델: `/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct`
- `PYTHONPATH=/workspace/lambo`

## 고정 실험 셋

set1의 실제 존재 조합 10개만 사용합니다.

| idx | type | level |
|---|---|---|
| 914 | financial | 1 |
| 725 | legal | 1 |
| 1065 | financial | 2 |
| 801 | legal | 2 |
| 1280 | financial | 3 |
| 469 | legal | 3 |
| 63 | paper | 3 |
| 1504 | financial | 4 |
| 596 | legal | 4 |
| 32 | paper | 4 |

## 실행 방법

### 1. Structured run

```bash
cd /workspace/lambo
PYTHONPATH=/workspace/lambo /workspace/StructRAG/venv/bin/python \
  /workspace/lambo/script/anchor/run_lambo_set1.py \
  --skip_judge \
  --output_dir /workspace/lambo/logs/lambo_set1_10_doccall
```

주요 옵션:

- `--force`: 기존 cache 무시 후 재실행
- `--max_items N`: 일부 샘플만 smoke test
- `--disable_llm_search`: Search R1의 LLM planning 비활성화

### 2. Local Loong judge

원본 `run_lambo_set1.py`의 judge 경로는 `Loong/src/step3_model_evaluate.py`를 호출하며, 기본 설정상 외부 judge endpoint를 사용합니다. 이 경로가 멈추는 경우를 대비해 동일 judge 프롬프트를 로컬 Qwen으로 실행하는 스크립트를 추가했습니다.

```bash
cd /workspace/lambo
PYTHONPATH=/workspace/lambo /workspace/StructRAG/venv/bin/python \
  /workspace/lambo/script/anchor/run_loong_judge_local.py
```

## GitHub 업로드

이 저장소는 원격 repo를 덮어쓰는 방식으로 업로드하는 것을 전제로 정리되어 있습니다.  
`/workspace/lambo`는 직접 git repo가 아니므로, 보통은 원격 repo를 새로 clone한 뒤 현재 작업물을 덮어씌우는 방식이 가장 안전합니다.

```bash
cd /workspace
git clone https://github.com/leadawon/LAMBO lambo_upload
rsync -av --delete --exclude '.git' /workspace/lambo/ /workspace/lambo_upload/
cd /workspace/lambo_upload
git status
git add .
git commit -m "Overwrite repo with latest LAMBO code and experiment report"
git push origin master
```

이때 `.gitignore`에 의해 `Loong/data`는 stage되지 않습니다.

## 출력 파일

기본 산출물은 `logs/lambo_set1_10_doccall` 아래에 저장됩니다.

- `manifest.json`: 고정 10개 manifest
- `lambo_predictions.jsonl`: 최종 예측
- `reports/structured_eval.json`: structured metric 요약
- `reports/summary_report.md`: 구조적 결과 요약
- `loong_judge_eval_local_qwen32b.jsonl`: local judge 원문
- `reports/loong_judge_local_qwen32b.json`: local judge 요약
- `samples/<sample_id>/anchors.json`: 문서별 anchor 결과
- `samples/<sample_id>/DOC*_search.json`: search 단계 캐시
- `samples/<sample_id>/DOC*_refine.json`: iterative trace와 retrieve 결과
- `samples/<sample_id>/answer_writer.json`: 최종 정답 조립 결과

## 구현 로직 요약

### Anchor generation

- `anchor_agent.py`는 `record["docs"]`를 문서 단위로 분리합니다.
- 각 문서의 원문 전체를 unit 목록으로 정리한 뒤, 문서 하나당 한 번의 LLM 호출로 contiguous anchor span을 생성합니다.
- anchor schema:

```json
{
  "anchor_id": "DOC1_A5",
  "doc_id": "DOC1",
  "doc_title": "《2024年一季度报告》",
  "order": 5,
  "section_path": "四、季度财务报表",
  "packet_span": "U120..U165",
  "char_span": [12345, 18222],
  "anchor_type": "table_region",
  "text": "...",
  "summary": "...",
  "prev_anchor_id": "DOC1_A4",
  "next_anchor_id": ""
}
```

### Search R1

- `search_r1.py`는 raw chunk를 먼저 보지 않습니다.
- 입력은 `question`, `instruction`, `doc_title`, `anchor summaries`, `previous_trace`, `known_items`입니다.
- ranking은 rule-based prior와 lexical overlap을 함께 사용합니다.
- 이후 LLM planning이 켜져 있으면 shortlist anchor summary만 보고 `<think>`와 `<search>`를 생성합니다.

### Refine Extractor

- `refine_extractor.py`는 현재 라운드에서 선택된 anchor 하나만 읽습니다.
- 이 단계에서 `<info>`는 orchestration 코드가 삽입하고, 모델은 `<think>`와 `<retrieve>`만 생성합니다.
- `<retrieve>`는 `items`, `enough`, `followup_query`를 포함합니다.
- 정보가 불충분하면 follow-up query를 바탕으로 다음 anchor를 다시 선택합니다.

### Final answer writing

- `answer_writer.py`는 문서별 추출 evidence만 모아 최종 답을 생성합니다.
- 전체 문서 원문을 다시 읽지 않습니다.
- benchmark task type에 따라 string / dict / list topology를 강제합니다.

## 실험 결과

실험 결과 경로:

- `logs/lambo_set1_10_doccall/reports/structured_eval.json`
- `logs/lambo_set1_10_doccall/reports/loong_judge_local_qwen32b.json`

### Structured metrics

| metric | value |
|---|---:|
| sample_count | 10 |
| ACC | 0.10 |
| EM | 0.10 |
| avg_pair_f1 | 0.3126 |

여기서 `ACC`는 sample-level exact match 기준으로 계산되어 `EM`과 동일합니다.

### Loong judge

아래 수치는 공식 GPT-4 judge가 아니라, 같은 Loong judge 프롬프트를 로컬 `Qwen2.5-32B-Instruct`에 적용한 결과입니다.

| metric | value |
|---|---:|
| sample_count | 10 |
| scoring_success_rate | 1.00 |
| avg_score | 58.5 |
| perfect_rate | 0.30 |

### Per-sample summary

| idx | type | level | EM | pair_f1 | judge |
|---|---|---:|---:|---:|---:|
| 914 | financial | 1 | 1 | - | 100 |
| 725 | legal | 1 | 0 | - | 100 |
| 1065 | financial | 2 | 0 | - | 95 |
| 801 | legal | 2 | 0 | - | 0 |
| 1280 | financial | 3 | 0 | 0.0000 | 20 |
| 469 | legal | 3 | 0 | 0.7059 | 80 |
| 63 | paper | 3 | 0 | 0.0000 | 20 |
| 1504 | financial | 4 | 0 | - | 100 |
| 596 | legal | 4 | 0 | 0.8571 | 70 |
| 32 | paper | 4 | 0 | 0.0000 | 0 |

## 결과 해석

현재 점수는 두 가지 현상이 섞여 있습니다.

1. 의미상 거의 맞았지만 strict EM에서는 틀린 경우
   - `725`: `《判决文书4》` vs `判决文书4`
   - `1065`: `楚天科技` vs `楚天科技股份有限公司`
   - `1504`: 수치와 추세는 맞지만 gold 문장과 서술 형식이 다름

2. 실제 추출 또는 최종 schema 조립이 실패한 경우
   - `801`: 문서 제목 목록 대신 긴 원문 조각이 최종 답으로 합쳐짐
   - `1280`: category-to-company dict 대신 anchor id가 key로 남음
   - `63`, `32`: paper 과제에서 문서 간 citation graph를 제대로 닫지 못함

## 현재 한계

- `paper` 계열은 문서별 local evidence만으로는 부족하고, document-level relation graph를 한 번 더 조립해야 합니다.
- `answer_writer` fallback이 일부 legal/financial task에서 overly permissive하게 문자열을 이어붙일 수 있습니다.
- 공식 Loong judge 경로는 외부 endpoint 의존성이 있어 재현성이 떨어집니다.
- structured EM은 매우 엄격해서, 표기 차이와 서술 차이까지 모두 오답으로 처리됩니다.

## 참고 코드

- `Agentic-R/`: Search-R1 iterative search 로직 참고
- `Loong/`: benchmark 데이터와 원본 judge/eval 코드
- `script/anchor/`: 현재 LAMBO 실험 구현
