# dawonv6

`dawonv6`는 `dawonv5`를 기반으로 **v5에서 설계만 하고 실제로는 실행되지 않았던 enrichment 코드를 실제로 실행되게 만든** 버전입니다. v5의 anchor schema / citation graph / summary renderer 는 그대로 재사용하되, runner 배선(wiring)과 exper99 실행 스크립트의 결정성(determinism)을 바로잡았습니다.

---

## v5에서 발견된 문제

### 1. (치명) dawonv5 anchor 코드가 실제로는 한 번도 실행되지 않았다

`dawonv5/run_set1_10.py` 와 `dawonv5/anchor/run_lambo_set1.py` 모두 다음과 같이 **`dawonv4.anchor.*` 를 import** 하고 있었습니다.

```python
# dawonv5/run_set1_10.py
from dawonv4.anchor.run_lambo_set1 import main

# dawonv5/anchor/run_lambo_set1.py
from dawonv4.anchor.anchor_agent import AnchorAgent
from dawonv4.anchor.search_agent import ExtractAgent, SearchAgent
from dawonv4.anchor.relation_refiner import RelationRefiner
...
```

결과적으로 v5 에서 추가한

- `anchor_schemas.AnchorEnrichedMetadata`
- `anchor_context_enricher.enrich_anchors`
- `anchor_citation_graph.build_citation_graph`
- `anchor_summary_renderer.render_summaries_for_doc`
- `anchor_agent.py` 의 v5 enrichment 호출 블록
- `search_r1.py` 의 v5 semantic_role 보너스 / rendered_summary 기반 chunk text
- `refine_extractor.py` 의 `_info_payload` 에 `semantic_role / owner_name / citation_summary` 포함

등은 **실제 exper99 실행 경로에 전혀 포함되지 않았습니다.** 다시 말해 v5 의 실행 결과는 v4 baseline 의 결과와 동일했고, v5 design report 의 downstream hook 는 전부 dead code 상태였습니다.

### 2. run_exper99.sh 가 full dataset 을 보낸 뒤 index 필터에만 의존

```bash
# dawonv5/run_exper99.sh (원본)
exec bash run.sh \
  --input_path "${INPUT_PATH}" \          # ← 전체 loong_process.jsonl
  --selected_indices_path "${INDICES_PATH}"
```

index filter 자체는 작동했지만 (manifest.json 에 99 개 entry 생성됨), StructRAG 비교 스크립트
`/workspace/StructRAG/scripts/32b/run_inference_exper99.sh` 와 **입력 경로가 달라지면서** 공정비교를 위한 안전 장치가 약했습니다. StructRAG 쪽은 balanced99 subset 파일 자체를 `--eval_data_path` 로 직접 넣습니다.

### 3. stale한 v4 문서(README.md / TODO_validation.md) 가 혼동을 유발

v5 루트에 dawonv4 이름이 박힌 옛 문서가 남아 있어서 v5 가 "사실상 v4 와 같은 동작" 을 한다는 사실이 드러나지 않았습니다.

---

## v6 에서 바뀐 것

### A. Runner 배선을 v6 자체로 돌림 (치명 버그 수정)

다음 파일의 `dawonv4.anchor.*` import 를 모두 `dawonv6.anchor.*` 로 교체했습니다.

- `run_set1_10.py`
- `anchor/run_lambo_set1.py` (10 라인)
- `anchor/run_loong_judge_local.py`

이로써 `anchor_agent.py` 안의 enrichment 블록이 실제로 호출되고, anchor record 에 `v5_metadata`, `rendered_summary` 가 부착되며, `search_r1._score_anchor` 의 semantic_role 보너스와 `refine_extractor._info_payload` 의 citation_summary 가 실제 파이프라인에서 효력을 가집니다.

검증:

```bash
/workspace/venvs/ragteamvenv/bin/python -c "
import sys; sys.path.insert(0,'/workspace/LAMBO')
from dawonv6.anchor.run_lambo_set1 import main
from dawonv6.anchor import anchor_schemas, anchor_context_enricher, anchor_citation_graph, anchor_summary_renderer
print('v6 wiring OK')"
```

### B. run_exper99.sh 를 StructRAG-style 로 변경 (99-sample 고정)

```bash
# dawonv6/run_exper99.sh
export LAMBO_DAWONV4_INPUT_PATH="${SUBSET_PATH}"
exec bash run.sh \
  --input_path "${SUBSET_PATH}" \
  "$@"
```

- `prepare_exper99_subset.py` 가 만든 `loong_set1_balanced99.jsonl` 을 그대로 `--input_path` 로 넘깁니다.
- runner 내부의 index-filter 로직에 의존하지 않고, 입력 파일 자체를 99개로 제한하므로 이후 runner 구현이 바뀌어도 항상 99 개만 처리됩니다.
- StructRAG `run_inference_exper99.sh` 와 동일한 패턴이 되어 공정비교 조건이 보장됩니다.

### C. 문서 정리

- v4 이름이 박힌 `README.md`, `TODO_validation.md` 제거.
- v5 의 design report (`dawonv5_design_report.md`), summary examples (`dawonv5_summary_examples.md`), validation todo (`dawonv5_validation_todo.md`) 는 schema 자체가 그대로 유지되므로 그대로 둠 (참고 문서).
- 이 README.md 가 v6 의 단일 진입 문서.

### D. v5 의 실행 로그 / 결과 파일은 제외

`dawonv5/logs/`, `dawonv5/data/` 는 v5 실행 잔여물이므로 v6 에는 복사하지 않았습니다. v6 처음 실행 시 `run_exper99.sh` 가 `dawonv6/data/loong_set1_balanced99*.{jsonl,json}` 를 다시 생성합니다.

---

## 유지된 것 (v5 와 동일)

- `anchor/anchor_schemas.py` — `AnchorEnrichedMetadata`, `CitationEdge` dataclass
- `anchor/anchor_context_enricher.py` — owner / role / content_type / time / unit 추출 (rule-based, LLM-free 동작 보장)
- `anchor/anchor_citation_graph.py` — explicit citation + structural adjacency edge 추출
- `anchor/anchor_summary_renderer.py` — ≤300자 rendered_summary
- `anchor/anchor_agent.py` 의 enrichment 호출 블록
- `anchor/search_r1.py` 의 v5 scoring hook
- `anchor/refine_extractor.py` 의 enriched `_info_payload`
- 모든 backward-compatibility guard (`isinstance(v5, dict)`, `.get()` fallback) — v5_metadata 가 없는 anchor 는 v4 동작 그대로

---

## 사용법

### 99-sample 실행 (StructRAG 공정비교용)

```bash
bash /workspace/LAMBO/dawonv6/run_exper99.sh
```

출력:

- `dawonv6/data/loong_set1_balanced99.jsonl` — 99-sample subset
- `dawonv6/logs/lambo_set1_exper99/` — manifest, samples/, reports/
- `dawonv6/logs/backend_server_exper99.log` — vLLM server log

### 일반 실행 (전체 set1)

```bash
bash /workspace/LAMBO/dawonv6/run.sh
```

### 환경 변수

v5 와 동일하게 `LAMBO_DAWONV4_*` prefix 를 사용합니다 (`run.sh` 의 vLLM health-check 가 이 이름에 묶여 있어 유지). OUTPUT_DIR / INPUT_PATH 는 자동으로 `dawonv6/` 하위를 가리킵니다.

---

## 검증 체크리스트 (v6 specific)

- [x] `dawonv6.anchor.*` import path 로 runner 가 동작한다
- [x] v5 의 `dawonv4.anchor.*` 참조가 runner/anchor 패키지에서 전부 제거됨 (`grep -rn 'from dawonv4\\.anchor' dawonv6/` 가 실행 경로에 없어야 함)
- [x] `run_exper99.sh` 가 `SUBSET_PATH` 를 `--input_path` 로 넘긴다
- [ ] 실제 실행 후 `logs/lambo_set1_exper99/samples/*/anchor.json` 에 `v5_metadata` / `rendered_summary` 필드가 들어있는지 확인 (v5 에서는 비어 있었음)
- [ ] `search_r1` 디버그 출력에서 v5 semantic_role 보너스가 score 에 반영되는지 확인
- [ ] v4 baseline 대비 retrieval quality 저하 없음 (5-sample spot check)
