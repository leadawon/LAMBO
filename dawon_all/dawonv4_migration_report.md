# dawonv4 migration report

이 보고서는 실제 `dawonv4` 코드 생성 전에 작성한 1단계 분석 초안이다. 기준은 현재 로컬 최신 baseline인 `script/anchor`이며, dawonv3는 독자 기능을 선별 이식할 참고 구현으로만 본다.

중요한 전제는 최신 baseline이 예전 baseline과 다르다는 점이다. 현재 `script/anchor/run_lambo_set1.py`는 `SearchR1 + RefineExtractor` 중심의 예전 루프가 아니라 `AnchorAgent -> SearchAgent/ExtractAgent -> RelationRefiner -> AnswerWriter -> llm_judge` 흐름으로 재설계되어 있다. 따라서 dawonv4는 dawonv3 파일을 통째로 덮어쓰지 않고, 최신 baseline의 agentic 구조를 source of truth로 유지한 뒤 dawonv3의 relation metadata, path resolver, 실행 wrapper만 최소 단위로 이식한다.

## A. baseline 그대로 재사용할 파일

- `script/anchor/evaluate_structured.py`: dawonv3와 기능 차이가 없고 structured eval의 공통 기준이므로 baseline 그대로 사용한다.
- `script/anchor/manifest.py`: 최신 baseline의 set1 manifest 기본 동작을 유지한다. 다만 selected indices 실행이 필요하면 `build_manifest_for_indices` helper만 추가 patch 후보로 둔다.
- `script/anchor/common.py`: 최신 baseline에 `instruction_from_record`가 추가되어 agentic prompt 흐름에서 쓰인다. dawonv3 버전으로 덮으면 이 구조가 깨지므로 baseline을 기본본으로 사용하고, 필요한 경우 `证券简称` title enrichment만 작은 patch로 검토한다.
- `script/anchor/backend.py`: 최신 baseline의 `meta_cognitive_rag.local_backend` 기본 경로를 유지한다. dawonv3 server backend는 실행 wrapper 지원이 필요할 때만 optional branch로 추가한다.
- `script/anchor/anchor_agent.py`: 최신 baseline은 task/type/level hints를 LLM에 노출하지 않는 새 anchor agent와 `doc_map` 구조를 갖는다. 이 파일은 baseline을 기본본으로 쓰고 relation enrichment 필드만 patch한다.
- `script/anchor/search_agent.py`: 최신 baseline의 핵심 search/extract 루프다. dawonv3의 `search_r1.py`로 대체하지 않고, relation summary/role/entity/target metadata를 doc_map과 trace에 싣는 최소 patch만 적용한다.
- `script/anchor/relation_refiner.py`: 최신 baseline의 cross-document relation fusion 단계다. dawonv3의 old refine loop로 되돌리지 않고, provenance/relation context를 fact formatter에 보존하는 정도만 검토한다.
- `script/anchor/answer_writer.py`: 최신 baseline은 `relations` object를 입력으로 final answer를 만든다. dawonv3의 doc_results 기반 answer_writer로 덮으면 새 구조가 깨지므로 baseline을 유지한다.
- `script/anchor/llm_judge.py`: 최신 baseline에 이미 local Qwen judge가 함수 형태로 통합되어 있다. dawonv3의 `run_loong_judge_local.py`를 중복 judge로 재도입하지 않는다.
- `script/anchor/prompts/answer_system.txt`, `answer_user.txt`, `extract_system.txt`, `extract_user.txt`, `search_system.txt`, `search_user.txt`, `refine_system.txt`, `refine_user.txt`, `anchor_system.txt`, `anchor_user.txt`: 최신 baseline의 agentic pipeline prompt다. dawonv3 prompt로 덮으면 팀원이 고친 새 설계가 사라진다.
- `script/anchor/export_anchor_pairs.py`: baseline-only 보조 export 도구다. dawonv3 독자 기능은 아니지만 baseline에 존재하는 파일이므로 dawonv4에 그대로 포함해도 안전하다.

## B. dawonv3에서만 존재하거나 dawonv3 쪽이 기능적으로 더 확장된 파일

- `dawonv3/anchor/anchor_relations.py`: baseline에 없다. anchor role/entity/target/value/provenance metadata와 relation graph, relation bias, provenance strength helper를 제공한다. dawonv4에서 relation-aware 기능을 복원하려면 가장 먼저 추가해야 한다.
- `dawonv3/anchor/paths.py`: baseline에 없다. Loong input/jsonl/src/model dir 및 Qwen model dir를 env override와 후보 경로로 찾는다. dawonv4를 독립 실행 가능하게 만들려면 필요하다. 단 env prefix는 `LAMBO_DAWONV4_*`를 우선하고 기존 `LAMBO_DAWON_*`는 호환 fallback으로 두는 전략이 안전하다.
- `dawonv3/prepare_qwen_workspace.py`: baseline에 없다. `/workspace/qwen/Qwen2.5-32B-Instruct` symlink 준비용 운영 유틸리티다. run wrapper가 server mode를 지원하려면 필요할 가능성이 높다.
- `dawonv3/run_set1_10.py`: baseline에는 root-level wrapper가 없다. dawonv4 package entrypoint를 제공하려면 `dawonv4.anchor.run_lambo_set1.main`으로 바꿔 새로 만든다.
- `dawonv3/run.sh`: baseline에는 dawon 전용 server wrapper가 없다. dawonv4에서는 최신 baseline CLI에 맞춰 `--skip_judge` 같은 옛 옵션을 제거하고, server env만 정리해 새로 작성한다.
- `dawonv3/README.md`: dawonv3 기준 문서라 그대로 복사하지 않는다. 다만 Qwen workspace, backend server, smoke command 설명을 dawonv4 기준으로 새 README에 반영한다.

## C. merge가 필요한 공통 파일

### `anchor_agent.py`

- baseline 유지할 부분: 최신 segmentation, `anchor_title`, `key_entities`, exact tiling repair, fallback tiling, `doc_map`, `query`/`instruction_from_record` 흐름.
- dawonv3에서 가져올 부분: `anchor_relations.enrich_anchor_graph` import, anchor 생성 후 relation enrichment 호출, doc-level `relation_summary`, anchor/doc_map에 role/entity/target/provenance/relation metadata 추가.
- 충돌 위험: 중간. dawonv3는 예전 schema의 `anchor_type`, `section_path`, `anchor_catalog`를 기대했지만 최신 baseline은 `anchor_title`, `key_entities`, `doc_map` 중심이다.
- 적용 전략: 최신 baseline 파일을 기본본으로 유지하고, `enrich_anchor_graph`가 없어도 기존 anchor fields를 최대한 활용하도록 추가 필드만 붙인다. dawonv3의 old anchor catalog 구조는 가져오지 않는다.

### `search_agent.py`

- baseline 유지할 부분: 최신 `SearchAgent`와 `ExtractAgent`의 `<think>/<search>/<info>/<extracted>` 루프, opened anchor fallback, prompt files, no task/type/level hints 원칙.
- dawonv3에서 가져올 부분: relation/role/entity/target metadata를 map formatting에 포함하는 것, opened anchor와 relation frontier를 고려한 candidate ordering, extracted item에 `source_relation_context`, `source_anchor_role`, `provenance_strength`, `disambiguation_needed`를 보존하는 것.
- 충돌 위험: 높음. dawonv3의 relation-aware ranking은 `SearchR1.choose_next_anchor`용이며 최신 baseline의 LLM-driven `SearchAgent`와 구조가 다르다.
- 적용 전략: dawonv3 `search_r1.py`를 통째로 이식하지 않는다. 대신 `anchor_relations.relation_context`와 `provenance_strength_from_anchor`를 `ExtractAgent` item 보강 및 `doc_map` 표시용으로만 연결한다.

### `relation_refiner.py`

- baseline 유지할 부분: 최신 baseline의 cross-document relation fusion과 `relations` output schema.
- dawonv3에서 가져올 부분: extracted item의 relation/provenance fields를 `_format_facts`에 함께 보여줘 refiner가 source strength와 relation context를 참고할 수 있게 하는 정도.
- 충돌 위험: 중간. dawonv3의 old `refine_extractor.py`는 per-document iterative extraction이고 최신 baseline의 `RelationRefiner`는 cross-document fusion이므로 직접 대응하지 않는다.
- 적용 전략: old refine loop를 되살리지 않는다. 최신 `RelationRefiner` input facts에 dawonv3 provenance metadata만 보존한다.

### `answer_writer.py`

- baseline 유지할 부분: `relations` object 기반 final answer writer와 prompt-driven output shape.
- dawonv3에서 가져올 부분: 직접 이식은 제한한다. dawonv3의 doc placeholder guard와 paper Reference/Citation 보정은 doc_results/item 기반이므로 최신 relations schema에 맞춘 재설계 없이는 위험하다.
- 충돌 위험: 높음. dawonv3 `answer_writer.py`를 덮으면 최신 baseline의 `RelationRefiner -> AnswerWriter` 계약이 깨진다.
- 적용 전략: 이번 1차 dawonv4에서는 baseline 유지. 필요하면 후속 TODO로 relations schema 기반 guard를 별도 설계한다.

### `backend.py`

- baseline 유지할 부분: 기본 `LocalTransformersBackend` 동작과 `QwenLocalClient.generate_text/generate_json` public surface.
- dawonv3에서 가져올 부분: optional OpenAI-compatible server backend, model path resolver, server env vars. 기본값은 baseline 동작을 유지한다.
- 충돌 위험: 중간. dawonv3 backend를 통째로 가져오면 baseline의 meta-cognitive backend 의존과 default model config가 바뀐다.
- 적용 전략: `LAMBO_DAWONV4_LLM_BACKEND=server`일 때만 server branch를 타게 하고, 기본은 baseline `LocalTransformersBackend`를 유지한다.

### `run_lambo_set1.py`

- baseline 유지할 부분: 최신 pipeline orchestration, `SearchAgent`, `ExtractAgent`, `RelationRefiner`, `AnswerWriter`, `llm_judge` 통합, `max_search_rounds` 옵션.
- dawonv3에서 가져올 부분: path resolver 기반 default input/output, selected indices 옵션, continue-on-error는 신중히 검토, dawonv4 import 경로.
- 충돌 위험: 높음. dawonv3 runner는 예전 `SearchR1/RefineExtractor` 루프와 official/local judge 구조이므로 통째 복사 금지.
- 적용 전략: 최신 baseline runner를 `dawonv4.anchor.*` import로 바꿔 기본본으로 삼고, `paths.py` 및 selected indices helper만 최소 patch한다. dawonv3 `--skip_judge`, official Loong judge wrapper는 재도입하지 않는다.

### `run_loong_judge_local.py`

- baseline 유지할 부분: baseline standalone local judge script.
- dawonv3에서 가져올 부분: 사실상 필요 없음. 최신 baseline은 `llm_judge.py`를 runner에서 이미 호출한다.
- 충돌 위험: 중간. 중복 judge가 생기면 보고서와 실행 산출물이 헷갈린다.
- 적용 전략: baseline script를 namespace만 `dawonv4`로 바꿔 보존하되, runner에는 연결하지 않는다.

### `manifest.py`

- baseline 유지할 부분: `SELECTED_SET1_INDICES`, `ManifestItem`, `build_set1_manifest` 기본 동작.
- dawonv3에서 가져올 부분: `build_manifest_for_indices(records, indices)`와 index range check.
- 충돌 위험: 낮음.
- 적용 전략: helper만 추가하고 `build_set1_manifest`는 이 helper를 호출하게 단순화한다.

### `common.py`

- baseline 유지할 부분: `instruction_from_record` 등 최신 agentic pipeline에서 쓰는 함수들.
- dawonv3에서 가져올 부분: `parse_docs_bundle`의 `证券简称` title enrichment 5줄.
- 충돌 위험: 낮음.
- 적용 전략: 최신 baseline 함수에 short-name 보강만 삽입한다.

## D. 절대 재도입하면 안 되는 부분

- dawonv3의 `run_lambo_set1.py` 전체: 최신 baseline의 agentic runner를 예전 `SearchR1/RefineExtractor` runner로 되돌린다.
- dawonv3의 `search_r1.py` 전체를 최신 `search_agent.py` 대신 쓰는 것: 최신 baseline의 LLM-driven map/open/extract 루프를 깨뜨린다.
- dawonv3의 `refine_extractor.py` 전체를 최신 `relation_refiner.py` 대신 쓰는 것: cross-document relation fusion 구조를 되돌린다.
- dawonv3의 `answer_writer.py` 전체: 최신 `relations` 기반 AnswerWriter 계약을 doc_results 기반 계약으로 되돌린다.
- dawonv3 prompt 전체: 최신 baseline의 `answer_*`, `extract_*`, `search_*`, 새 `refine_*`, 새 `anchor_*` prompt가 사라진다.
- dawonv3의 official Loong judge orchestration: 최신 baseline은 `llm_judge.py`를 이미 갖고 있으므로 judge를 중복해서 만들지 않는다.
- backend default를 dawonv3 방식으로 바꾸는 것: 최신 baseline의 기본 `LocalTransformersBackend` 동작을 source of truth로 유지한다. server backend는 opt-in이어야 한다.
- `--skip_judge` 같은 dawonv3 old runner 옵션을 그대로 wrapper에서 넘기는 것: 최신 baseline runner CLI에 없는 옵션이라 실행 실패를 만든다.
- hardcoded `dawonv3` 경로/문자열: dawonv4 생성 시 import, output dir, README, server log path는 모두 `dawonv4` 기준으로 정리한다.
