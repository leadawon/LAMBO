# dawonv3 baseline 차이 분석

## 1. 개요

이 문서는 리포지토리 루트의 `script/anchor` 구현을 baseline으로 보고, `dawonv3/anchor` 및 `dawonv3` 실행 파일이 baseline 위에 무엇을 추가했는지 분리한 기록이다.

이번 분석의 목적은 치팅 로직을 찾거나 고치는 것이 아니다. baseline에 치팅 로직이 있더라도 이 문서에서는 다루지 않고, dawonv3 고유 변경사항만 dawonv4 재적용 단위로 나눈다.

baseline은 루트 `README.md`에서 `script/anchor`를 핵심 구현으로 설명하며, 문서별 anchor 생성, summary 기반 anchor 선택, 선택 chunk 기반 refine, final answer writing, structured/judge 평가 흐름을 가진다. 코드 근거는 `README.md`의 "핵심 구현은 script/anchor" 설명과 `script/anchor/run_lambo_set1.py`의 end-to-end orchestration이다.

dawonv3는 같은 파이프라인 구조를 `dawonv3/anchor` 패키지로 분리한 뒤, anchor relation enrichment, relation-aware search, coverage-aware refine loop, relation/provenance 기반 answer post-processing, vLLM/OpenAI-compatible backend, 경로 자동 해결, balanced 99 subset 실행, local judge 통합을 추가한다.

단순 import namespace 변경도 존재한다. 예를 들어 `script.anchor.*`가 `dawonv3.anchor.*`로 바뀐 부분은 wrapper/config-path 성격으로 따로 분류했고, 실제 데이터 구조나 알고리즘 흐름을 바꾼 부분과 구분했다.

분석 범위는 사용자가 지정한 `script/anchor` 전체, 루트 `README.md`의 baseline 파이프라인 설명, `dawonv3/anchor` 전체, `dawonv3/README.md`, `dawonv3/run.sh`, `dawonv3/wait_run.sh`, `dawonv3/run_set1_10.py`, `dawonv3/prepare_qwen_workspace.py`, `dawonv3/prepare_exper99_subset.py`, `dawonv3/run_exper99.sh`이다. `__pycache__`, `logs`, 실행 산출물 JSON은 비교 근거에서 제외했다.

## 2. 한눈에 보는 변경 요약

새 파일 목록은 확실함이다.

- `dawonv3/anchor/anchor_relations.py`: baseline에 없는 relation/role/provenance enrichment 모듈이다.
- `dawonv3/anchor/paths.py`: baseline에 없는 경로 자동 해결 모듈이다.
- `dawonv3/README.md`: dawonv3 전용 실행 문서다.
- `dawonv3/run.sh`: vLLM/OpenAI-compatible backend 서버를 띄우고 set1 10개 실행을 감싸는 wrapper다.
- `dawonv3/wait_run.sh`: GPU idle 대기 후 backend 서버와 set1 10개 실행을 수행하는 wrapper다.
- `dawonv3/run_set1_10.py`: `dawonv3.anchor.run_lambo_set1.main`을 호출하는 얇은 wrapper다.
- `dawonv3/prepare_qwen_workspace.py`: Qwen 32B 모델 디렉토리를 `/workspace/qwen` 아래에 symlink로 준비하는 유틸리티다.
- `dawonv3/prepare_exper99_subset.py`: Loong SET1에서 domain별 균형 99개 subset과 인덱스 파일을 만드는 유틸리티다.
- `dawonv3/run_exper99.sh`: exper99 subset 생성, backend 서버 기동, 선택 인덱스 실행, 공식 judge 설정 전달을 묶은 wrapper다.
- `dawonv3/__init__.py`: 패키지 marker다. 사용자가 지정한 핵심 비교 범위의 중심은 아니지만, `dawonv3.*` import가 동작하려면 필요하다.

baseline에만 있는 파일은 확실함이다.

- `script/anchor/export_anchor_pairs.py`: `anchors.json`에서 anchor text-summary pair를 JSONL/Markdown으로 내보내는 보조 export 스크립트다. dawonv3에는 대응 파일이 없다. 다만 이 파일은 dawonv3 고유 기능이 아니므로 dawonv4 이식 대상이 아니다.

둘 다 있지만 내용이 바뀐 파일은 확실함이다.

- `anchor_agent.py`: relation enrichment 호출과 anchor schema 확장.
- `answer_writer.py`: relation/provenance 기반 item ranking, doc placeholder guard, paper Reference/Citation 보정.
- `backend.py`: meta-cognitive-RAG backend 의존에서 transformers 직접 로딩 또는 OpenAI-compatible server 호출 방식으로 교체.
- `common.py`: financial 문서 제목에 `证券简称`을 보강하는 parsing 변화.
- `manifest.py`: 고정 set1 manifest 생성을 `build_manifest_for_indices`로 일반화.
- `prompts/refine_user.txt`: 빈 evidence 처리와 paper Reference/Citation 방향 지시 추가.
- `refine_extractor.py`: coverage hints, relation frontier, expanded anchor window, provenance fields, sufficiency guard 추가.
- `run_lambo_set1.py`: dawonv3 package import, 경로 resolver, 선택 인덱스 실행, continue-on-error, local judge 통합, judge 경로/모델 옵션화.
- `run_loong_judge_local.py`: import namespace/default path 변경과 `run_local_judge` 함수화.
- `search_r1.py`: relation-aware, coverage-aware, utility-aware, redundancy-aware anchor ranking으로 확장.

사실상 wrapper/경로 변경만 있는 파일은 확실함이다.

- `dawonv3/run_set1_10.py`: `sys.path`에 project root를 넣고 `dawonv3.anchor.run_lambo_set1.main`만 호출한다.
- `dawonv3/README.md`: 문서 파일이다. 실행 방법과 환경변수를 설명하지만 알고리즘 자체는 없다.
- `dawonv3/run.sh`, `dawonv3/wait_run.sh`, `dawonv3/run_exper99.sh`: 실행 환경을 구성하는 shell wrapper다. 다만 backend 서버, subset, judge 옵션을 실제로 연결하므로 재현성에는 중요하다.
- `dawonv3/prepare_qwen_workspace.py`: 모델 위치 준비 유틸리티다.

config-path 성격이 강하지만 데이터 흐름도 바꾸는 파일은 확실함이다.

- `dawonv3/anchor/backend.py`: 단순 경로 변경이 아니라 LLM 호출 backend를 교체한다.
- `dawonv3/anchor/paths.py`: env override 및 후보 경로 탐색을 제공한다.
- `dawonv3/anchor/run_lambo_set1.py`: fixed path 대신 resolver와 CLI 인자로 Loong src/jsonl/model dir를 주입한다.
- `dawonv3/anchor/run_loong_judge_local.py`: 독립 CLI에서 callable function으로 바뀌어 `run_lambo_set1.py`가 직접 호출할 수 있다.

핵심 알고리즘 변경 파일은 확실함이다.

- `dawonv3/anchor/anchor_relations.py`
- `dawonv3/anchor/anchor_agent.py`
- `dawonv3/anchor/search_r1.py`
- `dawonv3/anchor/refine_extractor.py`
- `dawonv3/anchor/answer_writer.py`
- `dawonv3/anchor/common.py`의 `证券简称` title enrichment는 작은 변경이지만 search/entity matching에 영향을 줄 수 있다.
- `dawonv3/anchor/prompts/refine_user.txt`는 refine 모델 출력 정책을 바꾸므로 prompt-level 알고리즘 변경으로 취급한다.

likely-unchanged 파일은 확실함이다.

- `dawonv3/anchor/__init__.py`: baseline `script/anchor/__init__.py`와 동일하다.
- `dawonv3/anchor/evaluate_structured.py`: baseline `script/anchor/evaluate_structured.py`와 동일하다.
- `dawonv3/anchor/prompts/anchor_system.txt`, `anchor_user.txt`, `refine_system.txt`, `search_r1_system.txt`, `search_r1_user.txt`: baseline과 동일하다.

## 3. 파일별 상세 분석

### `dawonv3/anchor/anchor_relations.py`

분류: new, core-algorithm.

baseline 대비 차이 한 줄 요약: baseline에는 없는 anchor role/entity/target/provenance annotation과 anchor 간 relation graph 생성 모듈이다.

상세 변경 사항은 확실함이다. 파일 상단에서 `ANSWER_RELATION_TYPES`, `CONFLICT_RELATION_TYPES`, `SUPPORTIVE_RELATION_TYPES`를 정의한다. `annotate_anchor`는 `anchor_role_candidates`, `anchor_entities`, `anchor_targets`, `anchor_value_hints`, `provenance_hints`를 만든다. `build_anchor_relations`는 `prev_next`, `same_section`, `nearby_window`, `likely_same_entity`, `same_company_metric`, `same_target_paper`, `same_case`, `supports`, `disambiguates`, `prerequisite_of`, `conflicts_with` 관계를 anchor에 붙인다. `relation_bias_for_anchor`는 coverage hints와 선택된 relation frontier를 기반으로 search 점수 보정을 계산한다. `provenance_strength_from_anchor`는 refine/answer 단계에서 쓰는 strength와 disambiguation flag를 계산한다. 코드 근거는 `dawonv3/anchor/anchor_relations.py:10`, `dawonv3/anchor/anchor_relations.py:109`, `dawonv3/anchor/anchor_relations.py:223`, `dawonv3/anchor/anchor_relations.py:315`, `dawonv3/anchor/anchor_relations.py:340`, `dawonv3/anchor/anchor_relations.py:403`이다.

입력/출력 구조 변화는 확실함이다. 입력은 `record`, `doc_title`, anchor list이고, 출력은 doc-level `relation_summary`다. 더 중요하게는 anchor dict 자체가 mutate되어 `anchor_role_candidates`, `anchor_entities`, `anchor_targets`, `anchor_value_hints`, `provenance_hints`, `anchor_relations`, per-anchor `relation_summary`를 갖게 된다.

이 파일이 의존하는 dawonv3 전용 요소는 `common.compact_text`, `normalize_ws`, `quoted_terms`, `tokenize_query`이다. downstream 의존성은 더 강하다. `anchor_agent.py`가 `enrich_anchor_graph`를 호출하고, `search_r1.py`가 `relation_bias_for_anchor`를 호출하며, `refine_extractor.py`가 `relation_context`와 `provenance_strength_from_anchor`를 호출한다.

dawonv4로 옮길 때 주의점은 확실함이다. 이 파일만 복사하면 기능이 살아나지 않는다. `anchor_agent.py`, `search_r1.py`, `refine_extractor.py`, `answer_writer.py`의 relation/provenance 소비 부분과 함께 옮겨야 한다. relation type 이름을 바꾸면 search score와 refine frontier 조건이 동시에 깨질 수 있다.

### `dawonv3/anchor/paths.py`

분류: new, config-path.

baseline 대비 차이 한 줄 요약: baseline의 고정 `PROJECT_ROOT / Loong / ...` 경로를 env override와 후보 경로 탐색 방식으로 일반화한다.

상세 변경 사항은 확실함이다. `_resolve_path`는 환경변수 우선, 후보 경로 탐색, `strict=False`일 때 첫 후보 반환, `strict=True`일 때 FileNotFoundError를 구현한다. resolver는 `resolve_loong_process_path`, `resolve_loong_jsonl_path`, `resolve_loong_src`, `resolve_loong_model_dir`, `resolve_local_model_dir`이다. 코드 근거는 `dawonv3/anchor/paths.py:19`, `dawonv3/anchor/paths.py:48`, `dawonv3/anchor/paths.py:61`, `dawonv3/anchor/paths.py:74`, `dawonv3/anchor/paths.py:87`, `dawonv3/anchor/paths.py:100`이다.

입력/출력 구조 변화는 없다. 실행 설정 해석 방식만 바뀐다.

이 파일이 의존하는 dawonv3 전용 요소는 없다. 반대로 `backend.py`와 `run_lambo_set1.py`가 이 파일에 의존한다.

dawonv4로 옮길 때 주의점은 fixed baseline의 실제 배치 경로를 반영하는 것이다. 설계인 env override + 후보 탐색은 유용하지만, `/workspace/StructRAG`, `/workspace/Plan_Search_RAG`, `/workspace/qwen` 같은 후보는 dawonv4 환경에서 검토해야 한다.

### `dawonv3/anchor/anchor_agent.py`

분류: modified, core-algorithm.

baseline 대비 차이 한 줄 요약: anchor 생성 후 relation enrichment를 호출하고, anchor catalog/search view schema에 relation metadata를 추가한다.

상세 변경 사항은 확실함이다. baseline에는 없는 `from .anchor_relations import enrich_anchor_graph`가 추가되었다. anchor 생성 후 `anchors.sort(...)`를 수행하고 `prev_anchor_id`, `next_anchor_id`를 재설정한다. 그 뒤 `relation_summary = enrich_anchor_graph(...)`를 호출한다. doc payload에 `relation_summary`가 추가되고, `anchor_catalog` 항목에 `anchor_role_candidates`, `anchor_entities`, `anchor_targets`, `relation_summary`가 추가된다. `search_views.anchor_catalog_text`에는 `roles=...`가 추가되고, `search_views.relation_summary`도 추가된다. 코드 근거는 `dawonv3/anchor/anchor_agent.py:19`, `dawonv3/anchor/anchor_agent.py:294`, `dawonv3/anchor/anchor_agent.py:298`, `dawonv3/anchor/anchor_agent.py:305`, `dawonv3/anchor/anchor_agent.py:323`, `dawonv3/anchor/anchor_agent.py:335`이다. baseline contrast는 `script/anchor/anchor_agent.py` 같은 위치에서 `anchors`를 그대로 payload에 넣고 `anchor_catalog`에 id/type/section/summary만 넣는다.

입력/출력 구조 변화는 확실함이다. 입력은 동일하지만 `anchors.json` 안의 doc payload와 anchor dict가 확장된다. downstream search/refine는 새 필드가 없어도 일부 fallback은 가능하지만, dawonv3 의도인 relation-aware 흐름은 이 필드에 의존한다.

이 파일이 의존하는 dawonv3 전용 요소는 `anchor_relations.enrich_anchor_graph`이다.

dawonv4로 옮길 때 주의점은 anchor schema 확장과 downstream patch를 같은 단위로 옮기는 것이다. fixed baseline에서 anchor id/order 생성 규칙이 바뀌었다면 `prev_anchor_id`, `next_anchor_id`, relation graph 생성 순서가 충돌할 수 있다.

### `dawonv3/anchor/search_r1.py`

분류: modified, core-algorithm.

baseline 대비 차이 한 줄 요약: baseline의 lexical/prior 기반 anchor ranker를 relation-aware, coverage-aware, task-utility-aware, redundancy-aware ranker로 확장한다.

상세 변경 사항은 확실함이다. `relation_bias_for_anchor` import가 추가되었다. 새 helper로 `_contains_term`, `_title_terms`, `_important_terms`, `_query_entity_terms`, `_paper_direction`, `_text_similarity`, `_known_item_hints`, `_task_utility_score`, `_coverage_score`가 추가되었다. `_score_anchor`는 baseline처럼 `summary + section_path + doc_title` 중심으로만 점수를 내지 않고, anchor full text의 compact view까지 `combined`에 넣고 `local_score`, `utility_score`, `coverage_score`, `relation_score`, `prior_score`, `redundancy_penalty`를 합산한다. 코드 근거는 `dawonv3/anchor/search_r1.py:22`, `dawonv3/anchor/search_r1.py:121`, `dawonv3/anchor/search_r1.py:227`, `dawonv3/anchor/search_r1.py:252`, `dawonv3/anchor/search_r1.py:369`, `dawonv3/anchor/search_r1.py:435`, `dawonv3/anchor/search_r1.py:522`, `dawonv3/anchor/search_r1.py:534`, `dawonv3/anchor/search_r1.py:540`이다.

baseline contrast는 확실함이다. baseline `_score_anchor`는 query/instruction/quoted token overlap, section prior, anchor type prior, task prior, record type cue, position bonus를 하나의 score에 더하고, `coverage_hints`나 `avoid_anchor_ids`를 받지 않는다. 코드 근거는 `script/anchor/search_r1.py:120`, `script/anchor/search_r1.py:140`, `script/anchor/search_r1.py:194`, `script/anchor/search_r1.py:232`이다.

입력/출력 구조 변화는 확실함이다. `choose_next_anchor`에 `coverage_hints`, `avoid_anchor_ids` 인자가 추가되며 `known_items`에서 coverage hint를 합성한다. 반환 payload에는 `utility_hits`, `utility_score`, `relation_hits`, `relation_score`, `anchor_role_hits`, `coverage_hints`, `avoid_anchor_ids`, per-anchor `score_breakdown`이 추가된다. `run()`의 `tagged_trace` 안 `<search>` JSON에도 coverage/relation/role 정보가 들어간다. 코드 근거는 `dawonv3/anchor/search_r1.py:602`, `dawonv3/anchor/search_r1.py:624`, `dawonv3/anchor/search_r1.py:643`, `dawonv3/anchor/search_r1.py:651`, `dawonv3/anchor/search_r1.py:666`, `dawonv3/anchor/search_r1.py:675`, `dawonv3/anchor/search_r1.py:689`, `dawonv3/anchor/search_r1.py:828`이다.

이 파일이 의존하는 dawonv3 전용 요소는 `anchor_relations.relation_bias_for_anchor`, anchor에 추가된 `anchor_role_candidates`, `anchor_relations`, `relation_summary`, refine loop에서 넘기는 `coverage_hints`와 `avoid_anchor_ids`이다.

dawonv4로 옮길 때 주의점은 확실함이다. `search_r1.py`만 단독 이식하면 `coverage_hints`를 만들어주는 refine 쪽과 relation metadata를 만들어주는 anchor 쪽이 없어서 핵심 기능이 빠진다. fixed baseline의 search가 치팅 수정 대상이었을 가능성이 있으면 이 파일은 통째 복사가 아니라 line-level 수동 병합 대상이다.

### `dawonv3/anchor/refine_extractor.py`

분류: modified, core-algorithm.

baseline 대비 차이 한 줄 요약: 선택된 단일 anchor chunk만 읽는 baseline loop 위에 coverage tracking, relation frontier, expanded read window, provenance propagation, sufficiency guard를 추가한다.

상세 변경 사항은 확실함이다. `provenance_strength_from_anchor`, `relation_context` import가 추가되었다. `_normalize_items`는 `source_anchor_role`, `source_relation_context`, `provenance_strength`, `disambiguation_needed`를 item에 붙일 수 있게 확장되었다. `_must_find_gaps`, `_selected_relation_frontier`, `_relation_ordered_candidates`, `_coverage_hints`, `_is_sufficient`, `_expanded_anchor_window`가 새로 추가되었다. 코드 근거는 `dawonv3/anchor/refine_extractor.py:20`, `dawonv3/anchor/refine_extractor.py:71`, `dawonv3/anchor/refine_extractor.py:157`, `dawonv3/anchor/refine_extractor.py:198`, `dawonv3/anchor/refine_extractor.py:244`, `dawonv3/anchor/refine_extractor.py:253`, `dawonv3/anchor/refine_extractor.py:332`, `dawonv3/anchor/refine_extractor.py:381`이다.

baseline contrast는 확실함이다. baseline `_normalize_items`는 answer/value/evidence/source/confidence만 normalize하고, `_info_payload`는 id/type/section/summary/text만 반환한다. baseline run loop는 2라운드 이후 `search_agent.choose_next_anchor`에 `known_items`, `current_query_override`, `must_find_override`만 넘기고 coverage/relation frontier는 넘기지 않는다. 코드 근거는 `script/anchor/refine_extractor.py:68`, `script/anchor/refine_extractor.py:137`, `script/anchor/refine_extractor.py:251`, `script/anchor/refine_extractor.py:265`이다.

입력/출력 구조 변화는 확실함이다. `<info>`에 anchor role/entity/target/value/provenance/relation context가 들어갈 수 있다. `<retrieve>` payload에는 `coverage_hints`, `sufficiency_forced_false`, `sufficiency_reasons`가 들어갈 수 있다. 최종 refine JSON의 `items`에는 `source_anchor_role`, `source_relation_context`, `provenance_strength`, `disambiguation_needed`가 들어갈 수 있다. `selected_anchor_ids`는 동일하게 유지되지만 선택 전략은 relation/coverage 상태에 의해 달라진다. 코드 근거는 `dawonv3/anchor/refine_extractor.py:355`, `dawonv3/anchor/refine_extractor.py:507`, `dawonv3/anchor/refine_extractor.py:571`, `dawonv3/anchor/refine_extractor.py:593`, `dawonv3/anchor/refine_extractor.py:640`, `dawonv3/anchor/refine_extractor.py:674`, `dawonv3/anchor/refine_extractor.py:681`, `dawonv3/anchor/refine_extractor.py:688`이다.

이 파일이 의존하는 dawonv3 전용 요소는 `anchor_relations.py`가 anchor에 붙인 relation/provenance fields, 그리고 dawonv3 `SearchR1.choose_next_anchor`의 `coverage_hints`/`avoid_anchor_ids` 인자다.

dawonv4로 옮길 때 주의점은 확실함이다. fixed baseline에서 refine loop가 치팅 제거와 함께 바뀔 가능성이 높다. 따라서 dawonv3 파일을 덮어쓰지 말고, coverage hint 생성, relation frontier 선택, expanded window, provenance item field, sufficiency guard를 기능 단위로 병합해야 한다.

### `dawonv3/anchor/answer_writer.py`

분류: modified, core-algorithm.

baseline 대비 차이 한 줄 요약: final answer writing 후처리에 relation/provenance-aware item ranking, DOC placeholder 방지, paper Reference/Citation 방향 보정이 추가되었다.

상세 변경 사항은 확실함이다. `_flatten_items`가 item에 `doc_title`을 채워 넣고, `_item_rank_score`는 confidence뿐 아니라 `provenance_strength`, `disambiguation_needed`, `source_relation_context`의 relation type을 반영한다. `_is_doc_placeholder`, `_doc_title_answer`, `_ranked_values`는 `DOC\d+` 같은 placeholder 값을 title-derived answer로 바꾸거나 배제한다. `_canonical_relation_key`, `_target_paper_terms`, `_paper_relation_answer`는 paper task에서 Reference/Citation key를 canonicalize하고 target paper 기준으로 방향을 보정한다. `_guarded_answer`는 LLM final answer 또는 fallback answer를 최종적으로 한 번 더 정리한다. 코드 근거는 `dawonv3/anchor/answer_writer.py:23`, `dawonv3/anchor/answer_writer.py:33`, `dawonv3/anchor/answer_writer.py:47`, `dawonv3/anchor/answer_writer.py:92`, `dawonv3/anchor/answer_writer.py:112`, `dawonv3/anchor/answer_writer.py:151`, `dawonv3/anchor/answer_writer.py:305`이다.

baseline contrast는 확실함이다. baseline fallback ranking은 `confidence`만 사용하고, final answer가 empty일 때만 fallback을 쓰며 `_guarded_answer` 단계가 없다. 코드 근거는 `script/anchor/answer_writer.py:35`, `script/anchor/answer_writer.py:46`, `script/anchor/answer_writer.py:154`이다.

입력/출력 구조 변화는 중간 수준으로 확실함이다. public method signature는 같지만, doc_results item에 `provenance_strength`, `disambiguation_needed`, `source_relation_context`, `doc_title`이 있으면 final answer에 영향을 준다. output JSON schema는 `topology`, `final_answer`, `raw_text`, `evidence_item_count`로 baseline과 같지만 값 결정 로직은 바뀐다.

이 파일이 의존하는 dawonv3 전용 요소는 refine item에 추가되는 relation/provenance fields다. 이 필드가 없으면 helper가 대부분 fallback처럼 동작하지만 dawonv3 기능 효과가 약해진다.

dawonv4로 옮길 때 주의점은 확실함이다. 이 파일은 치팅 로직과 직접 관련이 있는지는 이 문서에서 판단하지 않는다. 다만 final answer를 바꾸는 알고리즘 변경이므로 fixed baseline의 answer_writer 수정과 충돌할 수 있다. relation/provenance item fields를 먼저 안정화한 뒤 보정 로직을 옮겨야 한다.

### `dawonv3/anchor/backend.py`

분류: modified, config-path/execution-backend.

baseline 대비 차이 한 줄 요약: baseline의 `meta_cognitive_rag.local_backend.LocalTransformersBackend` 의존을 제거하고, transformers 직접 로딩 또는 OpenAI-compatible server 호출 backend를 제공한다.

상세 변경 사항은 확실함이다. baseline은 `/workspace/meta-cognitive-RAG/src`를 `sys.path`에 넣고 `LocalTransformersBackend`, `LocalTransformersConfig`를 사용한다. dawonv3는 `requests`, `torch`, optional `transformers`를 import하고, `_OpenAICompatChatBackend`와 `_TransformersChatBackend`를 추가한다. `QwenLocalClient`는 `LAMBO_DAWON_LLM_BACKEND`가 `server/openai/openai_compat/vllm`이면 HTTP chat completions를 호출하고, 아니면 Hugging Face transformers로 직접 로딩한다. 모델 디렉토리는 `resolve_local_model_dir`로 찾는다. 코드 근거는 `script/anchor/backend.py:11`, `script/anchor/backend.py:15`, `dawonv3/anchor/backend.py:10`, `dawonv3/anchor/backend.py:14`, `dawonv3/anchor/backend.py:25`, `dawonv3/anchor/backend.py:134`, `dawonv3/anchor/backend.py:208`, `dawonv3/anchor/backend.py:222`이다.

입력/출력 구조 변화는 없다. `QwenLocalClient.generate_text`와 `generate_json`의 public surface는 유지된다. 다만 `metadata`는 dawonv3에서 `del metadata`로 버려지고, backend call path가 완전히 달라진다.

이 파일이 의존하는 dawonv3 전용 요소는 `paths.resolve_local_model_dir`와 `LAMBO_DAWON_*` 환경변수다.

dawonv4로 옮길 때 주의점은 확실함이다. 이것은 알고리즘 변경이 아니라 실행 backend 변경이지만, 추론 결과와 오류 양상에 영향을 줄 수 있다. fixed baseline이 사용하는 LLM backend가 이미 바뀌어 있다면 이 파일은 통째 복사보다 interface-compatible backend 선택지만 병합하는 편이 안전하다.

### `dawonv3/anchor/common.py`

분류: modified, small-logic.

baseline 대비 차이 한 줄 요약: document title parsing 시 본문에서 `证券简称`을 찾아 title에 붙인다.

상세 변경 사항은 확실함이다. `parse_docs_bundle`에서 `normalized_content`를 만든 뒤 `re.search(r"证券简称[:：]\s*([^\s|　]+)", normalized_content)`로 short name을 찾고, title에 없으면 `normalized_title = f"{normalized_title} {short_name}"`로 보강한다. 코드 근거는 `dawonv3/anchor/common.py:46`이고, baseline에는 같은 위치에 해당 로직이 없다.

입력/출력 구조 변화는 작지만 확실함이다. `record["docs"]`에서 추출되는 `doc_title` 값이 바뀔 수 있다. 이후 title term matching, relation entity extraction, answer placeholder conversion에 영향을 줄 수 있다.

이 파일이 의존하는 dawonv3 전용 요소는 없다.

dawonv4로 옮길 때 주의점은 fixed baseline이 title parsing을 수정했다면 충돌 가능성이 있다. 특히 financial task에서 company/entity matching을 강화하는 용도이므로 relation enrichment와 함께 검토해야 한다.

### `dawonv3/anchor/manifest.py`

분류: modified, config-path/execution-helper.

baseline 대비 차이 한 줄 요약: 고정 10개 set1 manifest 생성을 arbitrary selected indices manifest 생성으로 일반화한다.

상세 변경 사항은 확실함이다. `build_set1_manifest(records)`가 직접 `SELECTED_SET1_INDICES`를 loop하던 baseline과 달리 dawonv3에서는 `build_manifest_for_indices(records, SELECTED_SET1_INDICES)`를 호출한다. 새 `build_manifest_for_indices`는 index range check를 수행하고, 주어진 indices list로 manifest를 만든다. 코드 근거는 `dawonv3/anchor/manifest.py:32`, `dawonv3/anchor/manifest.py:36`, `dawonv3/anchor/manifest.py:42`이다. baseline contrast는 `script/anchor/manifest.py:32`이다.

입력/출력 구조 변화는 없다. ManifestItem schema는 동일하다. 선택 가능한 index source만 바뀐다.

이 파일이 의존하는 dawonv3 전용 요소는 `run_lambo_set1.py`의 `--selected_indices`, `--selected_indices_path` 옵션과 `run_exper99.sh`가 만든 indices JSON이다.

dawonv4로 옮길 때 주의점은 낮다. fixed baseline의 manifest 구조가 바뀌지 않았다면 helper 추가만 병합하면 된다.

### `dawonv3/anchor/evaluate_structured.py`

분류: likely-unchanged.

baseline 대비 차이 한 줄 요약: baseline과 동일하다.

상세 변경 사항은 없다. `diff` 기준으로 내용이 동일하다.

입력/출력 구조 변화는 없다.

이 파일이 의존하는 dawonv3 전용 요소는 없다.

dawonv4로 옮길 때 주의점은 없다. fixed baseline에서 evaluation을 고쳤다면 fixed baseline 버전을 유지하면 된다.

### `dawonv3/anchor/run_lambo_set1.py`

분류: modified, orchestration/config-path/judge-integration.

baseline 대비 차이 한 줄 요약: end-to-end 실행기는 dawonv3 namespace, path resolver, selected indices, continue-on-error, local judge, judge model/path 옵션을 통합한다.

상세 변경 사항은 확실함이다. import namespace가 `script.anchor.*`에서 `dawonv3.anchor.*`로 바뀌었다. `DAWON_ROOT`가 생기고 output default가 `dawonv3/logs/lambo_set1_10`으로 바뀌었다. `DEFAULT_INPUT_PATH`, `DEFAULT_LOONG_SRC`, `DEFAULT_LOONG_MODEL_DIR`, `DEFAULT_LOONG_JSONL`은 `paths.py` resolver로 만들어진다. CLI에는 `--selected_indices`, `--selected_indices_path`, `--continue_on_error`, `--skip_local_judge`, `--local_judge_max_output_tokens`, `--judge_eval_model`, `--judge_gen_model`, `--loong_src`, `--loong_model_dir`, `--loong_jsonl`이 추가되었다. 코드 근거는 `dawonv3/anchor/run_lambo_set1.py:12`, `dawonv3/anchor/run_lambo_set1.py:17`, `dawonv3/anchor/run_lambo_set1.py:29`, `dawonv3/anchor/run_lambo_set1.py:30`, `dawonv3/anchor/run_lambo_set1.py:41`, `dawonv3/anchor/run_lambo_set1.py:57`이다.

실행 흐름 변경도 확실함이다. `load_selected_indices`가 text/JSON/dict.indices에서 index list를 읽고, selected index가 있으면 `build_manifest_for_indices`를 호출한다. sample 처리 루프는 try/except로 감싸져 `sample_errors.jsonl`, sample별 `error.json`, `generate_response="meet error"` prediction row를 만들고, `--continue_on_error`가 없으면 다시 raise한다. 공식 Loong judge 호출은 `loong_src`, `loong_model_dir`, `loong_jsonl`, `loong_process_path`, judge model names를 인자로 받는다. local judge는 `run_local_judge`를 직접 호출해 `judge/loong_judge_eval_local_qwen32b.jsonl`과 `reports/loong_judge_local_qwen32b.json`을 만든다. 코드 근거는 `dawonv3/anchor/run_lambo_set1.py:74`, `dawonv3/anchor/run_lambo_set1.py:125`, `dawonv3/anchor/run_lambo_set1.py:266`, `dawonv3/anchor/run_lambo_set1.py:277`, `dawonv3/anchor/run_lambo_set1.py:303`, `dawonv3/anchor/run_lambo_set1.py:335`, `dawonv3/anchor/run_lambo_set1.py:377`, `dawonv3/anchor/run_lambo_set1.py:400`이다.

입력/출력 구조 변화는 확실함이다. 입력은 fixed set1뿐 아니라 selected indices로 바뀔 수 있다. output에는 `sample_errors.jsonl`, per-sample `error.json`, local judge output/summary가 추가될 수 있다. `summary_report.md`는 "Official Judge"와 "Local Judge" 섹션을 분리한다.

이 파일이 의존하는 dawonv3 전용 요소는 `paths.py`, `manifest.build_manifest_for_indices`, `run_loong_judge_local.run_local_judge`, dawonv3 package imports다.

dawonv4로 옮길 때 주의점은 확실함이다. 이 파일은 치팅 수정 가능성이 있는 pipeline orchestration과 직접 맞물린다. fixed baseline의 loop를 덮어쓰지 말고, path resolver, selected indices, error handling, local judge integration만 기능별 patch로 병합해야 한다.

### `dawonv3/anchor/run_loong_judge_local.py`

분류: modified, wrapper/judge-integration.

baseline 대비 차이 한 줄 요약: baseline에도 standalone local judge 스크립트가 있으나, dawonv3는 이를 `run_local_judge` 함수로 분리해 main runner에서 호출할 수 있게 했다.

상세 변경 사항은 확실함이다. import namespace가 `script.anchor.*`에서 `dawonv3.anchor.*`로 바뀌고 default paths가 `/workspace/LAMBO/dawonv3/logs/...`로 바뀌었다. baseline `main()` 안에 있던 실행 본문이 `run_local_judge(predictions, evaluate_output, summary_output, max_output_tokens, temperature)`로 이동했고 summary dict를 return한다. `main()`은 CLI args를 받아 이 함수를 호출하고 summary를 print한다. 코드 근거는 `dawonv3/anchor/run_loong_judge_local.py:10`, `dawonv3/anchor/run_loong_judge_local.py:45`, `dawonv3/anchor/run_loong_judge_local.py:93`, `dawonv3/anchor/run_loong_judge_local.py:168`이다.

입력/출력 구조 변화는 작지만 확실함이다. CLI 출력 파일 구조는 동일 계열이지만, 함수 호출자가 직접 path를 주입할 수 있다. return value가 생겨 `run_lambo_set1.py` report에 들어간다.

이 파일이 의존하는 dawonv3 전용 요소는 dawonv3 backend/common import와 `run_lambo_set1.py`의 local judge 호출부다.

dawonv4로 옮길 때 주의점은 baseline에도 local judge 자체는 존재한다는 점이다. "local judge 신규"라고 뭉뚱그리면 안 된다. 신규인 것은 callable 함수화와 end-to-end runner 통합이다.

### `dawonv3/anchor/prompts/refine_user.txt`

분류: modified, prompt-logic.

baseline 대비 차이 한 줄 요약: direct evidence가 없을 때 generic candidate를 내지 말라는 지시와 paper Reference/Citation 방향 지시가 추가되었다.

상세 변경 사항은 확실함이다. "If the selected chunk has no direct evidence..." 문장과 "For paper citation tasks..." 문장이 추가되었다. 코드 근거는 `dawonv3/anchor/prompts/refine_user.txt`의 selected chunk 지시 아래 추가 2줄이며, diff 기준 line count가 baseline 33줄에서 dawonv3 35줄로 늘었다.

입력/출력 구조 변화는 없다. 다만 LLM이 `<retrieve>` items를 생성하는 정책이 바뀌므로 refine output 내용에는 영향을 줄 수 있다.

이 파일이 의존하는 dawonv3 전용 요소는 없다. 그러나 answer_writer의 Reference/Citation 보정과 같은 방향의 task-specific 보강이다.

dawonv4로 옮길 때 주의점은 fixed baseline에서 prompt가 이미 치팅 제거 목적으로 수정되었을 수 있다는 점이다. 두 줄을 그대로 넣기보다 fixed prompt와 충돌 여부를 검토해야 한다.

### `dawonv3/anchor/__init__.py`

분류: likely-unchanged.

baseline 대비 차이 한 줄 요약: baseline과 동일한 package marker다.

상세 변경 사항, 입력/출력 구조 변화, dawonv3 전용 의존성은 없다.

dawonv4로 옮길 때 주의점은 없다.

### `dawonv3/anchor/prompts/anchor_system.txt`, `anchor_user.txt`, `refine_system.txt`, `search_r1_system.txt`, `search_r1_user.txt`

분류: likely-unchanged.

baseline 대비 차이 한 줄 요약: baseline과 동일하다.

상세 변경 사항은 없다. `diff` 기준으로 identical이다.

입력/출력 구조 변화는 없다.

이 파일들이 의존하는 dawonv3 전용 요소는 없다.

dawonv4로 옮길 때 주의점은 fixed baseline에서 prompt를 고쳤다면 fixed baseline 버전을 우선 유지해도 dawonv3 고유 기능 손실은 없다. 단, `refine_user.txt`만은 별도 항목처럼 변경이 있다.

### `dawonv3/README.md`

분류: new, wrapper/documentation.

baseline 대비 차이 한 줄 요약: baseline 루트 README의 알고리즘 설명 대신 dawonv3 실행 방법, Qwen symlink 준비, vLLM server backend, local judge, env override를 설명한다.

상세 변경 사항은 확실함이다. dawonv3 README는 `/workspace/qwen/Qwen2.5-32B-Instruct` 준비, `prepare_qwen_workspace.py`, `bash /workspace/LAMBO/dawonv3/run.sh`, `run_set1_10.py --skip_judge`, `--skip_judge`와 `--skip_local_judge`, `LAMBO_DAWON_*` 환경변수를 설명한다. 코드 근거는 `dawonv3/README.md:5`, `dawonv3/README.md:22`, `dawonv3/README.md:62`, `dawonv3/README.md:66`, `dawonv3/README.md:93`이다. baseline README는 `script/anchor`의 conceptual pipeline과 set1 10개 실험 결과까지 설명한다.

입력/출력 구조 변화는 없다. 문서이지만 dawonv3 실행 기본값을 이해하는 근거다.

이 파일이 의존하는 dawonv3 전용 요소는 `run.sh`, `run_set1_10.py`, `prepare_qwen_workspace.py`, `backend.py`, `run_lambo_set1.py`의 local judge 옵션이다.

dawonv4로 옮길 때 주의점은 README를 그대로 복사하지 않는 것이다. dawonv4의 경로와 fixed baseline 위 재적용 사실에 맞춰 새 README로 재작성해야 한다.

### `dawonv3/run.sh`

분류: new, wrapper/execution-backend.

baseline 대비 차이 한 줄 요약: Qwen workspace 준비, vLLM OpenAI-compatible backend 서버 기동, health check, set1 실행, cleanup을 한 번에 수행한다.

상세 변경 사항은 확실함이다. `PYTHON_BIN`, `OUTPUT_DIR`, `SERVER_*`, `LAMBO_DAWON_LLM_BACKEND` 기본값을 env로 받는다. `prepare_qwen_workspace.py`를 실행하고 `LAMBO_DAWON_MODEL_DIR`, `LAMBO_DAWON_BASE_URL` 등을 export한다. `vllm.entrypoints.openai.api_server`를 `setsid`로 띄우고 `/health`를 polling한다. `run_set1_10.py --output_dir ... --force --skip_judge "$@"`를 실행하며 cleanup에서 server process group을 종료한다. 코드 근거는 `dawonv3/run.sh:5`, `dawonv3/run.sh:32`, `dawonv3/run.sh:48`, `dawonv3/run.sh:100`, `dawonv3/run.sh:105`, `dawonv3/run.sh:117`, `dawonv3/run.sh:152`이다.

입력/출력 구조 변화는 없다. 실행 기본값으로 official judge는 skip되고 local judge는 `run_lambo_set1.py` 기본값에 따라 실행된다.

이 파일이 의존하는 dawonv3 전용 요소는 `prepare_qwen_workspace.py`, `run_set1_10.py`, backend의 server mode env vars다.

dawonv4로 옮길 때 주의점은 운영 wrapper로 분리해 적용하는 것이다. 알고리즘 patch와 동시에 섞어 병합하지 않는 편이 안전하다. fixed baseline에서 server port/model path가 다르면 env default를 수정해야 한다.

### `dawonv3/wait_run.sh`

분류: new, wrapper/execution-backend.

baseline 대비 차이 한 줄 요약: `run.sh`에 GPU idle 대기 로직을 더한 실행 wrapper다.

상세 변경 사항은 확실함이다. `nvidia-smi`로 GPU memory.used를 읽고, 지정 GPU 수가 `LAMBO_DAWON_GPU_IDLE_THRESHOLD_MB` 미만으로 `LAMBO_DAWON_GPU_IDLE_DURATION_SEC` 동안 유지될 때 실행을 시작한다. 이후 Qwen workspace 준비, vLLM server start, health check, `run_set1_10.py --force --skip_judge` 실행은 `run.sh`와 유사하다. 코드 근거는 `dawonv3/wait_run.sh:17`, `dawonv3/wait_run.sh:50`, `dawonv3/wait_run.sh:138`, `dawonv3/wait_run.sh:151`, `dawonv3/wait_run.sh:175`이다.

입력/출력 구조 변화는 없다.

이 파일이 의존하는 dawonv3 전용 요소는 `prepare_qwen_workspace.py`, `run_set1_10.py`, backend server env vars이며, 외부 의존성으로 `nvidia-smi`가 있다.

dawonv4로 옮길 때 주의점은 필수 알고리즘 기능이 아니라 운영 편의 기능이라는 점이다. GPU idle 대기 정책은 공유 서버 환경에 맞춰 조정해야 한다.

### `dawonv3/run_set1_10.py`

분류: new, wrapper.

baseline 대비 차이 한 줄 요약: dawonv3 package entrypoint를 간단히 호출한다.

상세 변경 사항은 확실함이다. `DAWON_ROOT.parent`를 `sys.path`에 추가하고 `from dawonv3.anchor.run_lambo_set1 import main` 후 `main()`을 호출한다. 코드 근거는 `dawonv3/run_set1_10.py:7`, `dawonv3/run_set1_10.py:12`, `dawonv3/run_set1_10.py:15`이다.

입력/출력 구조 변화는 없다. 모든 CLI와 출력은 `dawonv3/anchor/run_lambo_set1.py`가 결정한다.

이 파일이 의존하는 dawonv3 전용 요소는 dawonv3 package import다.

dawonv4로 옮길 때 주의점은 낮다. dawonv4 package name에 맞게 import만 바꾸면 된다.

### `dawonv3/prepare_qwen_workspace.py`

분류: new, wrapper/config-path.

baseline 대비 차이 한 줄 요약: 기존 Qwen model directory를 `/workspace/qwen/Qwen2.5-32B-Instruct`로 symlink 연결한다.

상세 변경 사항은 확실함이다. default target root는 `/workspace/qwen`, model name은 `Qwen2.5-32B-Instruct`다. source는 CLI `--source`, env `LAMBO_DAWON_EXISTING_MODEL_SOURCE`, `/workspace/StructRAG/model/<model>`, `/workspace/meta-cognitive-RAG/models/<model>` 순으로 찾는다. target이 없으면 `target_dir.symlink_to(source_dir, target_is_directory=True)`를 수행한다. 코드 근거는 `dawonv3/prepare_qwen_workspace.py:9`, `dawonv3/prepare_qwen_workspace.py:38`, `dawonv3/prepare_qwen_workspace.py:56`, `dawonv3/prepare_qwen_workspace.py:73`이다.

입력/출력 구조 변화는 없다. stdout에 status JSON을 출력한다.

이 파일이 의존하는 dawonv3 전용 요소는 없다. `run.sh`, `wait_run.sh`, `run_exper99.sh`가 이 유틸리티를 호출한다.

dawonv4로 옮길 때 주의점은 모델 복사를 피하는 운영 기능으로 유지하되, 실제 모델 위치 후보는 환경에 맞게 재검토해야 한다.

### `dawonv3/prepare_exper99_subset.py`

분류: new, wrapper/data-selection.

baseline 대비 차이 한 줄 요약: Loong SET1에서 paper/legal/financial을 domain별 33개씩 뽑아 99개 subset과 selected index manifest를 만든다.

상세 변경 사항은 확실함이다. default input은 `/workspace/StructRAG/loong/Loong/data/loong_process.jsonl`, default output은 `dawonv3/data/loong_set1_balanced99.*`이다. `--set-id`, `--per-domain`, `--domains`를 받고, `row["set"] == set_id` 및 type domain filter를 적용해 domain별 bucket을 채운다. output은 subset JSONL, `{"indices": [...]}` JSON, manifest JSON이다. 코드 근거는 `dawonv3/prepare_exper99_subset.py:11`, `dawonv3/prepare_exper99_subset.py:25`, `dawonv3/prepare_exper99_subset.py:61`, `dawonv3/prepare_exper99_subset.py:74`, `dawonv3/prepare_exper99_subset.py:100`이다.

입력/출력 구조 변화는 실행 단위에 있다. pipeline record schema는 바꾸지 않지만, `run_lambo_set1.py --selected_indices_path`가 읽을 index file을 생성한다.

이 파일이 의존하는 dawonv3 전용 요소는 `run_exper99.sh`의 selected indices 실행 흐름이다.

dawonv4로 옮길 때 주의점은 exper99 실험 재현이 목표일 때만 필요하다는 점이다. fixed baseline 검증의 첫 단계에는 set1 10개 smoke run이 더 작고 안전하다.

### `dawonv3/run_exper99.sh`

분류: new, wrapper/data-selection/execution-backend.

baseline 대비 차이 한 줄 요약: exper99 subset 생성부터 vLLM backend server 실행, selected indices pipeline 실행, judge path/model 옵션 전달까지 묶은 shell wrapper다.

상세 변경 사항은 확실함이다. `INPUT_PATH`, `DATA_DIR`, `SUBSET_PATH`, `INDICES_PATH`, `MANIFEST_PATH`, `JUDGE_EVAL_MODEL`, `JUDGE_GEN_MODEL`, `PROCESS_NUM_EVAL`, `LOONG_SRC`, `LOONG_MODEL_DIR`, `LOONG_JSONL`을 env override로 받는다. `prepare_exper99_subset.py`와 `prepare_qwen_workspace.py`를 실행하고, backend server를 띄운 뒤 `run_set1_10.py --selected_indices_path "${INDICES_PATH}" --continue_on_error --skip_local_judge --process_num_eval ... --judge_eval_model ... --judge_gen_model ...`를 호출한다. 코드 근거는 `dawonv3/run_exper99.sh:5`, `dawonv3/run_exper99.sh:24`, `dawonv3/run_exper99.sh:116`, `dawonv3/run_exper99.sh:122`, `dawonv3/run_exper99.sh:138`, `dawonv3/run_exper99.sh:173`이다.

입력/출력 구조 변화는 실행 규모와 평가 산출물에 있다. 기본 출력은 `dawonv3/logs/lambo_set1_exper99`이며, selected index file로 99개를 실행한다. local judge는 기본 호출에서 `--skip_local_judge`로 꺼져 있다.

이 파일이 의존하는 dawonv3 전용 요소는 `prepare_exper99_subset.py`, `prepare_qwen_workspace.py`, `run_set1_10.py`, `run_lambo_set1.py`의 selected indices/judge path 옵션, backend server mode다.

dawonv4로 옮길 때 주의점은 exper99는 큰 실행 wrapper라서 알고리즘 이식 완료 후 마지막에 붙이는 것이 좋다. fixed baseline의 judge config가 바뀌면 `JUDGE_EVAL_MODEL` default와 Loong path가 충돌할 수 있다.

### `script/anchor/export_anchor_pairs.py`

분류: baseline-only.

baseline 대비 차이 한 줄 요약: baseline 쪽에만 있는 anchor export helper이며 dawonv3 고유 기능이 아니다.

상세 변경 사항은 확실함이다. `anchors.json` 또는 sample/run directory를 받아 `anchor_pairs.jsonl`, `anchor_pairs.md`를 생성한다. 코드 근거는 `script/anchor/export_anchor_pairs.py:9`, `script/anchor/export_anchor_pairs.py:30`, `script/anchor/export_anchor_pairs.py:56`, `script/anchor/export_anchor_pairs.py:117`이다.

입력/출력 구조 변화는 dawonv3에는 없다.

이 파일이 의존하는 dawonv3 전용 요소는 없다.

dawonv4로 옮길 때 주의점은 이 파일을 dawonv3 기능으로 착각해 fixed baseline에 별도로 이식하지 않는 것이다. fixed baseline에 이미 있다면 그대로 두면 된다.

## 4. 로직 단위 diff

anchor 생성 단계의 dawonv3 고유 변경은 확실함이다. baseline은 anchor list와 summary catalog를 만든다. dawonv3는 anchor를 order 기준 정렬하고 prev/next id를 재설정한 뒤 `enrich_anchor_graph`로 role/entity/target/provenance/relation을 부여한다. 이후 `anchors.json`의 doc-level `relation_summary`, anchor-level relation fields, catalog roles가 downstream으로 전달된다. 관련 파일은 `anchor_relations.py`, `anchor_agent.py`, `common.py`이다.

search/planning 단계의 dawonv3 고유 변경은 확실함이다. baseline `SearchR1`는 current query와 anchor summary/section/type prior 중심으로 rank한다. dawonv3는 anchor text compact view, record type별 utility scoring, coverage hints, selected anchor avoidance, known value redundancy penalty, relation frontier bias, role hits를 점수에 넣는다. 관련 파일은 `search_r1.py`, `anchor_relations.py`, `refine_extractor.py`이다.

refine/evidence extraction 단계의 dawonv3 고유 변경은 확실함이다. baseline은 선택 anchor 하나를 `<info>`로 넣고 LLM이 `<retrieve>`를 반환하면 충분성에 따라 다음 anchor를 고른다. dawonv3는 선택 anchor 주변 same-section/relation neighbor를 포함하는 expanded window를 만들 수 있고, coverage hints로 missing slot/low confidence/doc placeholder/visited texts/relation frontier를 추적한다. 추출 item에는 provenance/relation context가 붙고, LLM이 enough=true를 내도 coverage가 부족하면 `sufficiency_forced_false`로 계속 탐색한다.

answer writing 단계의 dawonv3 고유 변경은 확실함이다. baseline은 LLM final answer가 비었을 때 fallback을 쓰고 confidence 중심으로 item을 정렬한다. dawonv3는 evidence item의 provenance/relation/disambiguation metadata로 fallback ranking을 조정하고, financial string answer의 `DOC1` placeholder를 방지하며, paper dict answer에서 Reference/Citation 방향을 보정한다.

evaluation/judge 단계의 dawonv3 고유 변경은 확실함이다. `evaluate_structured.py`는 baseline과 동일하다. official Loong judge 호출 자체는 baseline에도 있지만 dawonv3는 Loong src/jsonl/model dir와 judge model yaml을 CLI로 받는다. local Qwen judge 자체도 baseline에 standalone script로 존재하지만, dawonv3는 `run_local_judge` 함수로 만들고 `run_lambo_set1.py`에서 자동 호출 및 report 통합을 수행한다.

실행 환경 및 모델 경로 해결 단계의 dawonv3 고유 변경은 확실함이다. baseline backend는 meta-cognitive-RAG의 local backend에 직접 의존한다. dawonv3 backend는 transformers 직접 로딩과 OpenAI-compatible/vLLM server 호출을 선택한다. `paths.py`, `prepare_qwen_workspace.py`, `run.sh`, `wait_run.sh`, `run_exper99.sh`가 이 흐름을 지원한다.

## 5. 기능 단위 재적용 계획

반드시 함께 옮겨야 하는 변경 묶음은 다음과 같다.

1. anchor relation enrichment 묶음. `anchor_relations.py`, `anchor_agent.py`의 `enrich_anchor_graph` 호출과 anchor schema 확장, `common.py`의 financial short name title 보강을 함께 검토한다. search/refine가 role/entity/relation/provenance fields를 소비하기 때문이다.

2. relation-aware search 묶음. `search_r1.py`의 `coverage_hints`, `avoid_anchor_ids`, `relation_bias_for_anchor`, utility/coverage/redundancy scoring을 `refine_extractor.py`의 coverage loop와 함께 옮긴다. search만 옮기면 caller signature와 coverage data source가 끊긴다.

3. iterative refine coverage loop 묶음. `refine_extractor.py`의 coverage hints, relation frontier, expanded anchor window, provenance item fields, sufficiency guard를 한 묶음으로 옮긴다. 이 묶음은 anchor relation enrichment와 search 묶음에 의존한다.

4. relation/provenance-aware answer writing 묶음. `answer_writer.py`의 `_item_rank_score`, doc placeholder guard, paper Reference/Citation 보정은 refine item field가 있어야 효과가 난다. refine 묶음 이후에 옮기는 것이 안전하다.

5. backend/path resolver 묶음. `paths.py`, `backend.py`, `prepare_qwen_workspace.py`, `run.sh`의 `LAMBO_DAWON_*` env 흐름은 함께 검토한다. 알고리즘은 아니지만 실행 가능성에 영향을 준다.

6. judge/orchestration 묶음. `run_lambo_set1.py`, `run_loong_judge_local.py`, `manifest.py`, `run_set1_10.py`, `run_exper99.sh`, `prepare_exper99_subset.py`는 selected index 실행, continue-on-error, local judge, official judge path 옵션을 함께 다룬다.

독립적으로 옮길 수 있는 변경 묶음은 다음과 같다.

- `prepare_qwen_workspace.py`는 symlink 준비 도구라 독립성이 높다.
- `paths.py`는 backend/runner가 사용하도록 import만 붙이면 독립적으로 검토 가능하다.
- `manifest.py`의 `build_manifest_for_indices`는 run_lambo의 selected indices 옵션과 함께 작지만 독립적인 helper다.
- `run_set1_10.py`는 package name만 맞추면 재생성 가능한 wrapper다.
- `wait_run.sh`는 GPU idle 대기 운영 wrapper라 알고리즘과 독립적이다.
- `prepare_exper99_subset.py`와 `run_exper99.sh`는 exper99 재현이 필요할 때 마지막에 붙일 수 있다.

baseline fixed 버전과 충돌 가능성이 높은 부분은 다음과 같다.

- `refine_extractor.py`: 치팅 수정이 refine loop, enough 판단, selected chunk 사용 범위에 들어갔다면 dawonv3 coverage/expanded window와 직접 충돌할 수 있다.
- `search_r1.py`: fixed baseline에서 anchor 선택 로직을 고쳤다면 dawonv3 scoring을 통째로 덮는 것은 위험하다.
- `answer_writer.py`: final answer post-processing은 평가 결과를 직접 바꾸므로 fixed baseline의 bugfix와 충돌할 수 있다.
- `run_lambo_set1.py`: runner가 공식 judge, prediction row, error handling을 바꾸면 dawonv3 orchestration과 충돌한다.
- `backend.py`: fixed baseline이 이미 backend를 바꿨다면 dawonv3 server mode를 별도 option으로 합쳐야 한다.

충돌 가능성은 낮지만 놓치기 쉬운 부분은 다음과 같다.

- `common.py`의 `证券简称` title enrichment는 5줄짜리 작은 변경이지만 financial entity matching에 영향을 준다.
- `prompts/refine_user.txt`의 2줄 추가는 코드가 아니라서 diff 병합 시 누락되기 쉽다.
- `anchor_agent.py`의 `search_views.anchor_catalog_text`에 `roles=...`가 들어가는 점은 search prompt/debug view에 영향을 줄 수 있다.
- `run_loong_judge_local.py`는 local judge 자체가 신규가 아니라 callable화와 path 변경이 신규라는 점을 놓치기 쉽다.
- `manifest.py`의 selected index range check는 exper99 실행 안정성에 필요하다.
- `run_exper99.sh`는 기본으로 `--skip_local_judge`를 전달한다. set1 wrapper의 local judge 기본 동작과 다르다.

## 6. dawonv4 제작 체크리스트

1. fixed baseline 브랜치에서 시작한다. `script/anchor` 또는 dawonv4 target package가 fixed baseline의 치팅 수정 상태를 담고 있는지 확인한다.

2. baseline 전체를 dawonv3로 덮어쓰지 않는다. 먼저 fixed baseline의 `anchor_agent.py`, `search_r1.py`, `refine_extractor.py`, `answer_writer.py`, `run_lambo_set1.py`, `backend.py`를 기준 파일로 둔다.

3. 신규 독립 파일을 추가한다. 최소 후보는 `anchor_relations.py`와 `paths.py`다. 운영까지 포함하면 `prepare_qwen_workspace.py`, `run_set1_10.py`, `run.sh`도 추가한다. exper99가 목표일 때만 `prepare_exper99_subset.py`, `run_exper99.sh`, `wait_run.sh`를 추가한다.

4. `common.py`에 `证券简称` title enrichment를 patch할지 검토한다. fixed baseline의 `parse_docs_bundle`이 이미 바뀌었으면 중복 보강하지 않는다.

5. `anchor_agent.py`에 `enrich_anchor_graph` 호출, anchor sort/prev-next 재설정, doc-level/per-anchor relation fields, catalog roles, `search_views.relation_summary`를 patch한다.

6. anchor schema를 작은 샘플 1개로 확인한다. `anchors.json`에 `anchor_role_candidates`, `anchor_entities`, `anchor_targets`, `anchor_value_hints`, `provenance_hints`, `anchor_relations`, `relation_summary`가 생기는지 본다.

7. `search_r1.py`에 relation-aware/coverage-aware 변경을 수동 병합한다. `choose_next_anchor` signature가 `coverage_hints`와 `avoid_anchor_ids`를 받는지, 반환 payload에 relation/coverage/score breakdown이 들어가는지 확인한다.

8. `refine_extractor.py`에 coverage loop를 수동 병합한다. `_coverage_hints`, `_selected_relation_frontier`, `_expanded_anchor_window`, provenance item field, sufficiency guard가 search signature와 맞는지 확인한다.

9. `prompts/refine_user.txt`에 evidence 없음 처리와 paper Reference/Citation 방향 지시를 fixed prompt와 충돌 없이 반영한다.

10. `answer_writer.py`에 relation/provenance item ranking과 `_guarded_answer`를 병합한다. fixed baseline의 answer post-processing bugfix가 있으면 우선순위를 정해 수동 병합한다.

11. `backend.py`와 `paths.py`를 연결한다. `LAMBO_DAWON_MODEL_DIR`, `LAMBO_DAWON_LLM_BACKEND`, `LAMBO_DAWON_BASE_URL`이 dawonv4 naming과 맞는지 결정한다.

12. `manifest.py`에 `build_manifest_for_indices`를 추가하고, `run_lambo_set1.py`에 `--selected_indices`, `--selected_indices_path`를 patch한다.

13. `run_loong_judge_local.py`의 `run_local_judge` 함수화를 반영하고, `run_lambo_set1.py`에 `--skip_local_judge`, `--local_judge_max_output_tokens`, local judge output/report 생성을 patch한다.

14. `run_lambo_set1.py`에 official judge path/model 옵션화를 patch한다. `--loong_src`, `--loong_model_dir`, `--loong_jsonl`, `--judge_eval_model`, `--judge_gen_model`이 fixed baseline의 평가 흐름과 충돌하지 않는지 확인한다.

15. `run_lambo_set1.py`에 `--continue_on_error`와 `sample_errors.jsonl` 처리를 patch할지 결정한다. fixed baseline 검증 초기에는 에러를 숨기지 않도록 기본값은 continue off로 유지한다.

16. 첫 실행 검증은 `--max_items 1 --skip_judge --skip_local_judge --force` 같은 작은 범위로 한다. 확인 항목은 anchor schema, search output의 `coverage_hints`/`relation_hits`/`score_breakdown`, refine output의 provenance/relation fields, answer writer output이다.

17. 그 다음 set1 10개를 실행한다. official judge는 필요 시 skip하고 structured eval과 local judge를 분리해 본다.

18. exper99 재현이 필요할 때 마지막으로 `prepare_exper99_subset.py`와 `run_exper99.sh`를 붙이고, selected indices path 기반 실행이 되는지 확인한다.

19. dawonv4 README를 새로 쓴다. dawonv3 README를 그대로 복사하지 말고 fixed baseline 위에 어떤 묶음을 재적용했는지 명시한다.

## 7. 절대 하지 말아야 할 것

- baseline의 치팅 로직을 dawonv3 쪽으로 역이식하지 않는다.
- fixed baseline 전체를 dawonv3 디렉토리로 통째로 덮어쓰지 않는다.
- dawonv3 전체를 fixed baseline 위에 그대로 복사하지 않는다.
- import path만 `dawonv3`에서 `dawonv4`로 바꾸고 실제 데이터 흐름 차이를 놓치지 않는다.
- `anchor_relations.py` 없이 `search_r1.py`만 옮기지 않는다.
- `search_r1.py`만 옮기고 `refine_extractor.py`의 `coverage_hints`/`avoid_anchor_ids` caller 변경을 누락하지 않는다.
- relation metadata를 만드는 anchor 단계 없이 answer_writer의 provenance-aware ranking만 옮기지 않는다.
- `refine_user.txt`의 prompt 변경을 fixed baseline prompt와 비교하지 않고 덮어쓰지 않는다.
- 실행 wrapper, backend, judge 변경과 핵심 알고리즘 변경을 한 commit/patch 단위로 무차별 병합하지 않는다.
- `run_lambo_set1.py`의 `continue_on_error`를 켠 상태로 초기 검증을 해서 치명적 오류를 정상 prediction처럼 지나치지 않는다.
- `run_exper99.sh`를 dawonv4 첫 검증으로 쓰지 않는다. 먼저 작은 샘플과 set1 10개로 schema와 loop를 확인한다.

## 8. 확실성 구분

확실함:

- `anchor_relations.py`와 `paths.py`는 baseline에 없는 신규 파일이다.
- `evaluate_structured.py`, `__init__.py`, anchor/search/refine system/user prompt 중 `refine_user.txt`를 제외한 prompt들은 baseline과 동일하다.
- `anchor_agent.py`는 `enrich_anchor_graph` 호출과 relation schema를 추가한다.
- `search_r1.py`는 `coverage_hints`, `avoid_anchor_ids`, relation bias, utility/coverage/redundancy score breakdown을 추가한다.
- `refine_extractor.py`는 coverage hints, relation frontier, expanded anchor window, provenance item fields, sufficiency guard를 추가한다.
- `answer_writer.py`는 relation/provenance-aware ranking, doc placeholder guard, paper Reference/Citation 보정을 추가한다.
- `backend.py`는 LLM backend 실행 방식을 바꾼다.
- `run_lambo_set1.py`는 selected indices, error continuation, path resolver, local judge integration, official judge optionization을 추가한다.
- `run_loong_judge_local.py`는 baseline standalone local judge를 callable `run_local_judge`로 refactor한다.
- `prepare_exper99_subset.py`와 `run_exper99.sh`는 99개 subset 실행을 위한 운영 계층이다.

높은 확률:

- `common.py`의 short name title enrichment는 financial entity/relation matching을 돕기 위한 변경이다. 코드상 효과는 명확하지만, 실제 성능 영향은 데이터에 따라 다르므로 "효과 크기"는 이 문서에서 단정하지 않는다.
- `answer_writer.py`의 paper Reference/Citation 보정은 paper 계열 실패를 줄이려는 변경으로 보인다. 코드상 동작은 명확하지만, 모든 paper task에서 항상 옳은지는 별도 평가가 필요하다.
- `backend.py`의 server mode는 vLLM 실행 안정성과 속도를 위해 추가된 운영 변경으로 보인다. 실제 환경별 성능/메모리 영향은 별도 측정이 필요하다.

추가 검증 필요:

- fixed baseline에서 치팅 수정이 어느 파일을 건드렸는지 확인한 뒤 `search_r1.py`와 `refine_extractor.py` 병합 충돌을 재평가해야 한다.
- dawonv3의 expanded anchor window가 fixed baseline의 "선택된 chunk만 사용" 원칙과 어떻게 공존해야 하는지 팀 정책 확인이 필요하다. 이 문서는 해당 변경을 치팅으로 판단하지 않는다.
- local judge와 official judge 설정은 dawonv4 실행 환경의 Loong config에 맞춰 재확인해야 한다.
- exper99 wrapper는 대규모 실행용이므로 dawonv4에서 필요할 때 별도 dry run이 필요하다.

## 9. dawonv4 이식 우선순위

1단계: core data-flow를 먼저 복원한다. `anchor_relations.py`, `anchor_agent.py` relation schema, `search_r1.py` coverage/relation scoring, `refine_extractor.py` coverage loop와 provenance fields를 fixed baseline 위에 수동 병합한다.

2단계: answer와 runner를 연결한다. `answer_writer.py`의 relation/provenance-aware guard를 반영하고, `manifest.py`, `paths.py`, `run_lambo_set1.py`, `run_loong_judge_local.py`의 selected indices/local judge/path resolver 변경을 붙여 set1 10개가 재현되게 만든다.

3단계: 운영 편의와 대규모 실험을 붙인다. `backend.py` server mode, `prepare_qwen_workspace.py`, `run.sh`, `wait_run.sh`, `prepare_exper99_subset.py`, `run_exper99.sh`, dawonv4 README를 환경에 맞게 정리한다.
