# dawonv5 Design Report

## A. 현재 dawonv4의 anchor summary 생성 경로

1. **Unit Segmentation** (`anchor_agent.py:56-118`): 문서 → block 분할 → 작은 block 병합 → max_units_per_doc 제한
2. **LLM Anchor Generation** (`anchor_agent.py:198-319`): `anchor_system.txt` + `anchor_user.txt`로 unit 그룹핑, anchor_title/summary/key_entities 생성
3. **Fallback Tiling** (`anchor_agent.py:165-196`): LLM 실패 시 4-unit 단위로 기계적 분할
4. **Anchor Annotation** (`anchor_relations.py:109-188`): annotate_anchor()로 role_candidates, entities, targets, value_hints, provenance 부착
5. **Relation Building** (`anchor_relations.py:223-312`): prev_next, same_section, supports, disambiguates, conflicts_with 등 relation 구축

**현재 anchor 필드 (v4)**:
- 기본: anchor_id, doc_id, doc_title, order, anchor_title, summary, key_entities, packet_span, char_span, unit_ids, text, preview, prev_anchor_id, next_anchor_id
- annotation: anchor_role_candidates, anchor_entities, anchor_targets, anchor_value_hints, provenance_hints
- relations: anchor_relations, relation_summary

## B. 현재 metadata의 한계

1. **소속(owner) 정보 없음**: 어떤 회사/판결/논문의 것인지 미식별
2. **semantic role이 조잡**: role_candidates는 table_value/value_evidence/entity_binding/background_context 등 4~5종 뿐. 문서 내 기능(정의/주장/근거/판단 등) 미분류
3. **explicit citation 미추출**: "Table 3", "위 표", "supra" 등 명시적 cross-reference 무시
4. **시간/단위 정보 없음**: 기간, 금액 단위 미추출
5. **content type 없음**: 표/그림/목록 등 형태 미분류
6. **summary가 LLM 원문 그대로**: 검색에 최적화된 구조화 정보 부족
7. **search scoring에서 v4 annotation 미활용**: search_r1.py의 _score_anchor()가 anchor_role_candidates를 사용하지 않음

## C. dawonv5에서 추가한 schema

`AnchorEnrichedMetadata` (anchor_schemas.py) — dawonv4 annotation 위에 layered:
- anchor_owner_type, anchor_owner_name, anchor_parent_heading
- anchor_content_type, anchor_semantic_role
- anchor_time_scope, anchor_unit_hints
- citation_edges_out/in, citation_summary
- v5_provenance_flags, v5_confidence
- rendered_summary

`CitationEdge` — explicit cross-reference edge

## D. 새 module 설계

| Module | 역할 |
|--------|------|
| `anchor_schemas.py` | AnchorEnrichedMetadata, CitationEdge dataclass |
| `anchor_context_enricher.py` | owner/role/content_type/time/unit 추출. v4의 role_candidates 활용 |
| `anchor_citation_graph.py` | explicit citation + structural edge 추출. v4의 anchor_relations와 별도 저장 |
| `anchor_summary_renderer.py` | structured metadata → ≤300자 rendered summary |

## E. 수정 파일 목록

| 파일 | 변경 |
|------|------|
| `anchor_agent.py` | enrichment pipeline 호출, v5_metadata/rendered_summary 부착, doc_map에 v5 필드 추가 |
| `search_r1.py` | _score_anchor()에 v5 semantic role/citation/confidence 보너스, chunk_summary_text에 rendered_summary 사용 |
| `refine_extractor.py` | _info_payload()에서 rendered_summary 우선, semantic_role/owner_name/citation_summary 포함 |

## F. 필드별 추출 방식

| 필드 | 방식 | v4 활용 |
|------|------|---------|
| owner_type/name | rule-based (doc_title regex) | — |
| content_type | rule-based (pipe count, regex) | — |
| semantic_role | hybrid (keyword + v4 role_candidates 매핑) | decision_evidence→holding, case_identity→case_fact 등 |
| time_scope | rule-based regex | — |
| unit_hints | rule-based regex | — |
| citation_edges | rule-based (explicit ref regex + structural adjacency) | prev/next_anchor_id 활용 |
| rendered_summary | template 기반 | v4 summary, anchor_entities 활용 |
| confidence | rule-based (filled field count) | — |

## G. 리스크와 fallback

1. **v5_metadata 키 부재**: 모든 v5 코드가 `.get("v5_metadata")` → `isinstance(v5, dict)` 가드 사용. 없으면 v4 동작 그대로.
2. **citation matching 오류**: 매칭 실패 시 edge 생략 (false positive보다 miss 선택)
3. **owner 추론 오류**: doc_title 패턴 실패 시 doc_title 전체를 owner_name으로 사용
4. **LLM 없이도 작동**: 기본 use_llm=False. rule-based만으로 최소 enrichment 보장
5. **rendered_summary가 v4 summary보다 나쁠 때**: search/refine에서 rendered_summary가 빈 문자열이면 v4 summary fallback
