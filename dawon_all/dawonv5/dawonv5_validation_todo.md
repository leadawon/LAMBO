# dawonv5 Validation Checklist

## Backward Compatibility
- [ ] dawonv4의 run_lambo_set1.py가 dawonv5에서 import 에러 없이 실행되는지 확인
- [ ] anchors.json에 v4 필드(anchor_id, summary, key_entities, anchor_role_candidates 등) 모두 유지되는지 확인
- [ ] v5_metadata 키가 없는 anchor에서 search_r1.py scoring이 정상 동작하는지 확인
- [ ] v5_metadata 키가 없는 anchor에서 refine_extractor.py _info_payload가 정상 동작하는지 확인

## Import Check
- [x] anchor_schemas.py imports cleanly
- [x] anchor_context_enricher.py imports cleanly
- [x] anchor_citation_graph.py imports cleanly
- [x] anchor_summary_renderer.py imports cleanly
- [x] Modified anchor_agent.py imports cleanly
- [x] Modified search_r1.py imports cleanly
- [x] Modified refine_extractor.py imports cleanly

## Metadata Fallback
- [ ] text가 빈 anchor → v5_metadata 기본값, 크래시 없음
- [ ] anchor_title이 없는 anchor → parent_heading 빈 문자열
- [ ] doc_title이 없는 anchor → owner_name graceful fallback
- [ ] v4 annotation (role_candidates 등)이 없는 anchor → semantic_role "unknown"으로 fallback

## Citation Edge Extraction
- [x] Financial sample: table-adjacent paragraph에서 elaborates_table 감지
- [x] Legal sample: cites_previous edge 감지
- [ ] Paper sample: [N] bracket citation → attribution anchor 매칭
- [ ] self-referencing edge (source == target) 없음 확인

## Owner Inference
- [x] Financial: 회사명 추출 (广西能源股份有限公司)
- [x] Legal: court_case 타입 식별
- [ ] Paper: paper 타입 식별
- [ ] Unknown doc: "unknown" 타입 fallback

## Summary Quality
- [x] rendered_summary ≤ 300자
- [x] owner identity 포함
- [x] semantic role 포함 (available 시)
- [ ] hallucination 없음 확인 (anchor text에 없는 정보 금지)
- [ ] "This section discusses..." 패턴 미발생

## Search/Refine Hook
- [ ] search_r1.py v5 scoring이 enriched anchor에 보너스 부여 (rule_hits 출력에서 v5_role 확인)
- [ ] refine_extractor.py info_payload에 semantic_role, owner_name 포함 확인
- [ ] 전체 retrieval 품질이 v4 대비 저하되지 않는지 spot check (5 samples)

## Performance
- [ ] enrichment 추가 시간 < 0.5s per document (LLM 없이)
- [ ] 50+ anchor 문서에서 메모리 이슈 없음
