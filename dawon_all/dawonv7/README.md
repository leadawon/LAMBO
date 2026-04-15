# dawonv7

`dawonv7`은 `dawonv6`의 exper99 일부 실행 결과(20 샘플)를 분석한 뒤,
**Refiner 단계에서 `relations.records`가 100% 비어 있었던 치명적 문제**를
해결하기 위해 만든 버전입니다. v6의 anchor enrichment / search / prompt
배선은 그대로 두고, Refiner가 참고할 수 있는 **결정론적 cross-document
citation matching** 레이어를 앞단에 추가했습니다.

v6 README의 v5 → v6 변경사항 설명은 [dawonv6/README.md](../dawon_all/dawonv6/README.md)를 참고하세요 (dawon_all 통합 이후 경로).

---

## v6에서 발견된 문제 (20 샘플 실행 분석)

| # | 증상 | 심각도 |
|---|---|---|
| 1 | `samples/*/relations.json`의 `records_len == 0` (20/20 샘플) — LLM Refiner가 매번 빈 records를 반환 | Critical |
| 2 | level 4(citation chain) 태스크에서 Refiner가 "no direct citation relationship"이라고 단정하고 empty list를 반환 | Critical |
| 3 | `paper_level3_15`의 최종 answer가 `DOC1/DOC2/DOC3` 대신 raw paper title을 key로 사용 (schema violation) | High |
| 4 | Refiner 프롬프트에 **Documents roster** (DOC_id → doc_title)가 없음 — LLM이 "DOC2의 title이 DOC1의 reference list에 나타나는지"를 확인할 수 없음 | Root cause |
| 5 | Per-document 시점의 fact만 있고, docs 간 title 교차매칭이 전혀 수행되지 않음 | Root cause |
| 6 | `paper_level4_21`은 사용자 인터럽트로 중단됨 (anchors 단계까지만 실행) — 코드 버그 아님 | - |

Refiner가 받는 facts에는 `LLaMA`, `MS MARCO` 같은 외부 참조 엔티티만
나열되어 있고, *어떤 DOC가 어떤 DOC의 title을 인용하는가*는 어디에도
명시되지 않습니다. 그래서 LLM이 매번 "인용 체인을 구성할 수 없음" ⇒
empty records로 귀결.

---

## v7의 변경사항

### A. 결정론적 cross-doc citation matcher 신설

새 모듈 [`anchor/cross_doc_citation_matcher.py`](anchor/cross_doc_citation_matcher.py):

- 각 `doc_title`을 정규화한 뒤 다른 docs의 search facts(`fact`,
  `evidence_span`, `entities`, ...)에 들어있는지 완전일치 / 토큰
  coverage(≥0.8) 두 가지 방식으로 매칭.
- 결과를 `cited_doc_id → citing_doc_id` 쌍 리스트와 인접 그래프
  (`{doc_id: {cites, cited_by}}`) 형태로 반환.
- `longest_citation_chain`은 인접 그래프에서 가장 긴 선형 citation
  path를 반환 (level 4 태스크의 hot path).

### B. Refiner 프롬프트에 Documents roster + 후보 edges 주입

- [`anchor/prompts/refine_user.txt`](anchor/prompts/refine_user.txt)에
  새 placeholder `{cross_doc_block}` 추가.
- [`anchor/prompts/refine_system.txt`](anchor/prompts/refine_system.txt)에
  두 블록 사용법 명시 — Documents roster는 DOC_id를 canonical key로
  쓰게 하고, 감지된 citation 후보는 *가설*로 취급하되 근거 없이 전부
  버리지는 말도록 지시.

### C. Refiner 실패 시 자동 seed

[`anchor/relation_refiner.py`](anchor/relation_refiner.py)에서 LLM이
여전히 `records: []`를 돌려주면, detector가 찾은 citation edge를 graph
records로 자동 seed합니다. 이로써 v6에서처럼 Refiner 실패가 그대로
Answer Writer의 empty output으로 전파되는 경로를 차단.

### D. `relations.json`에 부가 정보 저장

- `cross_doc`: 전체 매칭 결과(roster + matches + adjacency)
- `longest_chain`: 결정론적 longest chain
- 별도 파일 `samples/*/cross_doc_citations.json` 에도 동일 내용 dump.

### E. Answer Writer 후처리 훅

[`anchor/answer_writer.py`](anchor/answer_writer.py)의 `_enforce_doc_keys`
— instruction에 `DOC1/DOC2/...` 문자열이 등장하는데 최종 dict의 key에
DOC_id가 하나도 없으면 이후 확장을 위해 훅을 둔 자리. (v7에서는
conservative하게 원본을 보존; 향후 title→doc_id 리매핑 보강 예정.)

---

## 유지된 것 (v6와 동일)

- anchor schemas / context enricher / citation graph / summary renderer
- search agent / refine extractor / search_r1
- runner 구조 (AnchorAgent → SearchAgent → RelationRefiner → AnswerWriter)
- prompt 템플릿 7종 (anchor / extract / search / search_r1 / refine / answer)
  중 refine 2종만 수정, 나머지는 그대로
- `LAMBO_DAWONV4_*` 환경변수 prefix (run.sh의 vLLM health-check와의 호환 유지)

---

## 사용법

```bash
# 99-sample StructRAG 공정비교
bash /workspace/LAMBO/dawonv7/run_exper99.sh

# 전체 set1
bash /workspace/LAMBO/dawonv7/run.sh
```

---

## 단위 검증 (이미 통과)

```python
from dawonv7.anchor.cross_doc_citation_matcher import (
    detect_cross_doc_citations, longest_citation_chain,
)
docs = [
  {'doc_id':'DOC1','doc_title':'TinyLlama: An Open-Source Small Language Model'},
  {'doc_id':'DOC2','doc_title':'OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models'},
  {'doc_id':'DOC3','doc_title':'Mixtral of Experts'},
]
doc_results = [
  {'doc_id':'DOC1','items':[{'fact':'Llama 2'}]},
  {'doc_id':'DOC2','items':[{'fact':'TinyLlama: An Open-Source Small Language Model'}]},
  {'doc_id':'DOC3','items':[{'fact':'OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models'}]},
]
out = detect_cross_doc_citations(docs=docs, doc_results=doc_results)
assert longest_citation_chain(out['adjacency']) == ['DOC3','DOC2','DOC1']
```

---

## 검증 체크리스트 (실 실행 후)

- [ ] `samples/*/relations.json`에 `cross_doc` / `longest_chain` 필드 및
      non-empty `records` 확인
- [ ] v6 20-sample 대비 `records_len > 0` 비율 상승
- [ ] level 4 citation chain 정답률 상승 (v6 baseline: 사실상 0)
- [ ] Documents roster 반영으로 `paper_level3_15` 류 schema 위반 재현 없음
