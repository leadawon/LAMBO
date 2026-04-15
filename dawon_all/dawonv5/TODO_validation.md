# DawonV4 Validation TODO

## Import 에러 점검

1. `python -m compileall dawonv4`로 문법과 import path 문자열을 먼저 확인한다.
2. `python /workspace/LAMBO/dawonv4/run_set1_10.py --help`가 최신 runner CLI를 출력하는지 확인한다.
3. `dawonv4.anchor.backend` import 시 `/workspace/meta-cognitive-RAG/src`가 접근 가능한지 확인한다.
4. server backend를 쓸 때 `vllm`이 `LAMBO_DAWONV4_SERVER_PYTHON` 환경의 Python에 설치되어 있는지 확인한다.

## Baseline 대비 추가된 파일 목록

1. `dawonv4/anchor/anchor_relations.py`: dawonv3 relation-aware anchor enrichment helper.
2. `dawonv4/anchor/paths.py`: dawonv4 우선 env prefix와 기존 dawon prefix fallback을 지원하는 path resolver.
3. `dawonv4/prepare_qwen_workspace.py`: Qwen model symlink 준비 helper.
4. `dawonv4/run_set1_10.py`: package root entrypoint.
5. `dawonv4/run.sh`: vLLM server wrapper와 최신 baseline runner 연결.
6. `dawonv4/README.md`: dawonv4 실행 문서.
7. `dawonv4/TODO_validation.md`: 이 검증 체크리스트.

## Merge된 파일 목록

1. `dawonv4/anchor/anchor_agent.py`: 최신 baseline anchor tiling 유지, `enrich_anchor_graph` 호출과 relation metadata 출력만 추가.
2. `dawonv4/anchor/search_agent.py`: 최신 baseline SearchAgent 유지, relation frontier ordering과 provenance metadata 전달만 추가.
3. `dawonv4/anchor/relation_refiner.py`: 최신 baseline RelationRefiner 유지, fact formatter에 provenance/relation context만 추가.
4. `dawonv4/anchor/backend.py`: 최신 baseline local backend 유지, `LAMBO_DAWONV4_LLM_BACKEND=server`일 때만 OpenAI-compatible backend 선택.
5. `dawonv4/anchor/run_lambo_set1.py`: 최신 baseline agentic runner 유지, dawonv4 path resolver와 selected-index 옵션만 추가.
6. `dawonv4/anchor/common.py`: 최신 baseline common 유지, `证券简称` title enrichment만 추가.
7. `dawonv4/anchor/manifest.py`: 최신 baseline manifest 유지, selected-index manifest builder만 추가.
8. `dawonv4/anchor/run_loong_judge_local.py`: dawonv4 경로 기본값만 정리. 최신 runner는 기본적으로 `llm_judge.py`를 사용한다.

## 실행 전 확인할 환경변수

```bash
export LAMBO_DAWONV4_INPUT_PATH=/workspace/LAMBO/Loong/data/loong_process.jsonl
export LAMBO_DAWONV4_MODEL_DIR=/workspace/qwen/Qwen2.5-32B-Instruct
export LAMBO_DAWONV4_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWONV4_LLM_BACKEND=server
export LAMBO_DAWONV4_SERVER_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWONV4_SERVER_HOST=127.0.0.1
export LAMBO_DAWONV4_SERVER_PORT=1225
export LAMBO_DAWONV4_SERVER_MODEL_NAME=Qwen
export LAMBO_DAWONV4_BASE_URL=http://127.0.0.1:1225/v1
```

기존 실험 환경과의 호환을 위해 `LAMBO_DAWON_*` fallback도 남겼지만, 새 검증 로그에는 `LAMBO_DAWONV4_*`를 우선 사용한다.

## Smoke test 커맨드

```bash
cd /workspace/LAMBO
python -m compileall dawonv4
python /workspace/LAMBO/dawonv4/run_set1_10.py --help
python /workspace/LAMBO/dawonv4/prepare_exper99_subset.py --help
bash /workspace/LAMBO/dawonv4/run_exper99.sh --help
LAMBO_DAWONV4_LLM_BACKEND=server bash /workspace/LAMBO/dawonv4/run.sh --max_items 1
```

직접 local transformers backend를 확인할 때:

```bash
LAMBO_DAWONV4_LLM_BACKEND=transformers bash /workspace/LAMBO/dawonv4/run.sh --max_items 1
```

## skip_judge / local_judge / backend server 확인 포인트

1. `--skip_judge`는 dawonv4 runner에 넣지 않는다. 최신 baseline runner는 structured eval과 `llm_judge.py` 기반 local judge를 내부에서 실행한다.
2. dawonv3의 official Loong judge orchestration은 dawonv4 기본 runner로 옮기지 않았다.
3. `run_loong_judge_local.py`는 별도 호환 유틸리티로 남아 있지만, dawonv4 기본 평가 경로는 `dawonv4/anchor/llm_judge.py`다.
4. server backend는 `LAMBO_DAWONV4_LLM_BACKEND=server`일 때만 선택되어야 한다.
5. backend server log 기본 경로는 `/workspace/LAMBO/dawonv4/logs/backend_server.log`다.
6. server 시작 실패 시 `LAMBO_DAWONV4_SERVER_HEALTH_TIMEOUT`, `LAMBO_DAWONV4_SERVER_MAX_MODEL_LEN`, `LAMBO_DAWONV4_SERVER_TP_SIZE`를 먼저 확인한다.
