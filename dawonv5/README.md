# DawonV4 LAMBO Runner

`dawonv4`는 최신 baseline `script/anchor`를 기본본으로 두고, dawonv3의 relation-aware anchor metadata와 독립 실행용 path/server wrapper만 선별 이식한 버전입니다.

핵심 파이프라인은 최신 baseline 흐름을 유지합니다.

```text
AnchorAgent -> SearchAgent / ExtractAgent -> RelationRefiner -> AnswerWriter -> llm_judge
```

dawonv3의 예전 `SearchR1 + RefineExtractor` 루프와 official Loong judge orchestration은 다시 도입하지 않았습니다.

## Qwen 32B 준비

로컬 모델 기본 경로는 `/workspace/qwen/Qwen2.5-32B-Instruct`입니다. 기존 모델 디렉토리가 있으면 아래 명령으로 `/workspace/qwen` 아래에 심볼릭 링크를 준비할 수 있습니다.

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv4/prepare_qwen_workspace.py
```

기본 후보 경로 외의 모델을 쓰려면 다음 중 하나를 지정합니다.

```bash
export LAMBO_DAWONV4_EXISTING_MODEL_SOURCE=/path/to/Qwen2.5-32B-Instruct
export LAMBO_DAWONV4_MODEL_DIR=/workspace/qwen/Qwen2.5-32B-Instruct
```

기존 dawon 계열 환경과의 호환을 위해 `LAMBO_DAWON_*` 값도 fallback으로 읽지만, 새 실행에서는 `LAMBO_DAWONV4_*`를 우선합니다.

## 실행

권장 실행:

```bash
bash /workspace/LAMBO/dawonv4/run.sh
```

이 wrapper는 기본적으로 vLLM OpenAI-compatible backend 서버를 띄운 뒤 최신 baseline runner를 실행합니다. 서버 없이 baseline의 local transformers backend를 쓰려면 다음처럼 실행합니다.

```bash
LAMBO_DAWONV4_LLM_BACKEND=transformers bash /workspace/LAMBO/dawonv4/run.sh
```

root entrypoint를 직접 실행할 수도 있습니다.

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv4/run_set1_10.py \
  --output_dir /workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10 \
  --force
```

특정 인덱스만 실행하려면 다음 옵션을 사용합니다.

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv4/run_set1_10.py \
  --selected_indices 914,725 \
  --force
```

SET1 balanced 99개 실험은 아래 wrapper를 사용합니다. 이 wrapper는 확인된 context-length 실패 index를 제외하고 domain별 33개씩 채운 `dawonv4/data/loong_set1_balanced99_indices.json`을 준비한 뒤 실행합니다.

```bash
bash /workspace/LAMBO/dawonv4/run_exper99.sh
```

기본 출력 경로는 아래입니다.

```text
/workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10
```

## 환경변수

```bash
export LAMBO_DAWONV4_INPUT_PATH=/workspace/LAMBO/Loong/data/loong_process.jsonl
export LAMBO_DAWONV4_MODEL_DIR=/workspace/qwen/Qwen2.5-32B-Instruct
export LAMBO_DAWONV4_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWONV4_LLM_BACKEND=server
export LAMBO_DAWONV4_SERVER_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWONV4_SERVER_HOST=127.0.0.1
export LAMBO_DAWONV4_SERVER_PORT=1225
export LAMBO_DAWONV4_SERVER_MODEL_NAME=Qwen
export LAMBO_DAWONV4_SERVER_LOG=/workspace/LAMBO/dawonv4/logs/backend_server.log
export LAMBO_DAWONV4_BASE_URL=http://127.0.0.1:1225/v1
```

## 결과 파일

```text
/workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10/manifest.json
/workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10/lambo_predictions.jsonl
/workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10/reports/structured_eval.json
/workspace/LAMBO/dawonv4/logs/lambo_agentic_set1_10/reports/llm_judge.json
```

## 이식 범위

relation-aware 기능은 `anchor_relations.py`를 추가하고, 최신 baseline의 `anchor_agent.py`, `search_agent.py`, `relation_refiner.py`에 metadata/provenance 전달만 연결했습니다. dawonv3의 old runner, old refine loop, old answer writer, 중복 official judge 흐름은 의도적으로 옮기지 않았습니다.
