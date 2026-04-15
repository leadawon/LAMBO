# DawonV3 LAMBO Runner

이 폴더는 기존 실행본에서 분리한 `dawonv3` 전용 실행 버전입니다.

## Qwen 32B 준비

로컬 모델 기본 경로는 `/workspace/qwen/Qwen2.5-32B-Instruct`입니다.

이미 이 워크스페이스에는 `/workspace/StructRAG/model/Qwen2.5-32B-Instruct`가 있어서, `dawonv3`에서는 그 모델을 다시 62GB 복사하지 않고 `/workspace/qwen` 아래에 심볼릭 링크로 연결해서 씁니다.

준비만 먼저 하고 싶으면:

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv3/prepare_qwen_workspace.py
```

## 실행

권장 실행:

```bash
bash /workspace/LAMBO/dawonv3/run.sh
```

이 명령은 아래를 자동으로 합니다.

- `/workspace/qwen/Qwen2.5-32B-Instruct` 준비
- `ragteamvenv` 파이썬 사용
- 기본값으로 vLLM OpenAI-compatible backend 서버 시작
- 파이프라인과 local judge는 그 backend 서버를 통해 Qwen 32B 호출
- set1 10개를 한 번에 실행
- 기본으로 `--force --skip_judge`를 넣어서 공식 judge 없이 새로 계산
- 실행이 끝나면 backend 서버도 자동 종료

기존 엔트리포인트를 직접 실행해도 됩니다.

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv3/run_set1_10.py --skip_judge
```

출력 로그는 기본적으로 아래에 저장됩니다.

```text
/workspace/LAMBO/dawonv3/logs/lambo_set1_10
```

## 주요 옵션

```bash
/workspace/venvs/ragteamvenv/bin/python /workspace/LAMBO/dawonv3/run_set1_10.py \
  --input_path /workspace/Plan_Search_RAG/Loong/data/loong_process.jsonl \
  --output_dir /workspace/LAMBO/dawonv3/logs/lambo_set1_10_doccall \
  --skip_judge \
  --force
```

## Judge 옵션 설명

- `structured_eval.json`
  - 항상 생성됩니다.
  - gold answer와 prediction을 구조적으로 비교한 EM / pair F1입니다.
- `--skip_judge`
  - Loong 원본 judge를 건너뜁니다.
  - 이 judge는 `step3_model_evaluate.py`와 `step4_cal_metric.py`를 통해 돌아가며, gold answer와 prediction을 LLM으로 평가하는 `LLM-as-eval` 계열입니다.
  - 현재 `dawonv3` 복사본에서는 원본 코드 흐름을 유지해서 `eval_model=gpt4o.yaml`을 호출합니다.
- `--skip_local_judge`
  - `dawonv3`에서 추가한 local Qwen judge를 건너뜁니다.
  - local judge는 Qwen 32B에게 `question + gold answer + prediction`을 주고, Loong judge 프롬프트 형식으로 1~100 점수를 매깁니다.

실무적으로는 아래 조합을 권장합니다.

```bash
bash /workspace/LAMBO/dawonv3/run.sh
```

이렇게 하면:

- 구조 기반 EM / pair F1은 계산하고
- 공식 judge는 생략하고
- local Qwen 32B judge 점수는 같이 계산합니다.

결과 파일은 보통 아래를 보면 됩니다.

- `/workspace/LAMBO/dawonv3/logs/lambo_set1_10/reports/structured_eval.json`
- `/workspace/LAMBO/dawonv3/logs/lambo_set1_10/reports/loong_judge_local_qwen32b.json`
- `/workspace/LAMBO/dawonv3/logs/lambo_set1_10/reports/summary_report.md`

## 환경변수

로컬 모델/데이터 경로가 기본값과 다르면 아래 환경변수로 덮어쓸 수 있습니다.

```bash
export LAMBO_DAWON_INPUT_PATH=/workspace/Plan_Search_RAG/Loong/data/loong_process.jsonl
export LAMBO_DAWON_LOONG_JSONL=/workspace/Plan_Search_RAG/Loong/data/loong.jsonl
export LAMBO_DAWON_LOONG_SRC=/workspace/LAMBO/Loong/src
export LAMBO_DAWON_LOONG_MODEL_DIR=/workspace/LAMBO/Loong/config/models
export LAMBO_DAWON_MODEL_DIR=/workspace/qwen/Qwen2.5-32B-Instruct
export LAMBO_DAWON_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWON_LLM_BACKEND=server
export LAMBO_DAWON_SERVER_PYTHON=/workspace/venvs/ragteamvenv/bin/python
export LAMBO_DAWON_SERVER_HOST=127.0.0.1
export LAMBO_DAWON_SERVER_PORT=1225
export LAMBO_DAWON_SERVER_MODEL_NAME=Qwen
export LAMBO_DAWON_SERVER_LOG=/workspace/LAMBO/dawonv3/logs/backend_server.log
```

## 참고

- 코드 수정은 `dawonv3` 아래에서만 진행하면 됩니다.
- 원본 `script/anchor`는 이 버전이 import하지 않습니다.
- `dawonv3`는 두 경로를 모두 지원합니다.
- 기본 `run.sh`는 backend 서버 모드입니다.
- 서버 없이 직접 추론으로 돌리고 싶으면 `LAMBO_DAWON_LLM_BACKEND=transformers bash /workspace/LAMBO/dawonv3/run.sh` 를 쓰면 됩니다.
