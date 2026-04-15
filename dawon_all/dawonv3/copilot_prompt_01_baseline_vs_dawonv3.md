다음 작업만 수행하세요. 아직 코드를 수정하지 말고, 비교 분석과 문서화만 하세요.

목표
- 이 저장소에서 baseline은 리포지토리 루트의 LAMBO 구현이다.
- dawonv3는 baseline 위에 기능을 추가한 파생 버전이다.
- 현재 baseline에는 치팅 로직이 포함되어 있을 수 있으나, 지금 단계에서는 그 로직을 찾거나 고치지 말고 무시한다.
- 지금 필요한 것은 dawonv3가 baseline 대비 어떤 기능을 새로 추가했는지, 어떤 파일을 수정했는지, 어떤 데이터 흐름이 바뀌었는지 정확히 분리해내는 것이다.
- 이후 팀원이 baseline의 치팅 로직을 수정하면, 우리는 이 문서를 보고 fixed baseline 위에 dawonv3 기능만 안전하게 다시 얹어 dawonv4를 만들 것이다.

중요 제약
- 치팅 로직 수정, 리팩토링, 버그 수정, 동작 변경을 하지 마라.
- baseline의 잘못된 부분을 dawonv3에 그대로 반영하려고 하지 마라.
- 목적은 오직 “dawonv3 고유 변경사항 추출”이다.
- 단순 import path 변경, 네임스페이스 변경, 실행 wrapper 추가와 실제 알고리즘 변경을 반드시 구분하라.
- 추측하지 말고, 코드 근거가 있는 내용만 적어라.

비교 범위
1. baseline 쪽 핵심 구현
- script/anchor 전체
- 루트 README 및 실행 진입점에서 baseline 파이프라인 설명에 해당하는 부분

2. dawonv3 쪽
- dawonv3/anchor 전체
- dawonv3/README.md
- dawonv3/run.sh
- dawonv3/wait_run.sh
- dawonv3/run_set1_10.py
- dawonv3/prepare_qwen_workspace.py
- dawonv3/prepare_exper99_subset.py
- dawonv3/run_exper99.sh

분석 방법
- 먼저 파일 단위 diff inventory를 만든다.
  - baseline에만 있는 파일
  - dawonv3에만 새로 생긴 파일
  - 둘 다 있지만 내용이 바뀐 파일
  - 사실상 wrapper/경로 변경만 있는 파일
  - 실제 로직이 바뀐 파일
- 그 다음 로직 단위 diff를 정리한다.
  - anchor 생성 단계
  - search/planning 단계
  - refine/evidence extraction 단계
  - answer writing 단계
  - evaluation/judge 단계
  - 실행 환경 및 모델 경로 해결 단계
- 마지막으로 “dawonv4 재적용 단위”로 묶어라.
  예:
  - anchor relation enrichment 묶음
  - relation-aware search 묶음
  - iterative refine coverage loop 묶음
  - local judge 묶음
  - run script / path resolver 묶음

반드시 산출해야 할 문서
- dawonv3/BASELINE_DIFF_ANALYSIS.md

문서 형식
1. 개요
- baseline과 dawonv3의 관계를 5~10줄로 설명
- 이번 문서의 목적을 설명

2. 한눈에 보는 변경 요약
- 새 파일 목록
- 수정 파일 목록
- 단순 실행/경로 변경 파일 목록
- 핵심 알고리즘 변경 파일 목록

3. 파일별 상세 분석
각 파일마다 아래 형식으로 정리
- 파일 경로
- 분류: new / modified / wrapper / config-path / likely-unchanged
- baseline 대비 차이 한 줄 요약
- 상세 변경 사항
- 입력/출력 구조 변화
- 이 파일이 의존하는 dawonv3 전용 요소
- dawonv4로 옮길 때 주의점

4. 기능 단위 재적용 계획
다음 관점으로 묶어서 정리
- 반드시 함께 옮겨야 하는 변경 묶음
- 독립적으로 옮길 수 있는 변경 묶음
- baseline fixed 버전과 충돌 가능성이 높은 부분
- 충돌 가능성은 낮지만 놓치기 쉬운 부분

5. dawonv4 제작 체크리스트
- 순서형 체크리스트로 작성
- “fixed baseline에서 시작 → 어떤 파일을 추가 → 어떤 파일을 patch → 어떤 실행 검증” 순으로 적기

6. 절대 하지 말아야 할 것
- baseline의 치팅 로직을 dawonv3 쪽으로 역이식하는 것
- baseline 전체를 dawonv3로 통째로 복사하는 것
- import path만 맞추고 실제 데이터 흐름 차이를 놓치는 것
- search/refine/relation 로직을 부분적으로만 옮겨서 인터페이스를 깨는 것

작성 원칙
- 한국어로 작성
- 표보다 문장 위주로 자세히 설명
- 각 주장에 대해 가능한 한 코드 근거를 곁들여라
- “확실함”, “높은 확률”, “추가 검증 필요”를 구분해서 표시하라

마지막으로 해야 할 일
- 문서 작성이 끝나면, dawonv3 기능을 dawonv4로 옮길 때의 우선순위를 1, 2, 3단계로 짧게 정리해서 문서 마지막에 추가하라.
