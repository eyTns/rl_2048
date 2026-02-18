# CLAUDE.md

## 작업 범위
- Python(`src/`) 작업은 보류한다. JavaScript(`static/js/`) 작업을 우선한다.

## 문서 편집 권한

- AI 편집 가능 문서
  - CLAUDE.md
  - docs/usage.md
  - docs/dqn_experiment_guide.md
  - docs/reinforcement_learning.md
  - docs/ml_visualization_plan.md
- AI 읽기 전용 문서
  - docs/conventions.md
  - docs/backlog.md
  - docs/karpathy-guidelines.md
- 새 문서 생성 시 1행에 `<!-- AI 수정 가능 -->` 또는 `<!-- 사용자 전용 -->` 표시를 넣는다

## 문서 작성
- 문서 항목에 괄호, 덧붙임, 화살표 등을 사용하기 전에 해당 내용이 꼭 필요한지, 다른 맥락에서 이미 표시되었는지 검토한다
- todo에 있는 목록은 사용자가 직접 수정해야 한다

## 환경 관리
- 패키지 관리와 가상환경은 `uv`를 사용한다
- 테스트 실행: `uv run pytest`
- 스크립트 실행: `uv run python <script.py>`
- 의존성 추가: `uv add <package>`
- 개발 의존성 추가: `uv add --dev <package>`

## 행동
- 코드를 수정할지 질문하지 마시오. 이때마다 사용자는 거절한 것으로 간주하시오.


## 간단한 리팩토링
- 코드 중간에 import 가 있으면 파일의 최상단으로 옮길 수 있다
- Python 3.10 이상에서는 타입 힌트가 `Optional[T]` 이면 `T | None` 으로 바꿀 수 있다
- Python 3.9 이상에서는 타입 힌트가 `Dict`, `List`, `Tuple` 이면 `dict`, `list`, `tuple` 로 바꿀 수 있다

## 리팩토링 절차
1. 테스트가 통과하는 상태에서 시작한다
2. 하나의 리팩토링을 적용한다
3. 테스트를 실행하여 동작이 바뀌지 않았는지 확인한다
4. 커밋한다
5. 다음 리팩토링으로 넘어간다

## 코드리뷰 해결 방법
1. 브랜치를 pull 해서 새로 추가된 리뷰 문서를 확인한다
2. 각 항목을 다음과 같이 처리한다
   - 고치기 쉽다면 고친다
   - 버그가 있다면 고친다
   - 리팩토링 의견은 반영한다
   - 문서 보강 의견은 반영한다
   - 기능 추가를 요구한다면 구현 맥락을 파악한다
     - 맥락을 모르겠다면 질문한다
3. 맥락과 사용자의 의도는 문서에 정리한다

## Claude Code 세션 관리법
1. 새로운 세션을 연다
2. 베이스로 할 브랜치를 선택한다
3. 무엇을 구현하려는지 주제를 이야기한다
4. 클로드코드가 새 브랜치를 생성한다
5. 브랜치가 잘 연결되어 있는지 파악한다
6. 구현을 한다
