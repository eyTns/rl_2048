# 알아볼 것 / 할 일

## 원칙

### 문서 작성
- 문서 항목에 괄호, 덧붙임, 화살표 등을 사용하기 전에 해당 내용이 꼭 필요한지, 다른 맥락에서 이미 표시되었는지 검토한다
- todo에 있는 목록은 사용자가 직접 수정해야 한다

### 환경 관리
- 패키지 관리와 가상환경은 `uv`를 사용한다
- 테스트 실행: `uv run pytest`
- 스크립트 실행: `uv run python <script.py>`
- 의존성 추가: `uv add <package>`
- 개발 의존성 추가: `uv add --dev <package>`

### 행동
- 코드를 수정할지 질문하지 마시오. 이때마다 사용자는 거절한 것으로 간주하시오.

### 간단한 리팩토링
- 코드 중간에 import 가 있으면 파일의 최상단으로 옮길 수 있다
- Python 3.10 이상에서는 타입 힌트가 `Optional[T]` 이면 `T | None` 으로 바꿀 수 있다
- Python 3.9 이상에서는 타입 힌트가 `Dict`, `List`, `Tuple` 이면 `dict`, `list`, `tuple` 로 바꿀 수 있다

### 일반적인 코드 리팩토링

리팩토링이란 외부 동작을 바꾸지 않으면서 코드의 내부 구조를 개선하는 것이다.
반드시 테스트가 통과하는 상태에서 시작하고, 한 번에 하나씩 작은 단위로 적용한다.

#### 1. 이름 개선 (Rename)

변수, 함수, 클래스의 이름이 역할을 정확히 드러내도록 바꾼다.
코드를 읽는 사람이 이름만 보고 의미를 파악할 수 있어야 한다.

```python
# 나쁜 예
k = rotation_map[action]
x = x / 15.0

# 좋은 예
rotation_count = rotation_map[action]
normalized = x / MAX_LOG2_VALUE
```

#### 2. 매직 넘버를 상수로 (Replace Magic Number with Constant)

의미가 드러나지 않는 숫자 리터럴에 이름을 붙인다.
숫자의 의도가 코드만으로 명확하지 않다면 상수로 추출한다.

```python
# 변경 전
x = x / 15.0
self.board[row, col] = 4 if random.random() < 0.1 else 2

# 변경 후
MAX_POWER = 15  # 2^15 = 32768
FOUR_SPAWN_RATE = 0.1

x = x / MAX_POWER
self.board[row, col] = 4 if random.random() < FOUR_SPAWN_RATE else 2
```

#### 3. 함수 추출 (Extract Function)

하나의 함수가 여러 단계의 일을 하면 각 단계를 별도 함수로 분리한다.
함수 이름이 주석을 대체할 수 있다.

```python
# 변경 전: backward()에서 gradient clipping이 6줄 반복
dw3 = np.clip(dw3, -1.0, 1.0)
dw2 = np.clip(dw2, -1.0, 1.0)
dw1 = np.clip(dw1, -1.0, 1.0)
db3 = np.clip(db3, -1.0, 1.0)
db2 = np.clip(db2, -1.0, 1.0)
db1 = np.clip(db1, -1.0, 1.0)

# 변경 후: 의도가 드러나는 함수로 분리
def _clip_gradients(self, *grads, limit=1.0):
    return [np.clip(g, -limit, limit) for g in grads]

dw3, dw2, dw1, db3, db2, db1 = self._clip_gradients(
    dw3, dw2, dw1, db3, db2, db1
)
```

#### 4. 변수 추출 (Extract Variable)

복잡한 표현식에 이름을 붙여 의도를 드러낸다.

```python
# 변경 전
y = HEADER_HEIGHT + TILE_MARGIN + row * (TILE_SIZE + TILE_MARGIN)

# 변경 후
cell_stride = TILE_SIZE + TILE_MARGIN
y = HEADER_HEIGHT + TILE_MARGIN + row * cell_stride
```

#### 5. 조건문 단순화 (Simplify Conditionals)

**조기 반환(early return)**: 예외 조건을 먼저 처리하여 중첩을 줄인다.

```python
# 변경 전
def get_text_color(self, value):
    if value in TEXT_COLORS:
        return TEXT_COLORS[value]
    return (249, 246, 242)

# 변경 후: dict.get()으로 단순화
def get_text_color(self, value):
    return TEXT_COLORS.get(value, (249, 246, 242))
```

**딕셔너리로 분기 대체**: if-elif 체인이 값 매핑에 불과하면 딕셔너리로 바꾼다.

```python
# 변경 전
if key == pygame.K_UP or key == pygame.K_w:
    action = Game2048.ACTION_UP
elif key == pygame.K_DOWN or key == pygame.K_s:
    action = Game2048.ACTION_DOWN
elif key == pygame.K_LEFT or key == pygame.K_a:
    action = Game2048.ACTION_LEFT
elif key == pygame.K_RIGHT or key == pygame.K_d:
    action = Game2048.ACTION_RIGHT

# 변경 후
KEY_ACTION_MAP = {
    pygame.K_UP: Game2048.ACTION_UP,
    pygame.K_w: Game2048.ACTION_UP,
    pygame.K_DOWN: Game2048.ACTION_DOWN,
    pygame.K_s: Game2048.ACTION_DOWN,
    # ...
}
action = KEY_ACTION_MAP.get(key)
```

#### 6. 중복 제거 (DRY - Don't Repeat Yourself)

같은 패턴이 반복되면 하나로 합친다.
단, 우연히 비슷해 보이는 코드를 억지로 합치지는 않는다.

```python
# 변경 전: web_ui.py에서 응답 생성이 3곳에서 반복
def state():
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}
def move(data):
    # ...
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}
def reset():
    # ...
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}

# 변경 후
def _game_response():
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}
```

#### 7. 매개변수 객체 도입 (Introduce Parameter Object)

관련된 매개변수가 여러 개 묶여 다니면 객체로 묶는다.

```python
# 변경 전: 콜백에 매개변수가 6개
self.on_episode_end_callback(
    episode_num=self.episode_count,
    steps=step_num,
    score=self.env.score,
    max_tile=int(self.env.board.max()),
    losses=losses,
    epsilon=self.epsilon
)

# 변경 후: 데이터 클래스로 묶음
@dataclass
class EpisodeResult:
    episode_num: int
    steps: int
    score: float
    max_tile: int
    losses: list[float]
    epsilon: float

self.on_episode_end_callback(EpisodeResult(...))
```

#### 8. 죽은 코드 제거 (Remove Dead Code)

사용하지 않는 import, 변수, 함수, 주석 처리된 코드를 제거한다.
버전 관리 시스템(git)이 이력을 보존하므로 "혹시 필요할까 봐" 남길 필요 없다.

```python
# 제거 대상의 예
from dataclasses import dataclass, field  # field를 쓰지 않으면 제거
```

#### 9. 클래스 추출 (Extract Class)

한 클래스가 서로 다른 두 가지 책임을 가지면 분리한다.

```python
# 변경 전: Game2048UI가 렌더링과 입력 처리를 모두 담당
class Game2048UI:
    def draw_tile(self, ...): ...
    def draw_header(self, ...): ...
    def draw_game_over(self, ...): ...
    def handle_input(self, key): ...
    def run(self): ...

# 변경 후: 렌더러를 분리
class BoardRenderer:
    def draw_tile(self, ...): ...
    def draw_header(self, ...): ...
    def draw_game_over(self, ...): ...

class Game2048UI:
    def __init__(self):
        self.renderer = BoardRenderer(self.screen)
    def handle_input(self, key): ...
    def run(self): ...
```

#### 10. 함수 인라인 (Inline Function)

함수 본문이 이름만큼이나 명확하고, 한 곳에서만 호출되면 인라인한다.
불필요한 간접 참조를 줄인다.

```python
# 인라인 대상의 예: 본문이 한 줄이고 이름이 새 정보를 주지 않는 경우
def get_state(self):
    return self.board.copy()

# 호출부에서 직접 사용하는 것이 더 명확할 수 있다
state = self.board.copy()
```

단, `get_state()`처럼 외부 API의 일부이거나 여러 곳에서 호출되면 유지한다.

#### 리팩토링 절차 요약

1. 테스트가 통과하는 상태에서 시작한다
2. 하나의 리팩토링을 적용한다
3. 테스트를 실행하여 동작이 바뀌지 않았는지 확인한다
4. 커밋한다
5. 다음 리팩토링으로 넘어간다

### 코드리뷰 해결 방법
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

## 알아볼 내용
- [ ] 학습이 어떤 계산식으로 이루어지는지 (Q-Learning 수식)
- [x] Q-Learning과 Deep Q-Learning과 RL의 차이
- [ ] Policy가 구체적으로 어떻게 구현되는지
- [x] DQN은 구체적으로 어떤 방식으로 동작하는지
- [ ] Policy와 Reward의 관계
- [ ] Q값을 정규화(normalize) 하는가?
- [ ] 머신러닝 시스템 설계 방법
- [ ] DQN의 레이어 구성
- [ ] 차원수와 모델 용량
- [ ] 회전/대칭 상태를 동일하게 처리하는 테크닉
- [ ] 학습 과정 시각화
- [ ] 2048 구현 PR 검토하기
- [ ] 게임 시각화
- [x] 게임 플레이 UI 생성
- [ ] 게임 플레이 UI PR 검토하기
- [ ] 머신러닝 장면 시각화
- [x] PR이 있을 때 추가 작업 시 브랜치 관리 방법
- [ ] Delayed Reward 개념 확인
- [ ] Discount Factor 개념 확인
- [ ] 시스템, 모듈설계
- [ ] 모듈과 레이어 정의하기
- [ ] 딥큐러닝 학습시 레이어 크기, 파라미터수 결정하기
- [ ] 적당한 이동방향을 선택할 전략
- [ ] 1~2수 미리읽기 전략 (모든 경우를 탐색)
- [ ] 각 수를 랜덤 시뮬레이션 여러 번 해보는 전략
- [ ] Q값은 증가하기만 하는지 확인
- [ ] TD 발산하는 원인 분석
- [ ] MC가 랜덤과 다를바 없는 이유 분석
- [ ] 맨처음 input 데이터로 512 같은 값 넣는지 확인
- [ ] 애초에 수가 너무 클때 ReLU, loss^2 이런것들이 망하는 이유 분석
