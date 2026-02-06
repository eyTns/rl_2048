# 사용법

## 설치

```bash
# 기본 설치 (RL 학습용)
uv sync

# UI 포함 설치
uv sync  # pygame, fastapi, uvicorn 포함
```

## 실행

### 컴퓨터 UI (pygame)

```bash
uv run python game_ui.py
```

- 조작: 화살표 키 또는 WASD
- 재시작: R
- 종료: ESC

### 모바일/웹 UI (FastAPI 서버)

```bash
uv run python web_ui.py
```

- 접속: http://localhost:8000
- 조작: 터치 스와이프 또는 키보드
- 단일 유저 전용 (ML 연동용)

### 정적 웹 UI (서버 불필요)

브라우저에서 `static_ui.html` 파일 직접 열기

- GitHub Pages 배포 가능
- 조작: 터치 스와이프 또는 키보드

## API 엔드포인트 (web_ui.py)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 게임 UI 페이지 |
| GET | `/state` | 현재 게임 상태 조회 |
| POST | `/move` | 이동 (body: `{"action": 0-3}`) |
| POST | `/reset` | 게임 리셋 |

action 값:
- 0: 위
- 1: 아래
- 2: 왼쪽
- 3: 오른쪽

---

## 학습

### 기본 사용

```python
from trainer import TrainConfig, create_trainer

# TD 학습 (gamma=0.999999)
config = TrainConfig(method='td', gamma=0.999999)
trainer = create_trainer(config)
trainer.train(episodes=1000, print_every=100)

# Monte Carlo 학습
config = TrainConfig(method='mc')
trainer = create_trainer(config)
trainer.train(episodes=1000, print_every=100)
```

### 설정 옵션 (TrainConfig)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `method` | `'td'` | 학습 방식: `'td'` 또는 `'mc'` |
| `gamma` | `0.999999` | 할인율 (TD 전용) |
| `learning_rate` | `0.001` | 학습률 |
| `epsilon_start` | `1.0` | 초기 탐험률 |
| `epsilon_end` | `0.01` | 최소 탐험률 |
| `epsilon_decay` | `0.995` | 탐험률 감소율 |
| `hidden_size` | `128` | 은닉층 크기 |

### TD vs Monte Carlo

| 방식 | 학습 시점 | 타겟 계산 | 특징 |
|------|----------|----------|------|
| TD | 매 스텝 | `r + γ × max Q(s')` | 빠른 학습, γ 감쇠 문제 |
| MC | 에피소드 끝 | 실제 누적 보상 | γ 없음, 분산 큼 |

### 시각화 콜백

```python
def on_step(step_num, state, action, reward, loss, q_values):
    """매 스텝 호출"""
    print(f"Step {step_num}: action={action}, reward={reward}")

def on_episode_end(episode_num, steps, score, max_tile, losses, epsilon):
    """에피소드 끝 호출"""
    print(f"Episode {episode_num}: score={score}, max={max_tile}")

trainer.on_step_callback = on_step
trainer.on_episode_end_callback = on_episode_end
```

### 모델 저장/로드

```python
# 저장
trainer.model.save("model.npz")

# 로드
trainer.model.load("model.npz")
```

### 구현 구조

```
┌─────────────────────────────────────────┐
│              Game2048 (env)             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         QNetwork (NumPy 기반)           │
│  - 입력: 4x4 보드 (log2 정규화)         │
│  - 은닉층: ReLU × 2                     │
│  - 출력: 4개 Q값                        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│            BaseTrainer                  │
│  - train_one_episode()                  │
│  - 콜백 지원                            │
└─────────────────────────────────────────┘
         ↙                    ↘
┌─────────────────┐    ┌─────────────────┐
│   TDTrainer     │    │   MCTrainer     │
│ 스텝마다 학습   │    │ 에피소드 끝     │
│ gamma 적용      │    │ 할인 없음       │
└─────────────────┘    └─────────────────┘
```
