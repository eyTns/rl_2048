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

# SARSA 학습 (gamma=0.9999)
config = TrainConfig(method='td', gamma=0.9999)
trainer = create_trainer(config)
trainer.train(episodes=1000, print_every=500)

# Monte Carlo 학습
config = TrainConfig(method='mc')
trainer = create_trainer(config)
trainer.train(episodes=1000, print_every=500)
```

### 설정 옵션 (TrainConfig)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `method` | `'td'` | 학습 방식: `'td'` (SARSA) 또는 `'mc'` |
| `gamma` | `0.9999` | 할인율 (TD, MC 공통) |
| `learning_rate` | `0.001` | 학습률 |
| `epsilon_start` | `1.0` | 초기 탐험률 |
| `epsilon_end` | `0.05` | 최소 탐험률 |
| `epsilon_decay` | `0.99` | 탐험률 감소율 |
| `hidden_size` | `128` | 은닉층 크기 |

### TD (SARSA) vs Monte Carlo

| 방식 | 학습 시점 | 타겟 계산 | 특징 |
|------|----------|----------|------|
| TD (SARSA) | 매 스텝 | `r/100 + γ × Q(s', a')` | 빠른 학습, 실제 다음 행동 사용 |
| MC | 에피소드 끝 | `G = r/100 + γ × G` (할인 누적) | γ 할인 적용, 분산 큼 |

**공통 적용 기법:**
- **보상 스케일링**: score / 100 (Q값 × 100 ≈ 예상 실제 점수)
- **Huber Loss** (δ=1.0): MSE 대신 사용하여 큰 오차에 둔감하게 처리 (발산 방지)
- **그래디언트 노름 클리핑** (1.0): 방향을 보존하면서 크기만 제한
- **D4 대칭 증강**: 각 보드를 8개 대칭 변환 (4회전 × 2대칭)으로 학습 데이터 8배 증강
- **Bootstrap explosion 방지**: D4 증강 시 8개 타겟을 먼저 계산(freeze)한 후 학습

### 시각화 콜백

```python
from trainer import StepInfo, EpisodeResult

def on_step(info: StepInfo):
    """매 스텝 호출"""
    print(f"Step {info.step_num}: action={info.action}, reward={info.reward}")

def on_episode_end(result: EpisodeResult):
    """에피소드 끝 호출"""
    print(f"Episode {result.episode_num}: score={result.score}, max={result.max_tile}")

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

### 커리큘럼 모드

```python
from game2048 import Game2048

# 커리큘럼 모드: 하단줄에 정렬된 랜덤 타일, 우상단에 2
env = Game2048(curriculum_mode=True)
state = env.reset()

# 개별 판만 오버라이드
state = env.reset(curriculum_mode=False)  # 이번 판만 일반 모드
```

### 구현 구조

```
┌──────────────────────────────────────────────┐
│   Game2048 (env) — 커리큘럼 모드 지원        │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│          QNetwork (NumPy 기반)               │
│  - 입력: 4x4 보드 → 원핫 인코딩 (256차원)    │
│  - 은닉층: ReLU × 2 (128)                    │
│  - 출력: 4개 Q값                             │
│  - 손실: Huber Loss (δ=1.0)                  │
│  - 클리핑: 그래디언트 노름 (1.0)              │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│            BaseTrainer                       │
│  - train_one_episode()                       │
│  - D4 대칭 증강 (8배)                        │
│  - 보상 스케일링 (score / 100)               │
│  - 콜백 지원                                 │
└──────────────────────────────────────────────┘
         ↙                    ↘
┌──────────────────┐    ┌──────────────────┐
│  TDTrainer       │    │  MCTrainer       │
│  (SARSA)         │    │  에피소드 끝     │
│  스텝마다 학습   │    │  gamma 할인 적용 │
│  gamma 적용      │    │  D4 증강 적용    │
│  D4 증강 적용    │    │                  │
└──────────────────┘    └──────────────────┘
```
