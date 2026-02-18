<!-- AI 수정 가능 -->

# DQN 2048 실험 가이드

성공적인 DQN 구현체(GitHub: IsacPasianotto, SergioIommi, Lok Hin Chan)와 논문 데이터를 종합한 실험 세팅.
목표: 1024~2048 타일 도달, 점수 10,000~20,000.

---

## 1. 현재 구현 vs 권장 설정

| 항목 | 현재 (ml_app.html) | 권장 | 비고 |
|------|-------------------|------|------|
| γ (gamma) | 0.999 | 0.99 | 0.9999에서 변경됨. 0.99가 더 안정적일 수 있음 |
| α (learning rate) | 0.0001 | 5e-5 ~ 1e-4 | 권장 범위 내로 조정 완료 |
| ε 시작 | 0.05 | 0.9 ~ 1.0 | 초기에 충분한 탐험 필요 |
| ε 최소 | 0.0001 | 0.01 | 너무 낮으면 exploitation에 갇힘 |
| ε 감쇠 | 0.99 (에피소드당) | 0.9999 (에피소드당) | 현재: 100 에피소드면 ε≈0.37. 권장: 10,000 에피소드면 ε≈0.37 |
| Replay Buffer | 없음 | 10,000 ~ 50,000 | 샘플 간 상관관계를 끊어 학습 안정화 |
| Target Network | 에피소드 단위 동기화 | 1,000 ~ 2,000 step마다 복사 | 구현 완료. step 단위 동기화는 미적용 |
| 리워드 함수 | log₂(reward) | log₂(merged_value) | 적용 완료 |
| Invalid Move | -10 | -0.1 또는 -10 | 적용 완료 |
| 투채널 인코딩 | 구현됨 (2채널) | 구현됨 (2채널) | 적용 완료 |
| D4 대칭 증강 | 8배 증강 활성 | 8배 증강 | 적용 완료 |

---

## 2. 리워드 함수

### 방법 A: 로그 방식 (안정적, 권장)

머지 발생 시 log₂(new_tile_value), 그 외 0.

```
4+4 → 8 → 보상 = log₂(8) = 3
128+128 → 256 → 보상 = log₂(256) = 8
```

값 범위가 1~17로 일정하여 학습이 안정적.

### 방법 B: Score 방식 (직관적)

실제 게임 점수를 그대로 사용.
단, 정규화 또는 Gradient Clipping 필수.

```
4+4 → 8 → 보상 = 8
128+128 → 256 → 보상 = 256
```

### 공통: Invalid Move 페널티

유효하지 않은 이동에 -0.1 페널티를 주어 벽에 부딪히는 행동을 빠르게 교정.

---

## 3. 핵심 수식

### Q-Learning Update (Target Network 사용)

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q_target(s', a') - Q(s, a) ]
```

- s, a: 현재 상태(보드)와 행동(상/하/좌/우)
- r: 리워드
- s': 다음 상태
- Q_target: 타겟 네트워크 (C 스텝마다 메인 네트워크에서 복사)

---

## 4. 실험 계획: 단계별 적용

HTML 파일에서 직접 실험. 한 번에 하나씩 변경하여 효과 측정.

### 실험 1: 하이퍼파라미터 조정 — 적용 완료

gamma: 0.999, learningRate: 0.0001로 변경됨. UI에서 추가 조정 가능.

미적용 항목 (UI에서 조정하여 실험):
```
ε 시작: 0.05 → 0.9
ε 감쇠: 0.99 → 0.9999
ε 최소: 0.0001 → 0.01
```

### 실험 2: 리워드 함수 변경 — 적용 완료

```javascript
// 현재 적용된 코드
function scaleReward(reward) {
    return reward > 0 ? Math.log2(reward) : -1;
}
```

머지 없는 이동은 -1 반환.

### 실험 3: D4 대칭 증강 — 적용 완료

TD, MC, TS 전 트레이너에서 8배 증강 활성화됨.
Invalid action에도 증강 8개 state 전부에 -10 학습 적용.

### 실험 4: Replay Buffer 구현

```javascript
class ReplayBuffer {
    constructor(maxSize = 10000) {
        this.buffer = [];
        this.maxSize = maxSize;
    }
    push(state, action, reward, nextState, done) {
        if (this.buffer.length >= this.maxSize) {
            this.buffer.shift();
        }
        this.buffer.push({ state, action, reward, nextState, done });
    }
    sample(batchSize = 32) {
        const indices = [];
        for (let i = 0; i < batchSize; i++) {
            indices.push(Math.floor(Math.random() * this.buffer.length));
        }
        return indices.map(i => this.buffer[i]);
    }
    get size() { return this.buffer.length; }
}
```

### 실험 5: Target Network — 적용 완료

에피소드 단위로 target network를 main network에서 동기화.
TD, TS 트레이너에서 Q(s') 계산 시 target network 사용.

```javascript
// 에피소드 끝에 동기화
this.model.copyWeightsTo(this.targetModel);

// 학습 시 target network로 Q(s') 계산
const qNext = this.targetModel.forward(augNext);
```

미적용: step 단위 동기화 (1,000~2,000 step마다). 에피소드 단위가 충분한지 실험 필요.

---

## 5. 고급 기법 (한계 돌파용)

순수 DQN의 한계(4096+ 타일 꾸준히 생성)를 넘기 위한 추가 기법:

- 분포형 강화학습 (Distributional RL)
- 우선순위 경험 재생 (PER): 중요한 경험을 더 자주 샘플링
- CNN을 깊게 쌓기: 4x4x16 입력에 Conv2D 적용
- Expectimax Search 혼합: 탐색 알고리즘과 DQN 결합

---

## 6. 측정 기준

각 실험의 성과를 비교하기 위한 지표:

- 100 에피소드 평균 점수
- 100 에피소드 중 최고 타일 분포 (512 도달률, 1024 도달률, 2048 도달률)
- 학습 loss 추이
- Q값 발산 여부

---

## 7. JS 연산 성능 최적화

| 항목 | 내용 | 상태 |
|------|------|------|
| A. scaleReward 룩업 테이블 | 연산 비중 미미 | 불채택 |
| B. augmentBoard 캐싱 | 복잡도 대비 효과 적음 | 불채택 |
| C. 중복 forward() 제거 | getAction이 qValues도 반환하여 호출부 중복 forward 삭제 | 적용 완료 |
| D. 연산 버퍼 재사용 | 생성자에서 버퍼 1회 할당, in-place 연산, transpose 제거 | 적용 완료 |
| E. renderBoard DOM 재사용 | initGrid로 16개 div 1회 생성, 이후 속성만 변경 | 적용 완료 |

미적용 후보: Flat 1D 배열, WebAssembly, Web Worker, WebGPU
