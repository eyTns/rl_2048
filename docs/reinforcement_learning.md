# Reinforcement Learning Study Notes

## 기본 개념
- **State (상태)**: 현재 환경의 상황 (2048에서는 4x4 보드의 16개 숫자)
- **Action (행동)**: 에이전트가 취할 수 있는 선택 (2048에서는 상/하/좌/우)
- **Reward (보상)**: 행동에 대한 피드백, 설계가 중요함 (단순 점수보다 빈칸 수, 합쳐짐 등 고려)
- **Policy (정책)**: "이 상태에서 어떤 행동을 할까"에 대한 매핑표, RL의 목표는 최적의 Policy를 찾는 것

## 알고리즘 관계
- **RL (Reinforcement Learning)**: 강화학습 전체를 아우르는 큰 분야
- **Q-Learning**: RL의 한 알고리즘, Q-table에 (상태, 행동)별 가치를 저장
- **Deep Q-Learning (DQN)**: Q-table 대신 신경망 사용, 상태가 많을 때 필수 (2048은 상태가 무수히 많으므로 DQN 필요)

## 알아볼 내용
- [ ] 학습이 어떤 계산식으로 이루어지는지 (Q-Learning 수식)
- [x] Q-Learning과 Deep Q-Learning과 RL의 차이 → 위에 정리됨
- [ ] Policy가 구체적으로 어떻게 구현되는지
- [ ] DQN은 구체적으로 어떤 방식으로 동작하는지
- [ ] Policy와 Reward의 관계 (포함관계인지?)

