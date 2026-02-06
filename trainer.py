from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from game2048 import Game2048
from model import QNetwork

class Step(BaseModel):
    """한 스텝의 경험"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class StepInfo(BaseModel):
    """스텝 콜백 정보"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_num: int
    state: np.ndarray
    action: int
    reward: float
    loss: float | None
    q_values: np.ndarray


class EpisodeResult(BaseModel):
    """에피소드 결과"""

    episode_num: int
    steps: int
    score: float
    max_tile: int
    losses: list[float]
    epsilon: float


class TrainConfig(BaseModel):
    """학습 설정 (모든 파라미터 범위 검증 포함)"""

    method: Literal["td", "mc"] = "td"
    gamma: float = Field(default=0.99, gt=0.0, le=1.0, description="할인율 (0 < γ ≤ 1)")
    learning_rate: float = Field(
        default=0.001, gt=0.0, le=1.0, description="학습률 (0 < lr ≤ 1)"
    )
    epsilon_start: float = Field(
        default=1.0, ge=0.0, le=1.0, description="초기 탐험율 (0 ≤ ε_start ≤ 1)"
    )
    epsilon_end: float = Field(
        default=0.01, ge=0.0, le=1.0, description="최소 탐험율 (0 ≤ ε_end ≤ 1)"
    )
    epsilon_decay: float = Field(
        default=0.995, gt=0.0, le=1.0, description="탐험율 감소 계수 (0 < decay ≤ 1)"
    )
    hidden_size: int = Field(default=128, ge=8, le=1024, description="은닉층 크기 (8–1024)")
    target_update_freq: int = Field(
        default=500, ge=1, description="타겟 네트워크 업데이트 주기 (스텝 단위, TD 전용)"
    )
    reward_log_scale: bool = Field(
        default=True, description="보상 로그 스케일링 (큰 보상 압축으로 안정적 수렴)"
    )

    @model_validator(mode="after")
    def validate_epsilon_ordering(self) -> "TrainConfig":
        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                f"epsilon_end ({self.epsilon_end}) must be <= "
                f"epsilon_start ({self.epsilon_start})"
            )
        return self


class BaseTrainer:
    """학습기 기본 클래스"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.env = Game2048()
        self.model = QNetwork(hidden_size=config.hidden_size)
        self.lr = config.learning_rate
        self.epsilon = config.epsilon_start

        # 통계
        self.episode_count = 0
        self.total_steps = 0

        # 콜백
        self.on_step_callback: Callable[[StepInfo], None] | None = None
        self.on_episode_end_callback: Callable[[EpisodeResult], None] | None = None

    def train_one_episode(self) -> tuple[int, float, list[float]]:
        """
        한 판 학습

        Returns:
            (총 스텝 수, 최종 점수, 손실 리스트)
        """
        episode: list[Step] = []
        losses: list[float] = []

        state = self.env.reset()
        step_num = 0

        while not self.env.done:
            # 행동 선택
            valid_actions = self.env.get_valid_actions()
            action = self.model.get_action(state, valid_actions, self.epsilon)

            # 환경 스텝
            next_state, reward, done, info = self.env.step(action)

            # 경험 저장
            step = Step(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
            )
            episode.append(step)

            # TD: 매 스텝 학습
            loss = self._on_step(step, episode)
            if loss is not None:
                losses.append(loss)

            # 콜백 호출
            if self.on_step_callback:
                self.on_step_callback(
                    StepInfo(
                        step_num=step_num,
                        state=state,
                        action=action,
                        reward=reward,
                        loss=loss,
                        q_values=self.model.forward(state),
                    )
                )

            state = next_state
            step_num += 1
            self.total_steps += 1

        # MC: 에피소드 끝에 학습
        episode_losses = self._on_episode_end(episode)
        if episode_losses:
            losses.extend(episode_losses)

        # epsilon 감소
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )
        self.episode_count += 1

        # 콜백 호출
        if self.on_episode_end_callback:
            self.on_episode_end_callback(
                EpisodeResult(
                    episode_num=self.episode_count,
                    steps=step_num,
                    score=self.env.score,
                    max_tile=int(self.env.board.max()),
                    losses=losses,
                    epsilon=self.epsilon,
                )
            )

        return step_num, self.env.score, losses

    def train(self, episodes: int, print_every: int = 100):
        """
        여러 판 학습

        Args:
            episodes: 학습할 에피소드 수
            print_every: 통계 출력 주기
        """
        scores = []
        max_tiles = []

        for ep in range(episodes):
            steps, score, losses = self.train_one_episode()
            scores.append(score)
            max_tiles.append(int(self.env.board.max()))

            if (ep + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_loss = np.mean(losses) if losses else 0
                max_tile = max(max_tiles[-print_every:])
                print(
                    f"Episode {ep + 1}: avg_score={avg_score:.1f}, "
                    f"max_tile={max_tile}, avg_loss={avg_loss:.4f}, "
                    f"epsilon={self.epsilon:.3f}"
                )

    def _scale_reward(self, reward: float) -> float:
        """보상 스케일링: log(1 + |r|) 변환으로 큰 보상을 압축하여 수렴 안정성 향상"""
        if self.config.reward_log_scale:
            return float(np.sign(reward) * np.log1p(np.abs(reward)))
        return reward

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """스텝마다 호출 (구현 필요)"""
        raise NotImplementedError

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 호출 (구현 필요)"""
        raise NotImplementedError


class TDTrainer(BaseTrainer):
    """TD (Temporal Difference) 학습기 — 벨만 방정식 기반

    수식: target = r + γ * max_a' Q_target(s', a')  (비종료)
          target = r                                  (종료)

    타겟 네트워크를 사용하여 "이동 타겟" 문제를 해결하고 안정적 수렴을 보장.
    """

    def __init__(self, config: TrainConfig):
        config.method = "td"
        super().__init__(config)
        self.gamma = config.gamma
        # 타겟 네트워크: 주기적으로 메인 네트워크에서 가중치를 복사
        self.target_model = QNetwork(hidden_size=config.hidden_size)
        self.target_model.copy_weights_from(self.model)
        self._step_counter = 0

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """매 스텝 TD 학습 (벨만 방정식)

        target = scale(r) + γ * max_a' Q_target(s', a')  if not done
        target = scale(r)                                  if done
        """
        scaled_reward = self._scale_reward(step.reward)

        if step.done:
            target = scaled_reward
        else:
            # 타겟 네트워크로 다음 상태의 최대 Q값 계산
            next_q = self.target_model.forward(step.next_state)
            valid_next = self.env.get_valid_actions()
            if valid_next:
                masked_q = np.full(4, -np.inf)
                for a in valid_next:
                    masked_q[a] = next_q[a]
                max_next_q = float(np.max(masked_q))
            else:
                max_next_q = 0.0
            target = scaled_reward + self.gamma * max_next_q

        self.model.forward(step.state)
        loss = self.model.backward(step.action, target, self.lr)

        # 타겟 네트워크 주기적 동기화
        self._step_counter += 1
        if self._step_counter % self.config.target_update_freq == 0:
            self.target_model.copy_weights_from(self.model)

        return loss

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """TD는 매 스텝 학습하므로 에피소드 끝 추가 학습 불필요"""
        return []


class MCTrainer(BaseTrainer):
    """Monte Carlo 학습기

    수식: G_t = scale(r_t) + γ * G_{t+1}  (할인 리턴)

    에피소드 종료 후 실제 누적 리턴으로 각 상태의 Q값을 학습.
    할인율(γ)을 적용하여 초기 상태의 리턴이 폭발하지 않도록 방지.
    """

    def __init__(self, config: TrainConfig):
        config.method = "mc"
        super().__init__(config)
        self.gamma = config.gamma

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """MC는 스텝에서 학습하지 않음"""
        return None

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 할인 리턴으로 전체 학습"""
        if not episode:
            return []

        # 역순으로 할인 return 계산: G_t = scale(r_t) + γ * G_{t+1}
        returns = []
        G = 0.0
        for step in reversed(episode):
            G = self._scale_reward(step.reward) + self.gamma * G
            returns.insert(0, G)

        # 각 스텝 학습
        losses = []
        for step, target in zip(episode, returns):
            self.model.forward(step.state)
            loss = self.model.backward(step.action, target, self.lr)
            losses.append(loss)

        return losses


def create_trainer(config: TrainConfig) -> BaseTrainer:
    """설정에 따라 적절한 Trainer 생성"""
    if config.method == "td":
        return TDTrainer(config)
    elif config.method == "mc":
        return MCTrainer(config)
    else:
        raise ValueError(f"Unknown method: {config.method}")


# 테스트
if __name__ == "__main__":
    print("=" * 50)
    print("TD Trainer 테스트 (벨만 방정식, gamma=0.99)")
    print("=" * 50)
    td_config = TrainConfig(method="td", gamma=0.99)
    td_trainer = create_trainer(td_config)
    td_trainer.train(episodes=100, print_every=20)

    print("\n" + "=" * 50)
    print("MC Trainer 테스트 (할인 리턴, gamma=0.99)")
    print("=" * 50)
    mc_config = TrainConfig(method="mc", gamma=0.99)
    mc_trainer = create_trainer(mc_config)
    mc_trainer.train(episodes=100, print_every=20)
