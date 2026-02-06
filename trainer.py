import numpy as np
from typing import Callable
from dataclasses import dataclass

from game2048 import Game2048
from model import QNetwork

TARGET_CLIP_RANGE = 100


@dataclass
class Step:
    """한 스텝의 경험"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class StepInfo:
    """스텝 콜백 정보"""
    step_num: int
    state: np.ndarray
    action: int
    reward: float
    loss: float | None
    q_values: np.ndarray


@dataclass
class EpisodeResult:
    """에피소드 결과"""
    episode_num: int
    steps: int
    score: float
    max_tile: int
    losses: list[float]
    epsilon: float


@dataclass
class TrainConfig:
    """학습 설정"""
    method: str = 'td'  # 'td' 또는 'mc'
    gamma: float = 0.999999  # TD용 할인율
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    hidden_size: int = 128


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
                done=done
            )
            episode.append(step)

            # TD: 매 스텝 학습
            loss = self._on_step(step, episode)
            if loss is not None:
                losses.append(loss)

            # 콜백 호출
            if self.on_step_callback:
                self.on_step_callback(StepInfo(
                    step_num=step_num,
                    state=state,
                    action=action,
                    reward=reward,
                    loss=loss,
                    q_values=self.model.forward(state),
                ))

            state = next_state
            step_num += 1
            self.total_steps += 1

        # MC: 에피소드 끝에 학습
        episode_losses = self._on_episode_end(episode)
        if episode_losses:
            losses.extend(episode_losses)

        # epsilon 감소
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        self.episode_count += 1

        # 콜백 호출
        if self.on_episode_end_callback:
            self.on_episode_end_callback(EpisodeResult(
                episode_num=self.episode_count,
                steps=step_num,
                score=self.env.score,
                max_tile=int(self.env.board.max()),
                losses=losses,
                epsilon=self.epsilon,
            ))

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
                print(f"Episode {ep + 1}: avg_score={avg_score:.1f}, "
                      f"max_tile={max_tile}, avg_loss={avg_loss:.4f}, "
                      f"epsilon={self.epsilon:.3f}")

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """스텝마다 호출 (구현 필요)"""
        raise NotImplementedError

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 호출 (구현 필요)"""
        raise NotImplementedError


class TDTrainer(BaseTrainer):
    """TD (Temporal Difference) 학습기"""

    def __init__(self, config: TrainConfig):
        config.method = 'td'
        super().__init__(config)
        self.gamma = config.gamma

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """매 스텝 TD 학습"""
        # 보상 정규화 (log 스케일)
        reward = np.log1p(step.reward) if step.reward > 0 else step.reward

        # 타겟 계산: r + γ * max Q(s', a')
        if step.done:
            target = reward
        else:
            next_q_values = self.model.forward(step.next_state)
            # NaN 체크
            if np.any(np.isnan(next_q_values)):
                next_q_max = 0.0
            else:
                next_q_max = np.max(next_q_values)
            target = reward + self.gamma * next_q_max

        # 타겟 클리핑
        target = np.clip(target, -TARGET_CLIP_RANGE, TARGET_CLIP_RANGE)

        # 순전파 (캐시 갱신) 후 역전파
        self.model.forward(step.state)
        loss = self.model.backward(step.action, target, self.lr)

        return loss

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """TD는 에피소드 끝에 추가 학습 없음"""
        return []


class MCTrainer(BaseTrainer):
    """Monte Carlo 학습기"""

    def __init__(self, config: TrainConfig):
        config.method = 'mc'
        super().__init__(config)
        self.normalize_returns = True  # return 정규화 여부

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """MC는 스텝에서 학습하지 않음"""
        return None

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 전체 학습"""
        if not episode:
            return []

        # 역순으로 return 계산 (감마=1, 할인 없음)
        returns = []
        G = 0.0
        for step in reversed(episode):
            G = step.reward + G
            returns.insert(0, G)

        # 정규화: log 스케일
        if self.normalize_returns:
            returns = [np.log1p(r) if r > 0 else r for r in returns]

        # 각 스텝 학습
        losses = []
        for step, target in zip(episode, returns):
            # 순전파
            self.model.forward(step.state)
            # 역전파
            loss = self.model.backward(step.action, target, self.lr)
            losses.append(loss)

        return losses


def create_trainer(config: TrainConfig) -> BaseTrainer:
    """설정에 따라 적절한 Trainer 생성"""
    if config.method == 'td':
        return TDTrainer(config)
    elif config.method == 'mc':
        return MCTrainer(config)
    else:
        raise ValueError(f"Unknown method: {config.method}")


# 테스트
if __name__ == "__main__":
    print("=" * 50)
    print("TD Trainer 테스트 (gamma=0.999999)")
    print("=" * 50)
    td_config = TrainConfig(method='td', gamma=0.999999)
    td_trainer = create_trainer(td_config)
    td_trainer.train(episodes=100, print_every=20)

    print("\n" + "=" * 50)
    print("MC Trainer 테스트")
    print("=" * 50)
    mc_config = TrainConfig(method='mc')
    mc_trainer = create_trainer(mc_config)
    mc_trainer.train(episodes=100, print_every=20)
