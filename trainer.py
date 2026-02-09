from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from game2048 import Game2048
from model import QNetwork

# D4 대칭군: 4회전 × 2대칭 = 8변환
# (rot_k, flip, action_map) — action: 0=상, 1=하, 2=좌, 3=우
BOARD_AUGMENTATIONS = [
    (0, False, [0, 1, 2, 3]),  # 원본
    (1, False, [2, 3, 1, 0]),  # 90° 반시계
    (2, False, [1, 0, 3, 2]),  # 180°
    (3, False, [3, 2, 0, 1]),  # 270° 반시계
    (0, True,  [0, 1, 3, 2]),  # 좌우 대칭
    (1, True,  [3, 2, 1, 0]),  # 90° + 좌우 대칭
    (2, True,  [1, 0, 2, 3]),  # 180° + 좌우 대칭
    (3, True,  [2, 3, 0, 1]),  # 270° + 좌우 대칭
]


def _augment_board(state: np.ndarray, rot_k: int, flip: bool) -> np.ndarray:
    """보드 변환: 회전 + 대칭"""
    s = np.rot90(state, rot_k)
    if flip:
        s = np.fliplr(s)
    return s

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
    """학습 설정"""

    method: Literal["td", "mc"] = "td"
    gamma: float = 0.9999  # TD용 할인율
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    hidden_size: int = 256


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

            # get_action 내부 forward 결과 재사용 (탐험 시에는 캐시 없을 수 있음)
            cached_q = self.model._cache.get("z3")
            q_values = cached_q[0] if cached_q is not None else self.model.forward(state)

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

            # 콜백 호출 (forward 재호출 없이 캐시된 q_values 사용)
            if self.on_step_callback:
                self.on_step_callback(
                    StepInfo(
                        step_num=step_num,
                        state=state,
                        action=action,
                        reward=reward,
                        loss=loss,
                        q_values=q_values,
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

    def train(self, episodes: int, print_every: int = 500):
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
                    f"탐험률={self.epsilon:.3f}"
                )

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """스텝마다 호출 (구현 필요)"""
        raise NotImplementedError

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 호출 (구현 필요)"""
        raise NotImplementedError


class TDTrainer(BaseTrainer):
    """SARSA 학습기: target = ln(r) + γ * Q(s', a')"""

    def __init__(self, config: TrainConfig):
        config.method = "td"
        super().__init__(config)
        self.gamma = config.gamma

    def _scale_reward(self, reward: float) -> float:
        """보상 스케일링: score / 100"""
        return reward / 100.0

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """SARSA: 1스텝 지연 학습"""
        if len(episode) < 2:
            return None

        prev_step = episode[-2]
        curr_step = episode[-1]
        r = self._scale_reward(prev_step.reward)

        # D4 대칭 8배 증강 비활성화 — 학습 성능 저하 의심
        # train_items = []
        # for rot_k, flip, action_map in BOARD_AUGMENTATIONS:
        #     aug_curr = _augment_board(curr_step.state, rot_k, flip)
        #     aug_curr_action = action_map[curr_step.action]
        #     q_next = float(self.model.forward(aug_curr)[aug_curr_action])
        #     target = r + self.gamma * q_next
        #
        #     aug_prev = _augment_board(prev_step.state, rot_k, flip)
        #     aug_prev_action = action_map[prev_step.action]
        #     train_items.append((aug_prev, aug_prev_action, target))

        # 원본만 사용
        q_next = float(self.model.forward(curr_step.state)[curr_step.action])
        target = r + self.gamma * q_next

        self.model.forward(prev_step.state)
        total_loss = self.model.backward(prev_step.action, target, self.lr)

        return total_loss

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """마지막 스텝 학습 (다음 행동 없음)"""
        if not episode:
            return []
        last_step = episode[-1]
        r = self._scale_reward(last_step.reward)

        # D4 대칭 8배 증강 비활성화 — 학습 성능 저하 의심
        # train_items = []
        # for rot_k, flip, action_map in BOARD_AUGMENTATIONS:
        #     aug_state = _augment_board(last_step.state, rot_k, flip)
        #     aug_action = action_map[last_step.action]
        #     train_items.append((aug_state, aug_action, r))
        #
        # total_loss = 0.0
        # for aug_state, aug_action, target in train_items:
        #     self.model.forward(aug_state)
        #     total_loss += self.model.backward(aug_action, target, self.lr)

        # 원본만 사용
        self.model.forward(last_step.state)
        total_loss = self.model.backward(last_step.action, r, self.lr)

        return [total_loss]


class MCTrainer(BaseTrainer):
    """Monte Carlo 학습기"""

    def __init__(self, config: TrainConfig):
        config.method = "mc"
        super().__init__(config)
        self.gamma = config.gamma

    def _scale_reward(self, reward: float) -> float:
        """보상 스케일링: score / 100"""
        return reward / 100.0

    def _on_step(self, step: Step, episode: list[Step]) -> float | None:
        """MC는 스텝에서 학습하지 않음"""
        return None

    def _on_episode_end(self, episode: list[Step]) -> list[float]:
        """에피소드 끝에 전체 학습"""
        if not episode:
            return []

        # 역순으로 할인 return 계산: G_t = r_t/512 + γ * G_{t+1}
        returns = []
        G = 0.0
        for step in reversed(episode):
            G = self._scale_reward(step.reward) + self.gamma * G
            returns.append(G)
        returns.reverse()

        # 각 스텝 학습
        losses = []
        for step, target in zip(episode, returns):
            # D4 대칭 8배 증강 비활성화 — 학습 성능 저하 의심
            # train_items = []
            # for rot_k, flip, action_map in BOARD_AUGMENTATIONS:
            #     aug_state = _augment_board(step.state, rot_k, flip)
            #     aug_action = action_map[step.action]
            #     train_items.append((aug_state, aug_action, target))
            # total_loss = 0.0
            # for aug_state, aug_action, t in train_items:
            #     self.model.forward(aug_state)
            #     total_loss += self.model.backward(aug_action, t, self.lr)
            # losses.append(total_loss / 8)

            # 원본만 사용
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
    print("SARSA Trainer 테스트 (gamma=0.9999)")
    print("=" * 50)
    td_config = TrainConfig(method="td", gamma=0.9999)
    td_trainer = create_trainer(td_config)
    td_trainer.train(episodes=100, print_every=20)

    print("\n" + "=" * 50)
    print("MC Trainer 테스트")
    print("=" * 50)
    mc_config = TrainConfig(method="mc")
    mc_trainer = create_trainer(mc_config)
    mc_trainer.train(episodes=100, print_every=20)
