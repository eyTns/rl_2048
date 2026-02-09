import random

import numpy as np

FOUR_SPAWN_RATE = 0.1


class Game2048:
    """2048 게임 환경 (RL 학습용)"""

    # 행동 정의
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    # 방향에 따른 회전 횟수 (왼쪽으로 밀기 기준)
    # UP: 시계 90도 → 왼쪽 밀기 → 반시계 90도
    # DOWN: 반시계 90도 → 왼쪽 밀기 → 시계 90도
    # LEFT: 회전 없음
    # RIGHT: 180도 → 왼쪽 밀기 → 180도
    ROTATION_MAP = {
        ACTION_UP: 1,
        ACTION_DOWN: 3,
        ACTION_LEFT: 0,
        ACTION_RIGHT: 2,
    }

    def __init__(self):
        self.board = None
        self.score = 0
        self.done = False
        self.reset()

    def reset(self):
        """게임 초기화, 초기 상태 반환"""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def get_state(self):
        """현재 보드 상태 반환 (4x4 numpy array)"""
        return self.board.copy()

    def step(self, action):
        """
        행동 수행

        Args:
            action: 0=상, 1=하, 2=좌, 3=우

        Returns:
            state: 새로운 상태
            reward: 보상 (합쳐진 타일 값의 합)
            done: 게임 종료 여부
            info: 추가 정보
        """
        if self.done:
            return self.get_state(), 0, True, {"valid_move": False}

        # 이동 전 보드 저장
        old_board = self.board.copy()

        # 이동 수행
        reward = self._move(action)

        # 유효한 이동인지 확인
        valid_move = not np.array_equal(old_board, self.board)

        if valid_move:
            self._spawn_tile()

            # 게임 종료 확인
            if not self._can_move():
                self.done = True

        self.score += reward

        return self.get_state(), reward, self.done, {"valid_move": valid_move}

    def _spawn_tile(self):
        """빈 칸에 2(90%) 또는 4(10%) 생성"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 4 if random.random() < FOUR_SPAWN_RATE else 2

    def _move(self, action):
        """보드 이동 및 합치기, 획득 점수 반환"""
        reward = 0
        rotation_count = self.ROTATION_MAP[action]

        rotated = np.rot90(self.board, rotation_count)

        for i in range(4):
            row = rotated[i]
            new_row, row_reward = self._merge_row(row)
            rotated[i] = new_row
            reward += row_reward

        # 원래 방향으로 복원
        self.board = np.rot90(rotated, -rotation_count)

        return reward

    def _merge_row(self, row):
        """한 줄을 왼쪽으로 밀고 합치기"""
        # 0이 아닌 값만 추출
        non_zero = row[row != 0]

        if len(non_zero) == 0:
            return np.zeros(4, dtype=np.int32), 0

        # 합치기
        merged = []
        reward = 0
        i = 0

        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                reward += merged_value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # 나머지는 0으로 채우기
        result = np.zeros(4, dtype=np.int32)
        result[: len(merged)] = merged

        return result, reward

    @staticmethod
    def _can_move_row_left(row) -> bool:
        """한 줄이 왼쪽으로 이동/합치기 가능한지 조건만으로 판단"""
        for i in range(3):
            if row[i] == 0 and row[i + 1] != 0:
                return True
            if row[i] != 0 and row[i] == row[i + 1]:
                return True
        return False

    def _can_move_direction(self, action: int) -> bool:
        """특정 방향으로 이동 가능한지 확인 (보드 복사/이동 없이)"""
        rotated = np.rot90(self.board, self.ROTATION_MAP[action])
        for i in range(4):
            if self._can_move_row_left(rotated[i]):
                return True
        return False

    def _can_move(self):
        """이동 가능 여부 확인"""
        for action in range(4):
            if self._can_move_direction(action):
                return True
        return False

    def get_valid_actions(self):
        """유효한 행동 목록 반환 (조건 비교만, 보드 복사/이동 없음)"""
        valid = []
        for action in range(4):
            if self._can_move_direction(action):
                valid.append(action)
        return valid

    def render(self):
        """현재 보드 상태 출력"""
        print(f"\nScore: {self.score}")
        print("-" * 25)
        for row in self.board:
            print("|", end="")
            for val in row:
                if val == 0:
                    print("     |", end="")
                else:
                    print(f"{val:^5}|", end="")
            print()
            print("-" * 25)
        print()


# 테스트
if __name__ == "__main__":
    game = Game2048()
    game.render()

    # 랜덤 플레이 테스트
    actions = ["상", "하", "좌", "우"]

    for _ in range(10):
        action = random.randint(0, 3)
        state, reward, done, info = game.step(action)
        print(f"행동: {actions[action]}, 보상: {reward}, 유효: {info['valid_move']}")
        game.render()

        if done:
            print("게임 종료!")
            break
