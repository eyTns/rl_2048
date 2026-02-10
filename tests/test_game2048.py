import numpy as np

from src.game2048 import Game2048


class TestGame2048:
    """Game2048 유닛테스트"""

    def test_reset_creates_board(self):
        """reset()이 4x4 보드를 생성하는지 확인"""
        game = Game2048()
        state = game.reset()

        assert state.shape == (4, 4)
        assert game.score == 0
        assert not game.done

    def test_reset_spawns_two_tiles(self):
        """reset()이 2개의 타일을 생성하는지 확인"""
        game = Game2048()
        game.reset()

        non_zero_count = np.count_nonzero(game.board)
        assert non_zero_count == 2

    def test_spawn_tile_values(self):
        """생성되는 타일이 2 또는 4인지 확인"""
        game = Game2048()
        game.reset()

        non_zero_values = game.board[game.board != 0]
        for val in non_zero_values:
            assert val in [2, 4]

    def test_merge_row_simple(self):
        """단순 합치기 테스트"""
        game = Game2048()
        row = np.array([2, 2, 0, 0], dtype=np.int32)
        result, reward = game._merge_row(row)

        assert list(result) == [4, 0, 0, 0]
        assert reward == 4

    def test_merge_row_multiple(self):
        """여러 개 합치기 테스트"""
        game = Game2048()
        row = np.array([2, 2, 4, 4], dtype=np.int32)
        result, reward = game._merge_row(row)

        assert list(result) == [4, 8, 0, 0]
        assert reward == 12

    def test_merge_row_no_merge(self):
        """합쳐지지 않는 경우"""
        game = Game2048()
        row = np.array([2, 4, 8, 16], dtype=np.int32)
        result, reward = game._merge_row(row)

        assert list(result) == [2, 4, 8, 16]
        assert reward == 0

    def test_merge_row_with_gaps(self):
        """빈칸이 있는 경우"""
        game = Game2048()
        row = np.array([2, 0, 2, 0], dtype=np.int32)
        result, reward = game._merge_row(row)

        assert list(result) == [4, 0, 0, 0]
        assert reward == 4

    def test_merge_row_chain_no_double_merge(self):
        """연쇄 합치기가 일어나지 않는지 확인 (2,2,4 -> 4,4가 되어야 함, 8이 아님)"""
        game = Game2048()
        row = np.array([2, 2, 4, 0], dtype=np.int32)
        result, reward = game._merge_row(row)

        assert list(result) == [4, 4, 0, 0]
        assert reward == 4

    def test_step_returns_correct_format(self):
        """step()이 올바른 형식을 반환하는지 확인"""
        game = Game2048()
        game.reset()

        state, reward, done, info = game.step(Game2048.ACTION_LEFT)

        assert isinstance(state, np.ndarray)
        assert state.shape == (4, 4)
        assert isinstance(reward, (int, float, np.integer))
        assert isinstance(done, bool)
        assert "valid_move" in info

    def test_step_invalid_move(self):
        """유효하지 않은 이동 테스트"""
        game = Game2048()
        game.board = np.array(
            [[2, 4, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )

        _, _, _, info = game.step(Game2048.ACTION_UP)

        # 위로 이동이 유효하지 않아야 함
        assert not info["valid_move"]

    def test_can_move_with_empty_cell(self):
        """빈 칸이 있으면 이동 가능"""
        game = Game2048()
        game.board = np.array(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]], dtype=np.int32
        )

        assert game._can_move()

    def test_can_move_with_adjacent_same(self):
        """인접한 같은 숫자가 있으면 이동 가능"""
        game = Game2048()
        game.board = np.array(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 4]], dtype=np.int32
        )

        assert game._can_move()

    def test_game_over(self):
        """게임 오버 조건 테스트"""
        game = Game2048()
        game.board = np.array(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]], dtype=np.int32
        )

        assert not game._can_move()

    def test_get_valid_actions(self):
        """유효한 행동 목록 테스트"""
        game = Game2048()
        game.board = np.array(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )

        valid = game.get_valid_actions()

        # 상으로는 못 감, 하/좌/우는 가능
        assert Game2048.ACTION_UP not in valid
        assert Game2048.ACTION_DOWN in valid
        assert Game2048.ACTION_RIGHT in valid

    def test_score_accumulates(self):
        """점수가 누적되는지 확인"""
        game = Game2048()
        game.board = np.array(
            [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )
        game.score = 0

        game.step(Game2048.ACTION_LEFT)
        assert game.score == 4

    def test_move_left(self):
        """왼쪽 이동 테스트"""
        game = Game2048()
        game.board = np.array(
            [[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )

        game._move(Game2048.ACTION_LEFT)
        assert game.board[0, 0] == 4
        assert game.board[0, 1] == 0

    def test_move_right(self):
        """오른쪽 이동 테스트"""
        game = Game2048()
        game.board = np.array(
            [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )

        game._move(Game2048.ACTION_RIGHT)
        assert game.board[0, 3] == 4
        assert game.board[0, 2] == 0

    def test_move_up(self):
        """위쪽 이동 테스트"""
        game = Game2048()
        game.board = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int32
        )

        game._move(Game2048.ACTION_UP)
        assert game.board[0, 0] == 4
        assert game.board[1, 0] == 0

    def test_move_down(self):
        """아래쪽 이동 테스트"""
        game = Game2048()
        game.board = np.array(
            [[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        )

        game._move(Game2048.ACTION_DOWN)
        assert game.board[3, 0] == 4
        assert game.board[2, 0] == 0
