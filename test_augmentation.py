"""D4 대칭 증강(augmentation) 유닛 테스트

핵심 불변식(invariant):
  원본 보드에 action A를 적용한 결과를 augment 한 것 ==
  augment된 보드에 action_map[A]를 적용한 결과

즉, augment(move(board, A)) == move(augment(board), action_map[A])
이 불변식이 깨지면 augmented 데이터로 학습 시 잘못된 action을 강화하게 됨.
"""

import numpy as np
import pytest

from game2048 import Game2048
from trainer import BOARD_AUGMENTATIONS, _augment_board


# ── 테스트용 비대칭 보드 (모든 변환에서 구별 가능) ──────────────────

ASYM_BOARD = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]],
    dtype=np.int32,
)


class TestAugmentBoardRotation:
    """_augment_board 회전이 np.rot90과 정확히 일치하는지 검증"""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_rotation_matches_np_rot90(self, k):
        expected = np.rot90(ASYM_BOARD, k)
        result = _augment_board(ASYM_BOARD, rot_k=k, flip=False)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_rotation_with_flip_matches_np(self, k):
        expected = np.fliplr(np.rot90(ASYM_BOARD, k))
        result = _augment_board(ASYM_BOARD, rot_k=k, flip=True)
        np.testing.assert_array_equal(result, expected)

    def test_identity_transform(self):
        result = _augment_board(ASYM_BOARD, rot_k=0, flip=False)
        np.testing.assert_array_equal(result, ASYM_BOARD)

    def test_four_rotations_return_to_original(self):
        """4번 90° 회전하면 원본으로 복귀"""
        board = ASYM_BOARD.copy()
        for _ in range(4):
            board = _augment_board(board, rot_k=1, flip=False)
        np.testing.assert_array_equal(board, ASYM_BOARD)

    def test_double_flip_returns_to_original(self):
        """좌우 대칭을 2번 적용하면 원본으로 복귀"""
        board = _augment_board(ASYM_BOARD, rot_k=0, flip=True)
        board = _augment_board(board, rot_k=0, flip=True)
        np.testing.assert_array_equal(board, ASYM_BOARD)


class TestAllEightTransformsUnique:
    """D4 대칭군의 8개 변환이 모두 서로 다른 보드를 생성하는지 검증"""

    def test_all_augmentations_produce_distinct_boards(self):
        boards = []
        for rot_k, flip, _ in BOARD_AUGMENTATIONS:
            b = _augment_board(ASYM_BOARD, rot_k, flip)
            boards.append(b.tobytes())
        assert len(set(boards)) == 8, "8개 변환이 모두 구별되어야 함"


class TestActionMapInvariant:
    """핵심 불변식: augment(move(board, A)) == move(augment(board), action_map[A])

    이 테스트가 실패하면 augmentation이 잘못된 action을 강화한다는 의미.
    """

    # 이동 가능하도록 설계된 테스트 보드들
    TEST_BOARDS = [
        # 보드 1: 다양한 이동이 가능한 비대칭 보드
        np.array(
            [[2, 4, 0, 0],
             [0, 2, 0, 0],
             [0, 0, 8, 0],
             [0, 0, 0, 16]],
            dtype=np.int32,
        ),
        # 보드 2: 합칠 수 있는 타일이 있는 보드
        np.array(
            [[2, 2, 4, 4],
             [8, 0, 0, 8],
             [0, 16, 16, 0],
             [32, 0, 0, 32]],
            dtype=np.int32,
        ),
        # 보드 3: 한 방향으로만 채워진 보드
        np.array(
            [[2, 4, 8, 16],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    ]

    @staticmethod
    def _move_board(board: np.ndarray, action: int) -> np.ndarray:
        """타일 생성 없이 보드 이동만 수행 (Game2048._move 로직 재현)"""
        game = Game2048()
        game.board = board.copy()
        game._move(action)
        return game.board.copy()

    @pytest.mark.parametrize("board_idx", [0, 1, 2])
    @pytest.mark.parametrize("aug_idx", range(8))
    @pytest.mark.parametrize("action", [0, 1, 2, 3])
    def test_invariant(self, board_idx, aug_idx, action):
        """augment(move(board, A)) == move(augment(board), action_map[A])"""
        board = self.TEST_BOARDS[board_idx]
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[aug_idx]
        mapped_action = action_map[action]

        # 경로 1: 원본에서 이동 → 결과를 augment
        moved_original = self._move_board(board, action)
        path1 = _augment_board(moved_original, rot_k, flip)

        # 경로 2: 원본을 augment → augmented 보드에서 mapped action으로 이동
        augmented_board = _augment_board(board, rot_k, flip)
        path2 = self._move_board(augmented_board, mapped_action)

        np.testing.assert_array_equal(
            path1, path2,
            err_msg=(
                f"불변식 위반! board={board_idx}, aug={aug_idx} "
                f"(rot={rot_k}, flip={flip}), action={action} -> mapped={mapped_action}\n"
                f"경로1 (move→aug):\n{path1}\n"
                f"경로2 (aug→move):\n{path2}"
            ),
        )


class TestActionMapConsistency:
    """action_map이 유효한 순열인지, 논리적으로 올바른지 검증"""

    def test_all_action_maps_are_permutations(self):
        """각 action_map이 [0,1,2,3]의 순열인지 확인"""
        for rot_k, flip, action_map in BOARD_AUGMENTATIONS:
            assert sorted(action_map) == [0, 1, 2, 3], (
                f"action_map {action_map} (rot={rot_k}, flip={flip})은 "
                f"[0,1,2,3]의 순열이 아님"
            )

    def test_identity_has_identity_map(self):
        """원본 변환(rot=0, flip=False)은 항등 매핑이어야 함"""
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[0]
        assert rot_k == 0 and not flip
        assert action_map == [0, 1, 2, 3]

    def test_180_rotation_swaps_up_down_and_left_right(self):
        """180° 회전: 상↔하, 좌↔우"""
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[2]
        assert rot_k == 2 and not flip
        # 상(0)→하(1), 하(1)→상(0), 좌(2)→우(3), 우(3)→좌(2)
        assert action_map[0] == 1  # 상→하
        assert action_map[1] == 0  # 하→상
        assert action_map[2] == 3  # 좌→우
        assert action_map[3] == 2  # 우→좌

    def test_flip_only_swaps_left_right(self):
        """좌우 대칭(rot=0, flip=True): 상하 유지, 좌↔우"""
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[4]
        assert rot_k == 0 and flip
        assert action_map[0] == 0  # 상→상
        assert action_map[1] == 1  # 하→하
        assert action_map[2] == 3  # 좌→우
        assert action_map[3] == 2  # 우→좌


class TestAugmentationGroupProperties:
    """D4 대칭군의 수학적 성질 검증"""

    @staticmethod
    def _compose(board, aug1_idx, aug2_idx):
        """두 augmentation을 순서대로 적용"""
        r1, f1, _ = BOARD_AUGMENTATIONS[aug1_idx]
        r2, f2, _ = BOARD_AUGMENTATIONS[aug2_idx]
        b = _augment_board(board, r1, f1)
        b = _augment_board(b, r2, f2)
        return b

    def test_closure(self):
        """D4 군은 닫혀있음: 두 변환의 합성도 D4에 포함되어야 함"""
        all_transforms = set()
        for rot_k, flip, _ in BOARD_AUGMENTATIONS:
            b = _augment_board(ASYM_BOARD, rot_k, flip)
            all_transforms.add(b.tobytes())

        for i in range(8):
            for j in range(8):
                composed = self._compose(ASYM_BOARD, i, j)
                assert composed.tobytes() in all_transforms, (
                    f"aug[{i}] ∘ aug[{j}]의 결과가 D4에 포함되지 않음"
                )

    def test_each_has_inverse(self):
        """각 변환에 역변환이 존재함"""
        original_bytes = ASYM_BOARD.tobytes()
        for i in range(8):
            found_inverse = False
            for j in range(8):
                composed = self._compose(ASYM_BOARD, i, j)
                if composed.tobytes() == original_bytes:
                    found_inverse = True
                    break
            assert found_inverse, f"aug[{i}]의 역변환이 D4 내에 없음"


class TestSpecificRotationValues:
    """구체적인 회전 결과 값 검증 (하드코딩된 기대값)"""

    def test_rot90_counterclockwise(self):
        """90° 반시계 회전 결과 확인"""
        board = np.array(
            [[1, 2],
             [3, 4]], dtype=np.int32
        )
        expected = np.array(
            [[2, 4],
             [1, 3]], dtype=np.int32
        )
        result = _augment_board(board, rot_k=1, flip=False)
        np.testing.assert_array_equal(result, expected)

    def test_rot180(self):
        """180° 회전 결과 확인"""
        board = np.array(
            [[1, 2],
             [3, 4]], dtype=np.int32
        )
        expected = np.array(
            [[4, 3],
             [2, 1]], dtype=np.int32
        )
        result = _augment_board(board, rot_k=2, flip=False)
        np.testing.assert_array_equal(result, expected)

    def test_rot270_counterclockwise(self):
        """270° 반시계 (= 90° 시계) 회전 결과 확인"""
        board = np.array(
            [[1, 2],
             [3, 4]], dtype=np.int32
        )
        expected = np.array(
            [[3, 1],
             [4, 2]], dtype=np.int32
        )
        result = _augment_board(board, rot_k=3, flip=False)
        np.testing.assert_array_equal(result, expected)

    def test_fliplr(self):
        """좌우 대칭 결과 확인"""
        board = np.array(
            [[1, 2],
             [3, 4]], dtype=np.int32
        )
        expected = np.array(
            [[2, 1],
             [4, 3]], dtype=np.int32
        )
        result = _augment_board(board, rot_k=0, flip=True)
        np.testing.assert_array_equal(result, expected)

    def test_rot90_then_flip(self):
        """90° 반시계 + 좌우 대칭"""
        board = np.array(
            [[1, 2],
             [3, 4]], dtype=np.int32
        )
        # rot90 CCW: [[2,4],[1,3]] → fliplr: [[4,2],[3,1]]
        expected = np.array(
            [[4, 2],
             [3, 1]], dtype=np.int32
        )
        result = _augment_board(board, rot_k=1, flip=True)
        np.testing.assert_array_equal(result, expected)
