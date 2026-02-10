"""D4 대칭 증강(augmentation) 유닛 테스트

핵심 불변식: augment(move(board, A)) == move(augment(board), action_map[A])
"""

import numpy as np

from src.game2048 import Game2048
from src.trainer import BOARD_AUGMENTATIONS, _augment_board

# 비대칭 보드 (모든 변환에서 구별 가능)
ASYM_BOARD = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]], dtype=np.int32)

# 이동 테스트용 보드
MOVE_BOARD = np.array(
    [[2, 4, 0, 0],
     [0, 2, 0, 0],
     [0, 0, 8, 0],
     [0, 0, 0, 16]], dtype=np.int32)


def _move_board(board, action):
    game = Game2048()
    game.board = board.copy()
    game._move(action)
    return game.board.copy()


def test_rotation_matches_np_rot90():
    for k in range(4):
        result = _augment_board(ASYM_BOARD, rot_k=k, flip=False)
        np.testing.assert_array_equal(result, np.rot90(ASYM_BOARD, k))


def test_rotation_with_flip():
    for k in range(4):
        result = _augment_board(ASYM_BOARD, rot_k=k, flip=True)
        np.testing.assert_array_equal(result, np.fliplr(np.rot90(ASYM_BOARD, k)))


def test_four_rotations_return_to_original():
    board = ASYM_BOARD.copy()
    for _ in range(4):
        board = _augment_board(board, rot_k=1, flip=False)
    np.testing.assert_array_equal(board, ASYM_BOARD)


def test_all_eight_transforms_unique():
    boards = set()
    for rot_k, flip, _ in BOARD_AUGMENTATIONS:
        boards.add(_augment_board(ASYM_BOARD, rot_k, flip).tobytes())
    assert len(boards) == 8


def test_action_maps_are_permutations():
    for _, _, action_map in BOARD_AUGMENTATIONS:
        assert sorted(action_map) == [0, 1, 2, 3]


def test_identity_map():
    rot_k, flip, action_map = BOARD_AUGMENTATIONS[0]
    assert (rot_k, flip, action_map) == (0, False, [0, 1, 2, 3])


def test_180_swaps():
    """180°: 상↔하, 좌↔우"""
    _, _, m = BOARD_AUGMENTATIONS[2]
    assert m == [1, 0, 3, 2]


def test_flip_swaps_left_right():
    """좌우 대칭: 상하 유지, 좌↔우"""
    _, _, m = BOARD_AUGMENTATIONS[4]
    assert m == [0, 1, 3, 2]


def test_invariant_rotations():
    """핵심: augment(move(board,A)) == move(augment(board), map[A]) — 회전 4종"""
    for aug_idx in range(4):  # rot only
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[aug_idx]
        for action in range(4):
            path1 = _augment_board(_move_board(MOVE_BOARD, action), rot_k, flip)
            path2 = _move_board(_augment_board(MOVE_BOARD, rot_k, flip), action_map[action])
            np.testing.assert_array_equal(path1, path2,
                err_msg=f"rot={rot_k}, action={action}")


def test_invariant_flips():
    """핵심: augment(move(board,A)) == move(augment(board), map[A]) — 대칭 4종"""
    for aug_idx in range(4, 8):  # flip variants
        rot_k, flip, action_map = BOARD_AUGMENTATIONS[aug_idx]
        for action in range(4):
            path1 = _augment_board(_move_board(MOVE_BOARD, action), rot_k, flip)
            path2 = _move_board(_augment_board(MOVE_BOARD, rot_k, flip), action_map[action])
            np.testing.assert_array_equal(path1, path2,
                err_msg=f"rot={rot_k}, flip={flip}, action={action}")


def test_d4_closure():
    """두 변환의 합성도 D4에 포함"""
    all_t = {_augment_board(ASYM_BOARD, r, f).tobytes() for r, f, _ in BOARD_AUGMENTATIONS}
    for i in range(8):
        for j in range(8):
            r1, f1, _ = BOARD_AUGMENTATIONS[i]
            r2, f2, _ = BOARD_AUGMENTATIONS[j]
            composed = _augment_board(_augment_board(ASYM_BOARD, r1, f1), r2, f2)
            assert composed.tobytes() in all_t


def test_d4_inverse():
    """각 변환에 역변환 존재"""
    orig = ASYM_BOARD.tobytes()
    for i in range(8):
        r1, f1, _ = BOARD_AUGMENTATIONS[i]
        found = any(
            _augment_board(_augment_board(ASYM_BOARD, r1, f1), r2, f2).tobytes() == orig
            for r2, f2, _ in BOARD_AUGMENTATIONS
        )
        assert found, f"aug[{i}]의 역변환 없음"
