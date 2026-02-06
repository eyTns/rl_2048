"""game_ui.py 테스트"""

from game_ui import COLORS, TEXT_COLORS, BoardRenderer


class TestColors:
    """색상 매핑 테스트"""

    def test_tile_color_zero(self):
        """빈 타일 색상"""
        renderer = BoardRenderer.__new__(BoardRenderer)
        assert renderer.get_tile_color(0) == COLORS[0]

    def test_tile_color_small_values(self):
        """2048 이하 타일 색상"""
        renderer = BoardRenderer.__new__(BoardRenderer)
        for value in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            assert renderer.get_tile_color(value) == COLORS[value]

    def test_tile_color_big_values(self):
        """2048 초과 타일 색상"""
        renderer = BoardRenderer.__new__(BoardRenderer)
        for value in [4096, 8192, 16384, 32768]:
            assert renderer.get_tile_color(value) == COLORS["big"]

    def test_text_color_small_values(self):
        """2048 이하 텍스트 색상"""
        renderer = BoardRenderer.__new__(BoardRenderer)
        for value in [2, 4]:
            assert renderer.get_text_color(value) == (119, 110, 101)  # 어두운 색
        for value in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            assert renderer.get_text_color(value) == (249, 246, 242)  # 밝은 색

    def test_text_color_big_values(self):
        """2048 초과 텍스트 색상"""
        renderer = BoardRenderer.__new__(BoardRenderer)
        for value in [4096, 8192, 16384]:
            assert renderer.get_text_color(value) == (249, 246, 242)
