import pygame
import sys
from game2048 import Game2048

# 색상 정의
COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    "big": (60, 58, 50),
}

TEXT_COLORS = {
    0: (205, 193, 180),
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
}

BACKGROUND_COLOR = (187, 173, 160)

# 화면 설정
TILE_SIZE = 100
TILE_MARGIN = 10
GRID_SIZE = 4
HEADER_HEIGHT = 100

WINDOW_WIDTH = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * TILE_MARGIN
WINDOW_HEIGHT = HEADER_HEIGHT + GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * TILE_MARGIN


class Game2048UI:
    """2048 게임 UI (pygame 기반)"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2048")

        self.font_large = pygame.font.Font(None, 55)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 30)

        self.game = Game2048()
        self.clock = pygame.time.Clock()

    def get_tile_color(self, value):
        """타일 값에 따른 배경색 반환"""
        return COLORS.get(value, COLORS["big"])

    def get_text_color(self, value):
        """타일 값에 따른 텍스트 색상 반환"""
        if value in TEXT_COLORS:
            return TEXT_COLORS[value]
        return (249, 246, 242)

    def draw_tile(self, row, col, value):
        """타일 그리기"""
        x = TILE_MARGIN + col * (TILE_SIZE + TILE_MARGIN)
        y = HEADER_HEIGHT + TILE_MARGIN + row * (TILE_SIZE + TILE_MARGIN)

        # 타일 배경
        color = self.get_tile_color(value)
        pygame.draw.rect(self.screen, color, (x, y, TILE_SIZE, TILE_SIZE), border_radius=5)

        # 타일 숫자
        if value != 0:
            text_color = self.get_text_color(value)
            if value >= 1000:
                font = self.font_medium
            else:
                font = self.font_large

            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
            self.screen.blit(text, text_rect)

    def draw_header(self):
        """헤더 (점수, 게임 상태) 그리기"""
        # 타이틀
        title = self.font_large.render("2048", True, (119, 110, 101))
        self.screen.blit(title, (TILE_MARGIN, 20))

        # 점수
        score_label = self.font_small.render("SCORE", True, (238, 228, 218))
        score_value = self.font_medium.render(str(self.game.score), True, (255, 255, 255))

        score_box_x = WINDOW_WIDTH - 120 - TILE_MARGIN
        pygame.draw.rect(self.screen, (187, 173, 160),
                        (score_box_x, 15, 120, 60), border_radius=5)

        self.screen.blit(score_label, (score_box_x + 35, 20))
        score_rect = score_value.get_rect(center=(score_box_x + 60, 55))
        self.screen.blit(score_value, score_rect)

    def draw_game_over(self):
        """게임 오버 화면 그리기"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))

        game_over_text = self.font_large.render("Game Over!", True, (119, 110, 101))
        text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))
        self.screen.blit(game_over_text, text_rect)

        restart_text = self.font_small.render("Press R to restart", True, (119, 110, 101))
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
        self.screen.blit(restart_text, restart_rect)

    def draw(self):
        """전체 화면 그리기"""
        self.screen.fill((250, 248, 239))

        # 게임 보드 배경
        board_y = HEADER_HEIGHT
        board_height = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * TILE_MARGIN
        pygame.draw.rect(self.screen, BACKGROUND_COLOR,
                        (0, board_y, WINDOW_WIDTH, board_height), border_radius=5)

        # 헤더
        self.draw_header()

        # 타일 그리기
        board = self.game.get_state()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self.draw_tile(row, col, board[row, col])

        # 게임 오버
        if self.game.done:
            self.draw_game_over()

        pygame.display.flip()

    def handle_input(self, key):
        """키 입력 처리"""
        if self.game.done:
            if key == pygame.K_r:
                self.game.reset()
            return

        action = None
        if key == pygame.K_UP or key == pygame.K_w:
            action = Game2048.ACTION_UP
        elif key == pygame.K_DOWN or key == pygame.K_s:
            action = Game2048.ACTION_DOWN
        elif key == pygame.K_LEFT or key == pygame.K_a:
            action = Game2048.ACTION_LEFT
        elif key == pygame.K_RIGHT or key == pygame.K_d:
            action = Game2048.ACTION_RIGHT

        if action is not None:
            self.game.step(action)

    def run(self):
        """게임 메인 루프"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self.handle_input(event.key)

            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    ui = Game2048UI()
    ui.run()
