from src.game.utils.config import Config
from src.game.services.visualization_service import VisualizationService
from src.game.utils.tools import sine


class Scoreboard:
    def __init__(self):
        self._current_score = 0

    def reset_current_score(self):
        self._current_score = 0

    def increase_current_score(self):
        self._current_score += 1

    def get_current_score(self):
        return self._current_score

    def draw(self, screen):
        y = sine(200.0, 1280, 10.0, 40)
        show_score = VisualizationService.get_main_font().render(str(self._current_score), True, (0, 0, 0))
        score_rect = show_score.get_rect(center=(Config.WIDTH // 2, y + 30))
        screen.blit(VisualizationService.get_score_backing(), (113, y))
        screen.blit(show_score, score_rect)
