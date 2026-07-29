from src.game.utils.config import Config
from src.game.services.visualization_service import VisualizationService
import pygame

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
        font = pygame.font.Font(None, 24)

        score_text = font.render(
            str(self._current_score),
            True,
            (120, 120, 120),  # grigio, meno contrastato
        )

        screen.blit(score_text, (10, 10))
