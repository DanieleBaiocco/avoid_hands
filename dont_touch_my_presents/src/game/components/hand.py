import math
import random

import pygame

from src.game.components.hand_side import HandSide
from src.game.utils.config import Config
from src.game.services.visualization_service import VisualizationService


class Hand(pygame.sprite.Sprite):
    """Grande ostacolo laterale con movimento verticale + pattern orizzontale.

    Questa versione e' pensata per un flusso continuo di 4 mani. La gestione
    del respawn e dello score e' demandata a GameEnv, che riposiziona ogni mano
    dietro alla coda degli altri ostacoli.
    """

    VERTICAL_SPEED = 4.0
    PATTERN_MIN_STEPS = 60
    PATTERN_MAX_STEPS = 160

    LEFT_BASE_RANGE = (-80, -55)
    RIGHT_BASE_RANGE = (315, 340)

    PATTERNS = {
        "slow_wave": {"amplitude": 55.0, "frequency": 0.060, "shape": "sine"},
        "fast_wave": {"amplitude": 50.0, "frequency": 0.160, "shape": "sine"},
        "wide_wave": {"amplitude": 85.0, "frequency": 0.075, "shape": "sine"},
        "sweep": {"amplitude": 75.0, "frequency": 0.070, "shape": "triangle"},
    }

    def __init__(self, hand_side: HandSide, initial_y: float = -80.0):
        super().__init__()
        self.side = hand_side
        self.initial_y = float(initial_y)
        self.new_y = self.initial_y
        self.offset_x = 0
        self.new_x = 0
        self._phase = 0.0
        self._pattern = "slow_wave"
        self._pattern_steps_remaining = 0
        self._load_hand()
        self.reset(self.initial_y)

    def _safe_base_x(self) -> int:
        if self.side == HandSide.RIGHT:
            return random.randint(*self.RIGHT_BASE_RANGE)
        return random.randint(*self.LEFT_BASE_RANGE)

    def _choose_pattern(self) -> None:
        self._pattern = random.choice(tuple(self.PATTERNS.keys()))
        self._pattern_steps_remaining = random.randint(
            self.PATTERN_MIN_STEPS,
            self.PATTERN_MAX_STEPS,
        )

    def reset(self, y: float | None = None) -> None:
        self._phase = random.uniform(0.0, math.tau)
        self.offset_x = self._safe_base_x()
        self.new_y = self.initial_y if y is None else float(y)
        self.new_x = self.offset_x
        self._choose_pattern()
        self.rect.center = (int(self.new_x), int(self.new_y))

    def respawn(self, y: float) -> None:
        """Riposiziona la mano sopra lo schermo con pattern/base nuovi."""
        self.reset(y)

    def _load_hand(self) -> None:
        if self.side == HandSide.RIGHT:
            self.image = VisualizationService.get_right_hand_image()
        else:
            self.image = VisualizationService.get_left_hand_image()

        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    @staticmethod
    def _triangle_wave(phase: float) -> float:
        return (2.0 / math.pi) * math.asin(math.sin(phase))

    def _horizontal_offset(self) -> float:
        pattern = self.PATTERNS[self._pattern]
        if pattern["shape"] == "triangle":
            wave = self._triangle_wave(self._phase)
        else:
            wave = math.sin(self._phase)
        return wave * pattern["amplitude"]

    def move(self, speed_multiplier: float = 1.0) -> bool:
        """Avanza la mano. Restituisce True quando e' completamente uscita."""
        pattern = self.PATTERNS[self._pattern]
        self._phase += pattern["frequency"]
        self._pattern_steps_remaining -= 1

        if self._pattern_steps_remaining <= 0:
            self._choose_pattern()

        self.new_x = int(self.offset_x + self._horizontal_offset())
        self.new_y += self.VERTICAL_SPEED * float(speed_multiplier)
        self.rect.center = (int(self.new_x), int(self.new_y))

        return self.rect.top > Config.HEIGHT

    def draw(self, screen):
        dotted_line = VisualizationService.get_dotted_line()
        screen.blit(dotted_line, (0, self.rect.y + 53))

        # Disegna la mano normalmente
        screen.blit(self.image, self.rect)

        # Bordo scuro della silhouette, solo visivo.
        outline_mask = self.mask.to_surface(
            setcolor=(20, 20, 20, 255),
            unsetcolor=(0, 0, 0, 0),
        )

        # Disegna la silhouette leggermente traslata attorno alla mano
        # per creare un contorno di circa 2 px.
        for dx, dy in [
            (-2, 0), (2, 0),
            (0, -2), (0, 2),
            (-1, -1), (1, -1),
            (-1, 1), (1, 1),
        ]:
            screen.blit(
                outline_mask,
                (self.rect.x + dx, self.rect.y + dy),
            )

        # Ridisegna la mano sopra il bordo
        screen.blit(self.image, self.rect)
