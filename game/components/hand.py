import random

import pygame

from src.game.components.hand_side import HandSide
from src.game.components.scoreboard import Scoreboard
from src.game.utils.config import Config
from src.game.services.visualization_service import VisualizationService
from src.game.utils.tools import sine


class Hand(pygame.sprite.Sprite):
    def __init__(self, hand_side: HandSide):
        super().__init__()
        self.new_spd = random.uniform(2.5, 3)
        self.new_y = 0
        self.offset_x = 0
        self.new_x = sine(100.0, 1280, 20.0, self.offset_x)
        self.side = hand_side
        self.can_score = True

        self._load_hand()

    def reset(self):
        self.new_spd = random.uniform(0.5, 8)
        self.can_score = True

        if self.side == HandSide.RIGHT:
            self.offset_x = random.randint(260, 380)
            self.new_y = -40
            self.new_x = 0

        if self.side == HandSide.LEFT:
            self.offset_x = random.randint(-50, 120)
            self.new_y = -320
            self.new_x = 0

    def _load_hand(self):
        if self.side == HandSide.RIGHT:
            self._load_right_hand()

        if self.side == HandSide.LEFT:
            self._load_left_hand()

    def _load_left_hand(self):
        self.image = VisualizationService.get_left_hand_image()
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.offset_x = random.randint(-50, 120)
        self.new_y = -320

    def _load_right_hand(self):
        self.image = VisualizationService.get_right_hand_image()
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.offset_x = random.randint(260, 380)
        self.new_y = -40

    def move(self, scoreboard: Scoreboard, player_position) -> int:
        reward = 0
        self.new_x = sine(100.0, 620, 20.0, self.offset_x)
        self.new_y += self.new_spd
        self.rect.center = (self.new_x, self.new_y)

        if self.rect.top > player_position.y - 35 and self.can_score:
            reward = 1
            scoreboard.increase_current_score()
            self.can_score = False


        if self.rect.top > Config.HEIGHT:
            self.rect.bottom = 0
            # Play Kung Fu Sound

            if self.side == HandSide.RIGHT:
                self.new_spd = random.uniform(0.5, 2)
                self.offset_x = random.randint(260, 380)
                self.new_y = -40

            if self.side == HandSide.LEFT:
                self.new_spd = random.uniform(4, 8)
                self.offset_x = random.randint(-50, 120)
                self.new_y = -320

            if self.new_spd >= 6:
                self.new_spd = 8

            self.can_score = True
        return reward

    def draw(self, screen):
        dotted_line = VisualizationService.get_dotted_line()
        screen.blit(dotted_line, (0, self.rect.y + 53))
        screen.blit(self.image, self.rect)
