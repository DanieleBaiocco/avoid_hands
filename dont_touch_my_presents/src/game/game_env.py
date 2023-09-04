from typing import Optional, Union, List

import gym
import numpy as np
import pygame
from PIL import Image
from gym.core import RenderFrame
from gym.spaces import Discrete, Box
from pygame.locals import *

from src.game.components.hand import Hand
from src.game.components.hand_side import HandSide
from src.game.components.player import Player
from src.game.components.scoreboard import Scoreboard
from src.game.utils.config import Config
from src.game.services.visualization_service import VisualizationService
from src.game.utils.tools import update_background_using_scroll


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode):
        self.window_size = (Config.WIDTH, Config.HEIGHT)  # The size of the PyGame window
        self.observation_space = Box(0, 255, self.window_size)

        self.action_space = Discrete(5)
        self.actions = {0: -1, 1: K_LEFT, 2: K_RIGHT, 3: K_UP, 4: K_DOWN}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        screen = pygame.display.set_mode(self.window_size)
        screen.fill((0, 255, 255))
        pygame.init()
        self.window = screen

        self.clock = 0
        self.scroll = None
        self.scoreboard = Scoreboard()
        self.P1 = Player()
        self.H1 = Hand(HandSide.RIGHT)
        self.H2 = Hand(HandSide.LEFT)
        self.hands = pygame.sprite.Group()
        self.hands.add(self.H1)
        self.hands.add(self.H2)
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.P1)
        self.all_sprites.add(self.H1)
        self.all_sprites.add(self.H2)
        self.render_fps = 50
    def _get_info(self):
        return {"score": self.scoreboard.get_current_score(), "time_passed": self.clock}

    def _reset_variables(self):
        self.scroll = 0
        self.clock = 0
        self.scoreboard.reset_current_score()
        self.P1.reset()
        self.H1.reset()
        self.H2.reset()

    def _update_components(self, action: int):
        self.P1.update(self.actions[action])
        rw1 = self.H1.move(self.scoreboard, self.P1.player_position)
        rw2 = self.H2.move(self.scoreboard, self.P1.player_position)
        self.scroll = update_background_using_scroll(self.scroll)
        VisualizationService.draw_background_with_scroll(self.window, self.scroll)
        return rw1 + rw2

    def _draw_components(self):
        self.P1.draw(self.window)
        self.H1.draw(self.window)
        self.H2.draw(self.window)
        self.scoreboard.draw(self.window)

    def _get_observation(self):
        result = np.array(pygame.image.tostring(self.window, "RGB"))
        img = Image.frombytes("RGB", self.window_size, result)
        return np.asarray(img, dtype="uint8")

    def _collide(self):
        terminated = False
        if pygame.sprite.spritecollide(self.P1, self.hands, False, pygame.sprite.collide_mask):
            terminated = True
        return terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_variables()
        self._update_components(0)
        self._draw_components()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        self.clock += 1
        reward = self._update_components(action)
        self._draw_components()
        terminated = self._collide()
        info = self._get_info()
        observation = self._get_observation()
        if terminated:
            return observation, -1, True, False, info
        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._get_observation()
