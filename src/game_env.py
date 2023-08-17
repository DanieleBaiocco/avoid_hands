import time
from typing import Optional, Union, List

import gym
import numpy as np
import pygame
from gym.core import RenderFrame, ActType
from gym.spaces import Discrete, Box
from pygame.locals import *
from pygame.time import Clock
from PIL import Image
from src.components.hand import Hand
from src.components.hand_side import HandSide
from src.components.player import Player
from src.components.scoreboard import Scoreboard
from src.config import Config
from src.services.visualization_service import VisualizationService
from src.utils.tools import update_background_using_scroll


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = (Config.WIDTH, Config.HEIGHT)  # The size of the PyGame window
        self.observation_space = Box(0, 255, self.window_size)

        self.action_space = Discrete(5)
        self.actions = {'no_op': -1, 'left': K_LEFT, 'right': K_RIGHT, 'up': K_UP, 'down': K_DOWN}
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

    def _get_info(self):
        return {"score": self.scoreboard.get_current_score(), "time_passed": self.clock}

    def _reset_variables(self):
        self.scroll = 0
        self.clock = 0
        self.scoreboard.reset_current_score()
        self.P1.reset()
        self.H1.reset()
        self.H2.reset()

    def _update_components(self, action):
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
        return np.asarray(img, dtype="int32")

    def _collide(self):
        terminated = False
        if pygame.sprite.spritecollide(self.P1, self.hands, False, pygame.sprite.collide_mask):
            terminated = True
        return terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_variables()
        self._update_components('no_op')
        self._draw_components()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: str):
        self.clock += 1
        reward = self._update_components(action)
        self._draw_components()
        terminated = self._collide()
        info = self._get_info()
        observation = self._get_observation()
        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass
