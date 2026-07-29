from typing import List, Optional, Union

import random
import numpy as np
import pygame

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
except ImportError:
    import gym
    from gym.spaces import Box, Discrete

from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP

from src.game.components.hand import Hand
from src.game.components.hand_side import HandSide
from src.game.components.hazards import AttackDirector
from src.game.components.player import Player
from src.game.components.scoreboard import Scoreboard
from src.game.services.visualization_service import VisualizationService
from src.game.utils.config import Config
from src.game.utils.tools import update_background_using_scroll


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # Quattro mani sfalsate: evita il lungo tempo morto dopo la coppia SX/DX.
    INITIAL_HAND_LAYOUT = (
        (HandSide.RIGHT, -80.0),
        (HandSide.LEFT, -270.0),
        (HandSide.RIGHT, -460.0),
        (HandSide.LEFT, -650.0),
    )

    RESPAWN_GAP_RANGE = (165, 225)

    def __init__(self, render_mode):
        self.window_size = (Config.WIDTH, Config.HEIGHT)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(Config.HEIGHT, Config.WIDTH, 3),
            dtype=np.uint8,
        )

        self.action_space = Discrete(5)
        self.actions = {0: -1, 1: K_LEFT, 2: K_RIGHT, 3: K_UP, 4: K_DOWN}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        pygame.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.window.fill((0, 255, 255))

        self.clock = 0
        self.scroll = 0
        self.scoreboard = Scoreboard()
        self.P1 = Player()

        self.hand_list = [Hand(side, y) for side, y in self.INITIAL_HAND_LAYOUT]
        self.hands = pygame.sprite.Group(*self.hand_list)

        # Alias H1/H2 mantenuti per eventuale codice legacy esterno.
        self.H1 = self.hand_list[0]
        self.H2 = self.hand_list[1]

        self.attack_director = AttackDirector()
        self.render_fps = 50

    def _get_info(self):
        return {
            "score": self.scoreboard.get_current_score(),
            "time_passed": self.clock,
            "difficulty_level": self.attack_director.level(self.clock),
        }

    def _reset_variables(self):
        self.scroll = 0
        self.clock = 0
        self.scoreboard.reset_current_score()
        self.P1.reset()

        for hand, (_, y) in zip(self.hand_list, self.INITIAL_HAND_LAYOUT):
            hand.reset(y)

        self.attack_director.reset()

    def _hand_speed_multiplier(self) -> float:
        # Aumenta gradualmente fino a +30% circa dopo 60 secondi.
        return 1.0 + min(0.30, self.clock / 10000.0)

    def _respawn_hand_behind_queue(self, hand: Hand) -> None:
        # new_y piu' piccolo = piu' in alto. La nuova mano viene messa dietro
        # alla piu' alta, con un gap limitato: niente lunghe pause.
        highest_y = min(h.new_y for h in self.hand_list if h is not hand)
        gap = random.randint(*self.RESPAWN_GAP_RANGE)
        hand.respawn(highest_y - gap)

    def _update_components(self, action: int):
        self.P1.update(self.actions[action])

        speed_multiplier = self._hand_speed_multiplier()
        exited_hands = []
        for hand in self.hand_list:
            if hand.move(speed_multiplier=speed_multiplier):
                exited_hands.append(hand)

        # Score = numero di grandi mani effettivamente superate. Non entra nel
        # reward: il reward resta puramente survival-based.
        for hand in exited_hands:
            self.scoreboard.increase_current_score()
            self._respawn_hand_behind_queue(hand)

        self.attack_director.update(self.clock, self.P1.player_position)

        self.scroll = update_background_using_scroll(self.scroll)
        VisualizationService.draw_background_with_scroll(self.window, self.scroll)

    def _draw_components(self):
        # Ordine: hazards -> mani -> player -> scoreboard, cosi' il player resta
        # leggibile anche quando lo schermo e' affollato.
        self.attack_director.draw(self.window)
        for hand in self.hand_list:
            hand.draw(self.window)
        self.P1.draw(self.window)
        self.scoreboard.draw(self.window)

    def _get_observation(self):
        frame = pygame.surfarray.array3d(self.window)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8, copy=False)

    def _collide(self):
        if pygame.sprite.spritecollide(
            self.P1,
            self.hands,
            False,
            pygame.sprite.collide_mask,
        ):
            return True

        return self.attack_director.collides(self.P1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._reset_variables()

        # Disegna lo stato iniziale senza far avanzare artificialmente gli
        # ostacoli di uno step durante il reset.
        VisualizationService.draw_background_with_scroll(self.window, self.scroll)
        self._draw_components()

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        self.clock += 1
        self._update_components(action)

        # Unico reward positivo: sopravvivere.
        reward = 0.001

        self._draw_components()

        terminated = self._collide()
        info = self._get_info()
        observation = self._get_observation()

        if terminated:
            reward -= 2.0

        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        return self._get_observation()

    def close(self):
        pygame.display.quit()
        pygame.quit()
