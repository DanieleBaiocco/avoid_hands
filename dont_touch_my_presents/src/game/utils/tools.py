import math

import pygame
from pygame.locals import *


def sine(speed: float, time: int, how_far: float, overall_y: int) -> int:
    t = pygame.time.get_ticks() / 2 % time
    y = math.sin(t / speed) * how_far + overall_y
    return int(y)


def update_background_using_scroll(scroll):
    scroll -= .5

    if scroll < -80:
        scroll = 0

    return scroll

