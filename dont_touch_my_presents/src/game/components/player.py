import pygame
from pygame.locals import *

from src.game.utils.config import Config

vec = pygame.math.Vector2


class Player(pygame.sprite.Sprite):
    """Piccolo player quadrato, pensato per un gameplay più simile a Undertale.

    La dinamica (accelerazione, attrito e azioni) resta la stessa del gioco
    originale; cambia solo la dimensione/visualizzazione del personaggio e la
    gestione dei bordi, così il player non rimane con velocità accumulata contro
    una parete.
    """

    SIZE = 18
    COLOR = (220, 35, 55)

    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((self.SIZE, self.SIZE), pygame.SRCALPHA)
        self.image.fill(self.COLOR)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.pos = vec((180, 550))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.player_position = vec(0, 0)

    def update(self, action):
        self.acc = vec(0, 0)

        if action == K_LEFT:
            self.acc.x = -Config.ACC
        if action == K_RIGHT:
            self.acc.x = +Config.ACC
        if action == K_UP:
            self.acc.y = -Config.ACC
        if action == K_DOWN:
            self.acc.y = +Config.ACC

        self.acc.x += self.vel.x * Config.FRIC
        self.acc.y += self.vel.y * Config.FRIC
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        half = self.SIZE / 2

        if self.pos.x > Config.WIDTH - half:
            self.pos.x = Config.WIDTH - half
            self.vel.x = min(0, self.vel.x)
        if self.pos.x < half:
            self.pos.x = half
            self.vel.x = max(0, self.vel.x)
        if self.pos.y > Config.HEIGHT - half:
            self.pos.y = Config.HEIGHT - half
            self.vel.y = min(0, self.vel.y)
        if self.pos.y < 200 + half:
            self.pos.y = 200 + half
            self.vel.y = max(0, self.vel.y)

        self.player_position = self.pos.copy()
        self.rect.center = self.pos

    def draw(self, screen):
        # Niente più grande mano di Babbo Natale dietro al player: resta solo
        # il piccolo quadrato, più leggibile anche per la rete neurale.
        screen.blit(self.image, self.rect)

    def reset(self):
        self.pos = vec((180, 550))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.player_position = self.pos.copy()
        self.rect.center = self.pos
