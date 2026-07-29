from __future__ import annotations

import math
import random

import pygame

from src.game.utils.config import Config


PLAYFIELD_TOP = 200


class Bullet(pygame.sprite.Sprite):
    """Piccolo proiettile con breve telegraph prima della caduta."""

    def __init__(
        self,
        x: float,
        vx: float,
        vy: float,
        warning_steps: int = 28,
        radius: int = 13,
    ):
        super().__init__()
        self.radius = int(radius)
        self.vx = float(vx)
        self.vy = float(vy)
        self.warning_steps = int(warning_steps)
        self.active = False
        self.dead = False
        self.x = float(x)
        self.y = float(PLAYFIELD_TOP - 25)

        size = self.radius * 2 + 2
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        # Bordo scuro molto netto: rimane leggibile anche dopo grayscale+resize.
        pygame.draw.circle(
            self.image,
            (25, 25, 25, 255),
            (size // 2, size // 2),
            self.radius,
        )
        pygame.draw.circle(
            self.image,
            (245, 60, 70, 255),
            (size // 2, size // 2),
            max(1, self.radius - 3),
        )
        pygame.draw.circle(
            self.image,
            (255, 235, 135, 255),
            (size // 2, size // 2),
            max(3, self.radius // 3),
        )
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self) -> None:
        if self.dead:
            return

        if self.warning_steps > 0:
            self.warning_steps -= 1
            if self.warning_steps == 0:
                self.active = True
                self.y = float(PLAYFIELD_TOP - self.radius - 2)
            self.rect.center = (int(self.x), int(self.y))
            return

        self.x += self.vx
        self.y += self.vy

        # Rimbalzo leggero sui bordi: evita traiettorie che spariscono subito.
        if self.x < self.radius:
            self.x = float(self.radius)
            self.vx = abs(self.vx)
        elif self.x > Config.WIDTH - self.radius:
            self.x = float(Config.WIDTH - self.radius)
            self.vx = -abs(self.vx)

        self.rect.center = (int(self.x), int(self.y))
        if self.rect.top > Config.HEIGHT:
            self.dead = True

    def collides(self, player: pygame.sprite.Sprite) -> bool:
        return self.active and pygame.sprite.collide_mask(self, player) is not None

    def draw(self, screen: pygame.Surface) -> None:
        if self.dead:
            return

        if self.warning_steps > 0:
            # Freccia/triangolo di preavviso nella posizione di spawn.
            x = int(self.x)
            y = PLAYFIELD_TOP + 8
            pulse = 8 + (self.warning_steps % 10)

            # Warning ad alto contrasto: triangolo scuro con interno chiaro.
            outer = [(x, y + pulse + 3), (x - 14, y - 10), (x + 14, y - 10)]
            inner = [(x, y + pulse - 1), (x - 9, y - 6), (x + 9, y - 6)]
            pygame.draw.polygon(screen, (20, 20, 20), outer)
            pygame.draw.polygon(screen, (255, 210, 40), inner)

            # Linea guida scura e più spessa: sopravvive meglio al downsampling.
            pygame.draw.line(
                screen,
                (25, 25, 25),
                (x, PLAYFIELD_TOP),
                (x, min(Config.HEIGHT, PLAYFIELD_TOP + 85)),
                5,
            )
            pygame.draw.line(
                screen,
                (255, 210, 40),
                (x, PLAYFIELD_TOP),
                (x, min(Config.HEIGHT, PLAYFIELD_TOP + 85)),
                2,
            )
            return

        screen.blit(self.image, self.rect)


class Laser:
    """Laser orizzontale/verticale con warning ben visibile prima dell'attivazione."""

    def __init__(
        self,
        orientation: str,
        coordinate: int,
        warning_steps: int = 55,
        active_steps: int = 34,
        thickness: int = 18,
    ):
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation deve essere 'horizontal' o 'vertical'")

        self.orientation = orientation
        self.coordinate = int(coordinate)
        self.warning_steps = int(warning_steps)
        self.active_steps = int(active_steps)
        # `thickness` resta lo spessore LETale/collisione.
        # Il laser viene disegnato molto più spesso per essere chiaramente
        # visibile dopo grayscale + resize, senza rendere più larga la hitbox.
        self.thickness = int(thickness)
        self.visual_thickness = max(self.thickness, int(round(self.thickness * 2.4)))
        self.active = False
        self.dead = False

    @property
    def rect(self) -> pygame.Rect:
        half = self.thickness // 2
        if self.orientation == "horizontal":
            return pygame.Rect(0, self.coordinate - half, Config.WIDTH, self.thickness)
        return pygame.Rect(
            self.coordinate - half,
            PLAYFIELD_TOP,
            self.thickness,
            Config.HEIGHT - PLAYFIELD_TOP,
        )

    @property
    def visual_rect(self) -> pygame.Rect:
        half = self.visual_thickness // 2
        if self.orientation == "horizontal":
            return pygame.Rect(
                0,
                self.coordinate - half,
                Config.WIDTH,
                self.visual_thickness,
            )
        return pygame.Rect(
            self.coordinate - half,
            PLAYFIELD_TOP,
            self.visual_thickness,
            Config.HEIGHT - PLAYFIELD_TOP,
        )

    def update(self) -> None:
        if self.dead:
            return
        if self.warning_steps > 0:
            self.warning_steps -= 1
            if self.warning_steps == 0:
                self.active = True
            return

        self.active_steps -= 1
        if self.active_steps <= 0:
            self.active = False
            self.dead = True

    def collides(self, player: pygame.sprite.Sprite) -> bool:
        return self.active and self.rect.colliderect(player.rect)

    def draw(self, screen: pygame.Surface) -> None:
        if self.dead:
            return

        collision_rect = self.rect
        visual_rect = self.visual_rect

        if self.active:
            # Fascia esterna molto scura: resta evidente in grayscale.
            pygame.draw.rect(screen, (20, 20, 20), visual_rect)

            # Corpo rosso del laser.
            core = collision_rect.inflate(
                0 if self.orientation == "horizontal" else 8,
                8 if self.orientation == "horizontal" else 0,
            )
            pygame.draw.rect(screen, (210, 25, 40), core)

            # Centro chiaro stretto per dare un bordo/contrasto netto.
            if self.orientation == "horizontal":
                inner = collision_rect.inflate(0, -max(4, self.thickness // 2))
            else:
                inner = collision_rect.inflate(-max(4, self.thickness // 2), 0)
            pygame.draw.rect(screen, (255, 245, 210), inner)

        else:
            # Warning molto contrastato. L'area visuale e' ~2.4x la hitbox,
            # ma NON modifica la collisione.
            overlay = pygame.Surface((Config.WIDTH, Config.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (20, 20, 20, 75), visual_rect)
            screen.blit(overlay, (0, 0))

            dash = 14
            if self.orientation == "horizontal":
                # Base scura continua + tratteggio chiaro sovrapposto.
                pygame.draw.line(
                    screen,
                    (25, 25, 25),
                    (0, self.coordinate),
                    (Config.WIDTH, self.coordinate),
                    7,
                )
                for x in range(0, Config.WIDTH, dash * 2):
                    pygame.draw.line(
                        screen,
                        (255, 215, 55),
                        (x, self.coordinate),
                        (min(x + dash, Config.WIDTH), self.coordinate),
                        3,
                    )
            else:
                pygame.draw.line(
                    screen,
                    (25, 25, 25),
                    (self.coordinate, PLAYFIELD_TOP),
                    (self.coordinate, Config.HEIGHT),
                    7,
                )
                for y in range(PLAYFIELD_TOP, Config.HEIGHT, dash * 2):
                    pygame.draw.line(
                        screen,
                        (255, 215, 55),
                        (self.coordinate, y),
                        (self.coordinate, min(y + dash, Config.HEIGHT)),
                        3,
                    )


class Gate:
    """Muro orizzontale in discesa con un varco mobile da attraversare."""

    def __init__(
        self,
        gap_center: float,
        gap_width: int = 96,
        speed: float = 3.0,
        thickness: int = 26,
        drift: float = 0.55,
    ):
        self.y = float(PLAYFIELD_TOP - thickness - 10)
        self.gap_center = float(gap_center)
        self.gap_width = int(gap_width)
        self.speed = float(speed)
        self.thickness = int(thickness)
        self.drift = float(drift) * random.choice((-1.0, 1.0))
        self.dead = False

    def update(self) -> None:
        if self.dead:
            return

        self.y += self.speed
        self.gap_center += self.drift

        half_gap = self.gap_width / 2
        min_center = half_gap + 8
        max_center = Config.WIDTH - half_gap - 8
        if self.gap_center <= min_center:
            self.gap_center = min_center
            self.drift = abs(self.drift)
        elif self.gap_center >= max_center:
            self.gap_center = max_center
            self.drift = -abs(self.drift)

        if self.y > Config.HEIGHT + self.thickness:
            self.dead = True

    def _rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        gap_left = int(self.gap_center - self.gap_width / 2)
        gap_right = int(self.gap_center + self.gap_width / 2)
        y = int(self.y)
        left = pygame.Rect(0, y, max(0, gap_left), self.thickness)
        right = pygame.Rect(
            min(Config.WIDTH, gap_right),
            y,
            max(0, Config.WIDTH - gap_right),
            self.thickness,
        )
        return left, right

    def collides(self, player: pygame.sprite.Sprite) -> bool:
        if self.dead:
            return False
        left, right = self._rects()
        return left.colliderect(player.rect) or right.colliderect(player.rect)

    def draw(self, screen: pygame.Surface) -> None:
        if self.dead:
            return
        left, right = self._rects()
        for rect in (left, right):
            if rect.width <= 0:
                continue
            pygame.draw.rect(screen, (75, 65, 95), rect)
            pygame.draw.rect(screen, (205, 180, 245), rect, 3)


class AttackDirector:
    """Genera attacchi secondari indipendenti, quindi possono sovrapporsi.

    La difficolta' cresce con il tempo sopravvissuto nello stesso episodio.
    Gli attacchi hanno sempre un preavviso o una traiettoria leggibile.
    """

    def __init__(self):
        self.bullets: list[Bullet] = []
        self.lasers: list[Laser] = []
        self.gates: list[Gate] = []
        self.next_bullet_step = 0
        self.next_laser_step = 0
        self.next_gate_step = 0
        self.reset()

    def reset(self) -> None:
        self.bullets.clear()
        self.lasers.clear()
        self.gates.clear()
        self.next_bullet_step = random.randint(180, 260)
        self.next_laser_step = random.randint(650, 850)
        self.next_gate_step = random.randint(900, 1200)

    @staticmethod
    def level(clock: int) -> int:
        # 0,1,2,3 circa ogni 20 secondi a 50 step/s.
        return min(3, max(0, int(clock) // 1000))

    def _schedule_bullet(self, clock: int, level: int) -> None:
        base = [125, 105, 85, 68][level]
        jitter = random.randint(-18, 25)
        self.next_bullet_step = clock + max(45, base + jitter)

    def _schedule_laser(self, clock: int, level: int) -> None:
        low, high = [(620, 850), (500, 700), (410, 590), (330, 500)][level]
        self.next_laser_step = clock + random.randint(low, high)

    def _schedule_gate(self, clock: int, level: int) -> None:
        low, high = [(900, 1200), (760, 1050), (620, 900), (520, 760)][level]
        self.next_gate_step = clock + random.randint(low, high)

    def _spawn_bullets(self, player_position, level: int) -> None:
        count = [1, 1, 2, 3][level]
        for _ in range(count):
            if level >= 2 and random.random() < 0.35:
                # Ogni tanto minaccia la zona del player, ma con offset per
                # evitare un puro "aimbot" inevitabile.
                x = float(player_position.x + random.randint(-75, 75))
                x = max(15.0, min(Config.WIDTH - 15.0, x))
            else:
                x = float(random.randint(18, Config.WIDTH - 18))

            vx = random.uniform(-1.45, 1.45)
            vy = random.uniform(3.7 + 0.35 * level, 4.5 + 0.45 * level)
            warning = max(18, 30 - 3 * level)
            self.bullets.append(Bullet(x=x, vx=vx, vy=vy, warning_steps=warning))

    def _spawn_laser(self, level: int) -> None:
        orientation = random.choice(("horizontal", "vertical"))
        if orientation == "horizontal":
            coordinate = random.randint(250, Config.HEIGHT - 55)
        else:
            coordinate = random.randint(35, Config.WIDTH - 35)

        warning = max(35, 58 - 5 * level)
        active = 30 + 3 * level
        self.lasers.append(
            Laser(
                orientation=orientation,
                coordinate=coordinate,
                warning_steps=warning,
                active_steps=active,
                thickness=18,
            )
        )

    def _spawn_gate(self, level: int) -> None:
        gap_width = [108, 98, 88, 80][level]
        half = gap_width // 2 + 10
        center = random.randint(half, Config.WIDTH - half)
        speed = 2.7 + 0.25 * level
        drift = 0.35 + 0.15 * level
        self.gates.append(
            Gate(
                gap_center=center,
                gap_width=gap_width,
                speed=speed,
                thickness=26,
                drift=drift,
            )
        )

    def update(self, clock: int, player_position) -> None:
        level = self.level(clock)

        if clock >= self.next_bullet_step:
            self._spawn_bullets(player_position, level)
            self._schedule_bullet(clock, level)

        if clock >= self.next_laser_step:
            self._spawn_laser(level)
            self._schedule_laser(clock, level)

        if clock >= self.next_gate_step:
            self._spawn_gate(level)
            self._schedule_gate(clock, level)

        for bullet in self.bullets:
            bullet.update()
        for laser in self.lasers:
            laser.update()
        for gate in self.gates:
            gate.update()

        self.bullets = [b for b in self.bullets if not b.dead]
        self.lasers = [l for l in self.lasers if not l.dead]
        self.gates = [g for g in self.gates if not g.dead]

    def collides(self, player: pygame.sprite.Sprite) -> bool:
        return (
            any(b.collides(player) for b in self.bullets)
            or any(l.collides(player) for l in self.lasers)
            or any(g.collides(player) for g in self.gates)
        )

    def draw(self, screen: pygame.Surface) -> None:
        # Gate dietro, poi laser, poi proiettili.
        for gate in self.gates:
            gate.draw(screen)
        for laser in self.lasers:
            laser.draw(screen)
        for bullet in self.bullets:
            bullet.draw(screen)
