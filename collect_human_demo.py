#!/usr/bin/env python3
"""
collect_human_demo.py

Registra dimostrazioni umane compatibili con la pipeline DQN del progetto:
- input preprocessato 220x110 grayscale
- stack di 4 frame
- action_repeat = 4
- max-pooling sugli ultimi 2 frame del repeat
- azioni: 0 NOOP, 1 LEFT, 2 RIGHT, 3 UP, 4 DOWN

Salva un file .npz compatto per episodio con:
initial_frame, next_frames, actions, rewards, dones, scores

Gli stack 220x110x4 vengono ricostruiti successivamente. Questo riduce molto
lo spazio rispetto a salvare states e next_states completi per ogni decisione.

Eseguire dalla root del progetto:
    python collect_human_demo.py

Comandi:
    Frecce  = movimento
    Nessun tasto = NOOP
    R       = ricomincia episodio (senza salvarlo)
    ESC     = termina
    SPACE   = dopo la morte, inizia un nuovo episodio
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Path del progetto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
GAME_ROOT = PROJECT_ROOT / "dont_touch_my_presents"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GAME_ROOT))

from rl_common import (  # noqa: E402
    DEFAULT_ACTION_REPEAT,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_STACK_SIZE,
    FrameStack,
    preprocess_frame,
    unpack_step,
)
from src.game.game_env import GameEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRAME_HEIGHT = DEFAULT_FRAME_HEIGHT   # 220
FRAME_WIDTH = DEFAULT_FRAME_WIDTH     # 110
STACK_SIZE = DEFAULT_STACK_SIZE       # 4
ACTION_REPEAT = DEFAULT_ACTION_REPEAT # 4

OUTPUT_DIR = PROJECT_ROOT / "demonstrations"
FPS = 50


def keyboard_action() -> int:
    """Converte lo stato della tastiera nelle stesse 5 azioni del DQN."""
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        return 1
    if keys[pygame.K_RIGHT]:
        return 2
    if keys[pygame.K_UP]:
        return 3
    if keys[pygame.K_DOWN]:
        return 4
    return 0  # NOOP


def process_events() -> str | None:
    """
    Restituisce:
      'quit'    -> chiudi programma
      'restart' -> annulla episodio corrente e ricomincia
      None      -> continua
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if event.key == pygame.K_r:
                return "restart"
    return None


def human_repeated_step(
    env: GameEnv,
    action: int,
    repeat: int,
    clock: pygame.time.Clock,
):
    """
    Equivalente a rl_common.repeated_step(), ma rallentato a FPS reale
    per permettere a un umano di giocare.

    L'azione scelta viene mantenuta per `repeat` step dell'ambiente.
    Come nel DQN, il next observation è il max-pooling degli ultimi 2 frame.
    """
    total_reward = 0.0
    recent_frames: list[np.ndarray] = []
    terminated = False
    truncated = False
    info = {}

    for _ in range(repeat):
        command = process_events()
        if command is not None:
            return None, 0.0, False, False, {}, command

        obs, reward, terminated, truncated, info = unpack_step(env.step(action))
        total_reward += reward
        recent_frames.append(obs)

        # Il GameEnv disegna già sulla finestra pygame durante env.step().
        pygame.display.flip()
        clock.tick(FPS)

        if terminated or truncated:
            break

    if len(recent_frames) >= 2:
        pooled_obs = np.maximum(recent_frames[-2], recent_frames[-1])
    else:
        pooled_obs = recent_frames[-1]

    return pooled_obs, total_reward, terminated, truncated, info, None


def next_demo_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("demo_*.npz"))

    max_id = 0
    for path in existing:
        try:
            max_id = max(max_id, int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    return output_dir / f"demo_{max_id + 1:04d}.npz"


def save_episode(
    path: Path,
    initial_frame: np.ndarray,
    next_frames: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    dones: list[bool],
    scores: list[float],
) -> None:
    if not actions:
        print("Nessuna transizione registrata: episodio non salvato.")
        return

    np.savez_compressed(
        path,
        format_version=np.int32(2),
        initial_frame=np.asarray(initial_frame, dtype=np.uint8),
        next_frames=np.asarray(next_frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int8),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        scores=np.asarray(scores, dtype=np.float32),
        frame_height=np.int32(FRAME_HEIGHT),
        frame_width=np.int32(FRAME_WIDTH),
        stack_size=np.int32(STACK_SIZE),
        action_repeat=np.int32(ACTION_REPEAT),
    )

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\nSalvato: {path}")
    print(f"Transizioni: {len(actions)}")
    print(f"Formato demo: compact v2")
    print(f"Dimensione compressa: {size_mb:.1f} MB")


def wait_after_episode() -> str:
    print("\nSPACE = nuova partita | ESC = termina")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_SPACE:
                    return "continue"
        time.sleep(0.01)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # IMPORTANTE:
    # Non impostare SDL_VIDEODRIVER=dummy: serve una finestra reale per giocare.
    env = GameEnv(render_mode="rgb_array")
    clock = pygame.time.Clock()
    stacker = FrameStack(STACK_SIZE)

    episode_number = 1

    print("=" * 60)
    print("HUMAN DEMONSTRATION RECORDER")
    print("=" * 60)
    print("Frecce   : movimento")
    print("Nessun tasto: NOOP")
    print("R        : annulla e ricomincia episodio")
    print("ESC      : termina")
    print()
    print(
        f"Pipeline DQN: {FRAME_HEIGHT}x{FRAME_WIDTH}x{STACK_SIZE}, "
        f"action_repeat={ACTION_REPEAT}"
    )

    try:
        while True:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            first_frame = preprocess_frame(obs, FRAME_HEIGHT, FRAME_WIDTH)
            state = stacker.reset(first_frame)

            initial_frame = first_frame.copy()
            next_frames: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            dones: list[bool] = []
            scores: list[float] = []

            pygame.display.flip()
            print(f"\n--- Episodio umano {episode_number} ---")

            cancelled = False

            while True:
                command = process_events()
                if command == "quit":
                    return
                if command == "restart":
                    print("Episodio annullato.")
                    cancelled = True
                    break

                # Una decisione umana ogni ACTION_REPEAT frame.
                action = keyboard_action()

                result = human_repeated_step(
                    env=env,
                    action=action,
                    repeat=ACTION_REPEAT,
                    clock=clock,
                )

                next_obs, reward, terminated, truncated, info, command = result

                if command == "quit":
                    return
                if command == "restart":
                    print("Episodio annullato.")
                    cancelled = True
                    break

                next_frame = preprocess_frame(
                    next_obs,
                    FRAME_HEIGHT,
                    FRAME_WIDTH,
                )
                next_state = stacker.append(next_frame)
                done = bool(terminated or truncated)

                # Formato compatto: per ricostruire tutti gli stack basta il
                # frame iniziale + un nuovo frame preprocessato per decisione.
                next_frames.append(next_frame.copy())
                actions.append(int(action))
                rewards.append(float(reward))
                dones.append(done)
                scores.append(float(info.get("score", 0.0)))

                state = next_state

                if done:
                    final_score = float(info.get("score", 0.0))
                    print(
                        f"Episodio terminato | score={final_score:.0f} | "
                        f"decisioni={len(actions)}"
                    )

                    path = next_demo_path(OUTPUT_DIR)
                    save_episode(
                        path,
                        initial_frame,
                        next_frames,
                        actions,
                        rewards,
                        dones,
                        scores,
                    )
                    episode_number += 1
                    break

            if cancelled:
                continue

            choice = wait_after_episode()
            if choice == "quit":
                return

    finally:
        env.close()


if __name__ == "__main__":
    main()
