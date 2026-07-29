#!/usr/bin/env python3
"""Registra episodi greedy usando un checkpoint DQN salvato.

Per DQN la priorità predefinita è:
1. checkpoints/dqn_latest.weights.h5 (ultimo checkpoint di training)
2. checkpoints/dqn_best.keras (fallback)

Usa --checkpoint best per forzare esplicitamente il miglior modello.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registra una run greedy usando un checkpoint salvato."
    )
    parser.add_argument(
        "--algorithm", choices=("dqn", "dqnlambda"), default="dqn",
        help="Algoritmo/checkpoint da usare.",
    )
    parser.add_argument(
        "--checkpoint", choices=("latest", "best"), default="latest",
        help=(
            "Per DQN: 'latest' carica l'ultimo checkpoint salvato e usa il best "
            "come fallback; 'best' forza dqn_best.keras."
        ),
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Percorso esplicito a un file .keras oppure .weights.h5.",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=Path("best_model_videos"),
        help="Cartella in cui salvare i video.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Massimo numero di decisioni agente per episodio.",
    )
    parser.add_argument("--name-prefix", default="model-run")
    parser.add_argument("--show-window", action="store_true")
    return parser.parse_args()


def resolve_model_path(project_root: Path, args: argparse.Namespace) -> Path:
    if args.model is not None:
        return args.model.expanduser().resolve()

    checkpoints = project_root / "checkpoints"
    if args.algorithm == "dqnlambda":
        return (checkpoints / "dqnlambda_best.keras").resolve()

    best = checkpoints / "dqn_best.keras"
    latest = checkpoints / "dqn_latest.weights.h5"

    if args.checkpoint == "best":
        return best.resolve()
    if latest.exists():
        return latest.resolve()
    return best.resolve()


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes deve essere almeno 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps deve essere almeno 1")

    if not args.show_window:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    project_root = Path(__file__).resolve().parent
    game_root = project_root / "dont_touch_my_presents"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(game_root))

    import tensorflow as tf
    try:
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        from gym.wrappers import RecordVideo

    from rl_common import (
        DEFAULT_ACTION_REPEAT,
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_STACK_SIZE,
        FrameStack,
        State,
        greedy_action,
        load_policy_metadata,
        preprocess_frame,
        repeated_step,
    )
    from src.game.game_env import GameEnv
    from train_dqn import build_q_network

    model_path = resolve_model_path(project_root, args)
    if not model_path.exists():
        raise SystemExit(
            f"Checkpoint non trovato: {model_path}\n"
            "Esegui prima il training oppure passa --model /percorso/checkpoint"
        )

    metadata = load_policy_metadata(model_path)
    frame_height = int(metadata.get("frame_height", DEFAULT_FRAME_HEIGHT))
    frame_width = int(metadata.get("frame_width", DEFAULT_FRAME_WIDTH))
    stack_size = int(metadata.get("stack_size", DEFAULT_STACK_SIZE))
    action_repeat = int(metadata.get("action_repeat", DEFAULT_ACTION_REPEAT))

    video_dir = args.video_dir.expanduser().resolve()
    video_dir.mkdir(parents=True, exist_ok=True)

    base_env = GameEnv(render_mode="rgb_array")
    num_actions = int(base_env.action_space.n)

    if model_path.name.endswith(".weights.h5"):
        # L'ultimo checkpoint salva i pesi: ricostruiamo la stessa architettura
        # e carichiamo i pesi PRIMA di fare qualsiasi prediction.
        model = build_q_network(
            (frame_height, frame_width, stack_size), num_actions
        )
        model.load_weights(model_path)
        use_legacy_pipeline = False
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
        model_input_shape = tuple(model.input_shape[1:])
        use_legacy_pipeline = (
            not metadata.get("preprocessing")
            and len(model_input_shape) == 3
            and model_input_shape[0] == 220
            and model_input_shape[1] == 110
        )
        if use_legacy_pipeline:
            frame_height, frame_width, stack_size, action_repeat = 220, 110, 4, 1

    env = RecordVideo(
        base_env,
        video_folder=str(video_dir),
        episode_trigger=lambda _episode_id: True,
        name_prefix=args.name_prefix,
    )

    scores: list[float] = []
    try:
        for episode in range(args.episodes):
            episode_seed = None if args.seed is None else args.seed + episode
            reset_result = env.reset(seed=episode_seed)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            final_info = {}
            if use_legacy_pipeline:
                legacy_state = State(observation)
                for step in range(1, args.max_steps + 1):
                    q_values = model(legacy_state.state, training=False)
                    action = int(tf.argmax(q_values[0]).numpy())
                    observation, _reward, terminated, truncated, final_info = repeated_step(
                        env, action, 1
                    )
                    legacy_state = State(observation, previous_state=legacy_state)
                    if terminated or truncated:
                        break
            else:
                stacker = FrameStack(stack_size)
                state = stacker.reset(
                    preprocess_frame(observation, frame_height, frame_width)
                )
                for step in range(1, args.max_steps + 1):
                    action = greedy_action(model, state)
                    observation, _reward, terminated, truncated, final_info = repeated_step(
                        env, action, action_repeat
                    )
                    state = stacker.append(
                        preprocess_frame(observation, frame_height, frame_width)
                    )
                    if terminated or truncated:
                        break

            score = float(final_info.get("score", 0.0))
            scores.append(score)
            print(
                f"Episodio {episode + 1}/{args.episodes}: "
                f"score={score:g}, decisioni={step}"
            )
    finally:
        env.close()

    print(f"Checkpoint caricato: {model_path}")
    print(
        f"Pipeline: {frame_height}x{frame_width}x{stack_size}, "
        f"action_repeat={action_repeat}"
    )
    print(f"Video salvati in: {video_dir}")
    print(f"Miglior score registrato: {max(scores):g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
