"""Utility condivise tra notebook di training e CLI di registrazione."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Risoluzione originale del progetto, condivisa tra training e CLI.
# Manteniamo 220x110x4 per non impoverire troppo l'informazione visiva.
DEFAULT_FRAME_HEIGHT = 220
DEFAULT_FRAME_WIDTH = 110
DEFAULT_STACK_SIZE = 4
DEFAULT_ACTION_REPEAT = 2


def preprocess_frame(
    observation: np.ndarray,
    height: int = DEFAULT_FRAME_HEIGHT,
    width: int = DEFAULT_FRAME_WIDTH,
) -> np.ndarray:
    """Converte un frame RGB in grayscale uint8 ridimensionato.

    Usa TensorFlow per evitare una dipendenza implicita da Pillow nella pipeline
    di inferenza e per avere lo stesso comportamento in notebook e CLI.
    """
    import tensorflow as tf

    observation = np.asarray(observation)
    if observation.ndim != 3 or observation.shape[-1] != 3:
        raise ValueError(f"Frame RGB atteso, ricevuto {observation.shape}")

    frame = tf.convert_to_tensor(observation, dtype=tf.uint8)
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, (height, width), method="bilinear")
    frame = tf.squeeze(frame, axis=-1)
    frame = tf.clip_by_value(tf.round(frame), 0, 255)
    return np.ascontiguousarray(frame.numpy().astype(np.uint8))


class FrameStack:
    """Mantiene gli ultimi N frame preprocessati in ordine temporale."""

    def __init__(self, stack_size: int = DEFAULT_STACK_SIZE):
        self.stack_size = int(stack_size)
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack_size)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(np.asarray(first_frame, dtype=np.uint8))
        return self.state()

    def append(self, frame: np.ndarray) -> np.ndarray:
        if not self.frames:
            return self.reset(frame)
        self.frames.append(np.asarray(frame, dtype=np.uint8))
        return self.state()

    def state(self) -> np.ndarray:
        if len(self.frames) != self.stack_size:
            raise RuntimeError("Frame stack non inizializzato.")
        return np.stack(tuple(self.frames), axis=-1)


def unpack_step(result):
    """Normalizza step() di Gym/Gymnasium nel formato a 5 valori."""
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), False, info
    raise RuntimeError(f"Formato inatteso restituito da env.step(): {len(result)} valori")


def repeated_step(env, action: int, repeat: int = DEFAULT_ACTION_REPEAT):
    """Ripete l'azione e fa max-pooling sugli ultimi due frame RGB."""
    total_reward = 0.0
    recent_frames: list[np.ndarray] = []
    terminated = truncated = False
    info: dict[str, Any] = {}

    for _ in range(int(repeat)):
        obs, reward, terminated, truncated, info = unpack_step(env.step(action))
        total_reward += reward
        recent_frames.append(obs)
        if terminated or truncated:
            break

    if not recent_frames:
        raise RuntimeError("env.step() non ha prodotto alcun frame.")
    if len(recent_frames) >= 2:
        obs = np.maximum(recent_frames[-2], recent_frames[-1])
    else:
        obs = recent_frames[-1]
    return obs, total_reward, terminated, truncated, info


def greedy_action(model, state: np.ndarray) -> int:
    """Seleziona l'azione con Q-value massimo, senza esplorazione."""
    import tensorflow as tf

    q_values = model(tf.expand_dims(state, axis=0), training=False)
    return int(tf.argmax(q_values[0]).numpy())


class BestModelCheckpoint:
    """Salva atomicamente il modello completo quando migliora il best score."""

    def __init__(
        self,
        model,
        model_path: os.PathLike[str] | str,
        algorithm: str,
        static_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path)
        if self.model_path.suffix != ".keras":
            raise ValueError("Il percorso del checkpoint deve terminare con '.keras'.")

        self.algorithm = algorithm
        self.static_metadata = dict(static_metadata or {})
        self.metadata_path = self.model_path.with_suffix(".json")
        self.best_score = -math.inf
        self.best_episode: Optional[int] = None
        self._load_existing_metadata()

    def _load_existing_metadata(self) -> None:
        if not self.metadata_path.exists() or not self.model_path.exists():
            return
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.best_score = float(metadata.get("best_score", -math.inf))
            episode = metadata.get("episode")
            self.best_episode = int(episode) if episode is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.best_score = -math.inf
            self.best_episode = None

    def update(
        self,
        score: float,
        episode: int,
        timestep: int,
        time_passed: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        score = float(score)
        if not math.isfinite(score) or score <= self.best_score:
            return False

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_model_path = self.model_path.with_name(f".{self.model_path.stem}.tmp.keras")
        temp_metadata_path = self.metadata_path.with_name(f".{self.metadata_path.stem}.tmp.json")

        metadata: dict[str, Any] = {
            "algorithm": self.algorithm,
            "best_score": score,
            "episode": int(episode),
            "timestep": int(timestep),
            "time_passed": None if time_passed is None else float(time_passed),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(self.model_path),
            **self.static_metadata,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            self.model.save(temp_model_path, overwrite=True)
            temp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            os.replace(temp_model_path, self.model_path)
            os.replace(temp_metadata_path, self.metadata_path)
        finally:
            if temp_model_path.exists():
                temp_model_path.unlink()
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()

        self.best_score = score
        self.best_episode = int(episode)
        return True


def load_policy_metadata(model_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Carica i metadati del checkpoint, se presenti."""
    path = Path(model_path).with_suffix(".json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}

# ---------------------------------------------------------------------------
# Compatibilità con il notebook legacy (220x110x4, resize nearest-neighbor).
# Il nuovo DQN non usa queste funzioni, ma mantenerle evita di rompere
# rl_project_legacy.ipynb e i vecchi checkpoint DQN(lambda).
# ---------------------------------------------------------------------------

def preprocess_observation(
    observation: np.ndarray,
    previous_state: Optional["State"],
    output_dim: Optional[list[int]] = None,
):
    import tensorflow as tf

    if output_dim is None:
        output_dim = [220, 110]
    output = tf.convert_to_tensor(observation)
    output = tf.expand_dims(output, axis=0)
    output = tf.image.rgb_to_grayscale(output)
    output = tf.image.resize(
        output, output_dim, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    output = tf.squeeze(output, axis=3)
    if previous_state is None:
        output = tf.stack([output] * 4, axis=3)
    else:
        output = tf.expand_dims(output, axis=3)
        output = tf.concat((previous_state.state[:, :, :, 1:], output), axis=3)
    return output


class State:
    """State legacy usato dal vecchio notebook 220x110x4."""

    def __init__(self, observation: np.ndarray, previous_state: Optional["State"] = None):
        self.state = preprocess_observation(observation, previous_state)

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, State):
            return False
        return not (self.state.numpy() - obj.state.numpy()).any()
