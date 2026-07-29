"""Utility per demo umane: Behavior Cloning e inizializzazione replay DQN.

Formato demo supportato:
- compact v2: initial_frame + next_frames.

Nel Behavior Cloning, per ogni episodio vengono escluse:
- la transizione terminale (done=True);
- le N transizioni immediatamente precedenti, configurabili con
  exclude_pre_terminal.

Il replay DQN conserva invece tutte le transizioni, comprese quelle terminali.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


BASE_KEYS = (
    "actions",
    "rewards",
    "dones",
    "initial_frame",
    "next_frames",
)


@dataclass(frozen=True)
class DemoSummary:
    files: tuple[Path, ...]
    transitions: int
    action_counts: np.ndarray


def find_demo_files(demo_dir: str | Path) -> list[Path]:
    """Trova le demo compact v2 nella cartella indicata."""
    demo_dir = Path(demo_dir)
    return sorted(path for path in demo_dir.glob("*.npz") if path.is_file())


def _scalar_int(data, key: str, default: int | None = None) -> int | None:
    if key not in data.files:
        return default
    return int(np.asarray(data[key]).reshape(()).item())


def validate_demo_metadata(
    data,
    path: Path,
    frame_height: int,
    frame_width: int,
    stack_size: int,
    action_repeat: int,
) -> None:
    """Valida una demo compact v2 e la compatibilita' col training corrente."""
    missing = [key for key in BASE_KEYS if key not in data.files]
    if missing:
        raise ValueError(
            f"Demo {path.name} incompleta/non compatibile con compact v2: "
            f"mancano {missing}."
        )

    expected = {
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "stack_size": int(stack_size),
        "action_repeat": int(action_repeat),
    }
    for key, wanted in expected.items():
        saved = _scalar_int(data, key, None)
        if saved is not None and saved != wanted:
            raise ValueError(
                f"Demo incompatibile {path.name}: {key}={saved}, atteso {wanted}."
            )


def _validate_transition_lengths(data, path: Path) -> int:
    n = len(data["actions"])
    if not (
        len(data["rewards"]) == len(data["dones"]) == len(data["next_frames"]) == n
    ):
        raise ValueError(f"{path.name}: lunghezze transition arrays incoerenti.")
    return int(n)


def _make_bc_keep_mask(
    dones: np.ndarray,
    exclude_pre_terminal: int = 18,
) -> np.ndarray:
    """Mask delle transizioni da tenere nel Behavior Cloning.

    La transizione terminale viene SEMPRE esclusa.
    Inoltre vengono escluse le `exclude_pre_terminal` transizioni
    immediatamente precedenti a ogni terminale.

    Esempio con exclude_pre_terminal=5:
        ... | TIENI | X | X | X | X | X | X(done=True)

    Quindi vengono scartate 6 transizioni totali per episodio:
    5 precedenti + quella terminale.
    """
    dones = np.asarray(dones, dtype=np.bool_)
    keep = np.ones(len(dones), dtype=np.bool_)

    exclude_pre_terminal = max(0, int(exclude_pre_terminal))

    for terminal_idx in np.flatnonzero(dones):
        start = max(0, int(terminal_idx) - exclude_pre_terminal)
        keep[start : int(terminal_idx) + 1] = False

    return keep


def summarize_demos(
    demo_dir: str | Path,
    num_actions: int,
    frame_height: int,
    frame_width: int,
    stack_size: int,
    action_repeat: int,
    exclude_pre_terminal: int = 18,
) -> DemoSummary:
    """Riassume le demo usando le stesse transizioni effettive del BC.

    I conteggi delle azioni e il numero di transizioni escludono:
    - la transizione terminale;
    - le N transizioni immediatamente precedenti.

    In questo modo i class weights del Behavior Cloning vengono calcolati
    esattamente sui campioni che verranno realmente usati per il training.
    """
    files = find_demo_files(demo_dir)
    if not files:
        raise FileNotFoundError(
            f"Nessuna demo trovata in {Path(demo_dir)}. "
            "Registra prima alcune partite con collect_human_demo.py."
        )

    counts = np.zeros(int(num_actions), dtype=np.int64)
    total = 0

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            validate_demo_metadata(
                data,
                path,
                frame_height,
                frame_width,
                stack_size,
                action_repeat,
            )

            actions = np.asarray(data["actions"], dtype=np.int64)
            dones = np.asarray(data["dones"], dtype=np.bool_)
            n = _validate_transition_lengths(data, path)

            if actions.ndim != 1:
                raise ValueError(f"{path.name}: actions deve essere 1D.")
            if dones.ndim != 1:
                raise ValueError(f"{path.name}: dones deve essere 1D.")
            if actions.size != n or dones.size != n:
                raise ValueError(f"{path.name}: arrays transizioni incoerenti.")
            if actions.size == 0:
                continue

            if int(actions.min()) < 0 or int(actions.max()) >= num_actions:
                raise ValueError(
                    f"{path.name}: azione fuori range 0..{num_actions - 1}."
                )

            keep_mask = _make_bc_keep_mask(
                dones,
                exclude_pre_terminal=exclude_pre_terminal,
            )
            kept_actions = actions[keep_mask]

            if kept_actions.size:
                counts += np.bincount(
                    kept_actions,
                    minlength=num_actions,
                )[:num_actions]
                total += int(kept_actions.size)

    if total == 0:
        raise ValueError(
            "Le demo esistono, ma dopo il filtro pre-terminale "
            "non restano transizioni per il Behavior Cloning."
        )

    return DemoSummary(tuple(files), total, counts)


def compute_action_weights(action_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(action_counts, dtype=np.float64)
    total = float(counts.sum())
    num_actions = int(counts.size)

    if total <= 0:
        return np.ones(num_actions, dtype=np.float32)

    safe = np.maximum(counts, 1.0)
    weights = np.sqrt(total / (num_actions * safe))

    weighted_mean = float(np.sum(weights * counts) / total)
    if weighted_mean > 0:
        weights /= weighted_mean

    return np.clip(weights, 0.35, 3.0).astype(np.float32)


def split_demo_files(
    files: Sequence[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Compatibilita' con eventuali script che fanno ancora split automatico.

    Se train/validation sono gia' separati in cartelle, questa funzione
    semplicemente non serve e puo' essere ignorata.
    """
    files = list(files)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    if len(files) <= 1 or validation_fraction <= 0:
        return files, []

    val_count = max(1, int(round(len(files) * validation_fraction)))
    val_count = min(val_count, len(files) - 1)

    return files[val_count:], files[:val_count]


def _compact_state_batches(
    data,
    path: Path,
    batch_size: int,
    frame_height: int,
    frame_width: int,
    stack_size: int,
    action_weights: np.ndarray | None,
    exclude_pre_terminal: int = 18,
):
    """Ricostruisce gli stack compact v2 in streaming.

    I frame esclusi dal BC vengono comunque usati per aggiornare correttamente
    lo stack temporale; semplicemente non diventano esempi supervisionati.
    """
    actions = np.asarray(data["actions"], dtype=np.int32)
    dones = np.asarray(data["dones"], dtype=np.bool_)
    next_frames = np.asarray(data["next_frames"], dtype=np.uint8)
    initial = np.asarray(data["initial_frame"], dtype=np.uint8)

    n = _validate_transition_lengths(data, path)

    if initial.shape != (frame_height, frame_width):
        raise ValueError(
            f"{path.name}: initial_frame shape {initial.shape} non valida."
        )

    if next_frames.shape != (n, frame_height, frame_width):
        raise ValueError(
            f"{path.name}: next_frames shape {next_frames.shape} non valida."
        )

    if actions.shape != (n,):
        raise ValueError(f"{path.name}: actions shape {actions.shape} non valida.")

    if dones.shape != (n,):
        raise ValueError(f"{path.name}: dones shape {dones.shape} non valida.")

    keep_mask = _make_bc_keep_mask(
        dones,
        exclude_pre_terminal=exclude_pre_terminal,
    )

    state = np.repeat(initial[..., None], stack_size, axis=-1)

    bx: list[np.ndarray] = []
    by: list[int] = []
    bw: list[float] = []

    for i in range(n):
        if keep_mask[i]:
            action = int(actions[i])

            bx.append(state.copy())
            by.append(action)
            bw.append(
                1.0
                if action_weights is None
                else float(action_weights[action])
            )

        # Anche se il campione e' escluso dal BC, lo stack temporale
        # deve continuare ad avanzare correttamente.
        state = np.concatenate(
            [state[..., 1:], next_frames[i][..., None]],
            axis=-1,
        )

        if len(bx) >= batch_size:
            yield (
                np.asarray(bx, dtype=np.uint8),
                np.asarray(by, dtype=np.int32),
                np.asarray(bw, dtype=np.float32),
            )
            bx.clear()
            by.clear()
            bw.clear()

    if bx:
        yield (
            np.asarray(bx, dtype=np.uint8),
            np.asarray(by, dtype=np.int32),
            np.asarray(bw, dtype=np.float32),
        )


def make_demo_batch_generator(
    files: Sequence[Path],
    batch_size: int,
    frame_height: int,
    frame_width: int,
    stack_size: int,
    action_repeat: int,
    action_weights: np.ndarray | None,
    shuffle: bool,
    seed: int,
    exclude_pre_terminal: int = 5,
):
    """Factory di batch BC per sole demo compact v2.

    Carica al massimo una demo alla volta in RAM.
    """
    files = tuple(Path(p) for p in files)
    epoch_counter = {"value": 0}

    def generator() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        epoch = epoch_counter["value"]
        epoch_counter["value"] += 1

        rng = np.random.default_rng(seed + epoch)
        file_order = list(files)

        if shuffle:
            rng.shuffle(file_order)

        for path in file_order:
            with np.load(path, allow_pickle=False) as data:
                validate_demo_metadata(
                    data,
                    path,
                    frame_height,
                    frame_width,
                    stack_size,
                    action_repeat,
                )

                yield from _compact_state_batches(
                    data=data,
                    path=path,
                    batch_size=int(batch_size),
                    frame_height=frame_height,
                    frame_width=frame_width,
                    stack_size=stack_size,
                    action_weights=action_weights,
                    exclude_pre_terminal=exclude_pre_terminal,
                )

    return generator


def populate_frame_replay_from_demos(
    replay,
    demo_dir: str | Path,
    frame_height: int,
    frame_width: int,
    stack_size: int,
    action_repeat: int,
    max_transitions: int | None = None,
    seed: int = 0,
) -> dict:
    """Popola FrameReplayBuffer con episodi umani compact v2 completi.

    IMPORTANTE:
    qui NON viene applicato il filtro del Behavior Cloning.

    Il replay DQN conserva:
    - tutte le transizioni pre-terminali;
    - la transizione terminale done=True;
    - il reward negativo della collisione.
    """
    files = find_demo_files(demo_dir)

    if not files:
        return {
            "files": 0,
            "transitions": 0,
            "skipped": 0,
        }

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    loaded_files = 0
    loaded_transitions = 0
    skipped = 0

    # Lascia un piccolo margine per l'inizio di nuovi episodi nel replay.
    hard_limit = replay.capacity - max(
        2,
        min(100, replay.capacity // 20),
    )

    if max_transitions is not None:
        hard_limit = min(
            hard_limit,
            max(0, int(max_transitions)),
        )

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            validate_demo_metadata(
                data,
                path,
                frame_height,
                frame_width,
                stack_size,
                action_repeat,
            )

            actions = np.asarray(data["actions"], dtype=np.int32)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=np.bool_)

            first_frame = np.asarray(
                data["initial_frame"],
                dtype=np.uint8,
            )
            next_frames = np.asarray(
                data["next_frames"],
                dtype=np.uint8,
            )

            n = _validate_transition_lengths(data, path)

            if n == 0 or loaded_transitions + n > hard_limit:
                skipped += 1
                continue

            if actions.shape != (n,):
                raise ValueError(
                    f"{path.name}: actions shape {actions.shape} non valida."
                )

            if rewards.shape != (n,):
                raise ValueError(
                    f"{path.name}: rewards shape {rewards.shape} non valida."
                )

            if dones.shape != (n,):
                raise ValueError(
                    f"{path.name}: dones shape {dones.shape} non valida."
                )

            if first_frame.shape != (frame_height, frame_width):
                raise ValueError(
                    f"{path.name}: initial_frame shape "
                    f"{first_frame.shape} non valida."
                )

            if next_frames.shape != (n, frame_height, frame_width):
                raise ValueError(
                    f"{path.name}: next_frames shape "
                    f"{next_frames.shape} non valida."
                )

            replay.start_episode(first_frame)

            for i in range(n):
                done = bool(dones[i]) or i == n - 1

                replay.append(
                    int(actions[i]),
                    float(rewards[i]),
                    done,
                    next_frames[i],
                )

                if done and i != n - 1:
                    raise ValueError(
                        f"{path.name}: done prima della fine; "
                        "atteso un episodio per file."
                    )

            loaded_files += 1
            loaded_transitions += n

    return {
        "files": loaded_files,
        "transitions": loaded_transitions,
        "skipped": skipped,
    }
