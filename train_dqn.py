"""DQN migliorato per Avoid Hands.

Integra le parti più utili di dqn_fixed:
- ambiente senza dipendenze dal wall-clock (implementato nei componenti del gioco);
- preprocessing 220x110 grayscale + stack di 4 frame;
- action repeat configurabile con max-pooling degli ultimi due frame;
- replay buffer memory-efficient basato sui frame;
- Double DQN;
- Huber loss e gradient clipping;
- warm-up, train-every e soft update Polyak della target network;
- valutazione greedy periodica su seed fissi e salvataggio del best sulla media reward;
- resume completo periodico di online/target/Adam/replay buffer.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from demo_utils import compute_action_weights, populate_frame_replay_from_demos

BEST_SELECTION_METRIC = "greedy_mean_episode_reward"


from rl_common import (
    BestModelCheckpoint,
    DEFAULT_ACTION_REPEAT,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_STACK_SIZE,
    FrameStack,
    preprocess_frame,
    repeated_step,
)


@dataclass
class DQNConfig:
    seed: int = 42

    frame_height: int = DEFAULT_FRAME_HEIGHT
    frame_width: int = DEFAULT_FRAME_WIDTH
    stack_size: int = DEFAULT_STACK_SIZE
    action_repeat: int = DEFAULT_ACTION_REPEAT

    # Replay dell'agente: contiene esclusivamente le nuove esperienze generate
    # durante il DQN ed e' circolare.
    replay_capacity: int = 50_000
    # Ora il warm-up conta solo le transizioni generate dall'agente. Le demo
    # protette non fanno partire immediatamente gli aggiornamenti DQN.
    learning_starts: int = 5_000
    batch_size: int = 32

    # Pipeline ibrida: Behavior Cloning -> DQN.
    # Se il file BC esiste e non c'e' un resume, viene usato come init.
    demo_dir: str = "demonstrations/train"
    bc_pretrained_weights_path: str = "checkpoints/dqn_bc_pretrained.weights.h5"
    initialize_from_bc_if_available: bool = True
    populate_replay_from_demos: bool = True

    # Replay demo separato, protetto e mai sovrascritto dall'agente.
    # La capacity e' leggermente maggiore del limite per contenere anche i frame
    # terminali e i confini tra episodi.
    demo_replay_capacity: int = 65_000
    demo_replay_max_transitions: int = 60_000

    # Composizione dei minibatch DQN. All'inizio 16/32 campioni arrivano dalle
    # demo; la quota scende gradualmente fino a 8/32, ma non arriva mai a zero.
    demo_batch_fraction_start: float = 0.50
    demo_batch_fraction_end: float = 0.50
    demo_batch_fraction_decay_steps: int = 100_000

    # Loss supervisionata ausiliaria applicata soltanto alla porzione demo del
    # minibatch. La TD loss continua a essere calcolata su demo + agente.
    # Il peso parte alto per proteggere la policy BC e poi diminuisce, lasciando
    # progressivamente piu' liberta' al reinforcement learning.
    bc_aux_loss_weight_start: float = 0.50
    bc_aux_loss_weight_end: float = 0.30
    bc_aux_loss_decay_steps: int = 200_000
    # Deve coincidere con il filtro usato nel pretraining BC: la loss
    # supervisionata non imita le azioni terminali o immediatamente precedenti
    # alla collisione, mentre la TD loss continua a usarle normalmente.
    bc_aux_exclude_pre_terminal: int = 18

    # Dopo imitation learning non serve ripartire da epsilon=1: conserveremmo
    # poco della policy imitata. Rimane comunque esplorazione significativa.
    epsilon_start_after_bc: float = 0.15
    training_origin_path: str = "checkpoints/dqn_training_origin.json"

    gamma: float = 0.99
    learning_rate: float = 1e-5
    train_every: int = 4
    gradient_steps: int = 1
    # Soft/Polyak update dopo ogni gradient step:
    # target <- (1 - tau) * target + tau * online
    # tau piccolo evita i salti bruschi dell'hard update periodico.
    target_soft_tau: float = 0.0005
    max_grad_norm: float = 10.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    num_episodes: int = 30_000
    max_agent_steps: int = 300_000
    max_steps_per_episode: int = 5_000

    # Valutazione separata dalla raccolta epsilon-greedy del training.
    # I seed sono sempre gli stessi, cosi' le medie sono confrontabili.
    greedy_eval_every_episodes: int = 50
    greedy_eval_episodes: int = 10
    greedy_eval_seed_base: int = 10_000
    # Salva subito una baseline della rete BC (o del checkpoint ripreso) prima
    # di proseguire il training.
    greedy_eval_at_start: bool = True

    best_model_path: str = "checkpoints/dqn_best.keras"
    latest_weights_path: str = "checkpoints/dqn_latest.weights.h5"
    latest_every_episodes: int = 150

    # Checkpoint completo per un resume fedele: online + target + Adam + replay.
    # E' volutamente meno frequente perche' il replay buffer puo' essere grande.
    full_state_dir: str = "checkpoints/dqn_training_state"
    full_state_every_episodes: int = 250

    resume_if_available: bool = True
    resume_fallback_to_best: bool = True

    record_video: bool = False
    video_every_episodes: int = 150
    video_folder: str = "videos/training"
    log_every_episodes: int = 5


class FrameReplayBuffer:
    """Replay buffer che memorizza un solo frame uint8 per step.

    Gli stack di frame vengono ricostruiti al campionamento, evitando di
    duplicare 4 volte i pixel per ogni transizione.
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: tuple[int, int],
        stack_size: int,
        seed: int = 0,
    ):
        if capacity <= stack_size + 2:
            raise ValueError("Replay capacity troppo piccola.")

        self.capacity = int(capacity)
        self.frame_shape = tuple(frame_shape)
        self.stack_size = int(stack_size)
        self.rng = np.random.default_rng(seed)

        self.frames = np.empty((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.frame_ids = np.full(capacity, -1, dtype=np.int64)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self.episode_steps = np.full(capacity, -1, dtype=np.int32)
        self.valid_transition = np.zeros(capacity, dtype=np.bool_)

        self.next_slot = 0
        self.size = 0
        self.global_frame_id = 0
        self.episode_counter = -1

        self.current_idx: Optional[int] = None
        self.current_episode_id: Optional[int] = None
        self.current_episode_step = 0

    @property
    def num_transitions(self) -> int:
        return int(self.valid_transition.sum())

    @property
    def memory_megabytes(self) -> float:
        arrays = [
            self.frames,
            self.actions,
            self.rewards,
            self.dones,
            self.frame_ids,
            self.episode_ids,
            self.episode_steps,
            self.valid_transition,
        ]
        return sum(a.nbytes for a in arrays) / (1024**2)

    def _store_observation(self, frame: np.ndarray, episode_id: int, episode_step: int) -> int:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape {frame.shape}; attesa {self.frame_shape}.")

        idx = self.next_slot
        old_frame_id = self.frame_ids[idx]
        if old_frame_id >= 0:
            previous_idx = (idx - 1) % self.capacity
            if (
                self.valid_transition[previous_idx]
                and self.frame_ids[previous_idx] + 1 == old_frame_id
            ):
                self.valid_transition[previous_idx] = False

        self.valid_transition[idx] = False
        self.frames[idx] = frame
        self.frame_ids[idx] = self.global_frame_id
        self.episode_ids[idx] = episode_id
        self.episode_steps[idx] = episode_step

        self.global_frame_id += 1
        self.next_slot = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.current_idx = idx
        return idx

    def start_episode(self, first_frame: np.ndarray) -> None:
        self.episode_counter += 1
        self.current_episode_id = self.episode_counter
        self.current_episode_step = 0
        self._store_observation(first_frame, self.current_episode_id, 0)

    def append(self, action: int, reward: float, done: bool, next_frame: np.ndarray) -> None:
        if self.current_idx is None or self.current_episode_id is None:
            raise RuntimeError("Chiamare start_episode() prima di append().")

        transition_idx = self.current_idx
        self.actions[transition_idx] = int(action)
        self.rewards[transition_idx] = float(reward)
        self.dones[transition_idx] = bool(done)
        self.valid_transition[transition_idx] = True

        self.current_episode_step += 1
        next_idx = self._store_observation(
            next_frame, self.current_episode_id, self.current_episode_step
        )
        if self.frame_ids[next_idx] != self.frame_ids[transition_idx] + 1:
            raise RuntimeError("Sequenza dei frame non valida.")

        if done:
            self.current_idx = None
            self.current_episode_id = None

    def _encode_stack(self, idx: int) -> np.ndarray:
        frame_id = self.frame_ids[idx]
        episode_id = self.episode_ids[idx]
        episode_step = self.episode_steps[idx]
        if frame_id < 0:
            raise RuntimeError("Tentativo di leggere uno slot vuoto.")

        frames_newest_first = [self.frames[idx]]
        for offset in range(1, self.stack_size):
            previous_idx = (idx - offset) % self.capacity
            is_consecutive = (
                self.frame_ids[previous_idx] == frame_id - offset
                and self.episode_ids[previous_idx] == episode_id
                and self.episode_steps[previous_idx] == episode_step - offset
            )
            if not is_consecutive:
                break
            frames_newest_first.append(self.frames[previous_idx])

        oldest = frames_newest_first[-1]
        while len(frames_newest_first) < self.stack_size:
            frames_newest_first.append(oldest)
        return np.stack(frames_newest_first[::-1], axis=-1)

    def sample(self, batch_size: int, return_indices: bool = False):
        candidates = np.flatnonzero(self.valid_transition)
        if candidates.size < batch_size:
            raise RuntimeError(
                f"Solo {candidates.size} transizioni disponibili; batch richiesto: {batch_size}."
            )

        indices = self.rng.choice(candidates, size=batch_size, replace=False)
        states = np.empty(
            (batch_size, *self.frame_shape, self.stack_size), dtype=np.uint8
        )
        next_states = np.empty_like(states)

        for row, idx in enumerate(indices):
            next_idx = (idx + 1) % self.capacity
            if self.frame_ids[next_idx] != self.frame_ids[idx] + 1:
                raise RuntimeError("Next-frame sovrascritto o non consecutivo.")
            states[row] = self._encode_stack(idx)
            next_states[row] = self._encode_stack(next_idx)

        batch = (
            states,
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            next_states,
            self.dones[indices].astype(np.float32),
        )
        if return_indices:
            return batch, indices.copy()
        return batch

    def save(self, path: str | Path) -> None:
        """Salva atomicamente il replay buffer senza cambiarne la struttura."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.npz")

        # Prima del primo wrap gli slot usati sono [0:size). Dopo il wrap il
        # buffer e' pieno, quindi vanno salvati tutti gli slot circolari.
        n = self.capacity if self.size == self.capacity else self.size

        try:
            np.savez_compressed(
                tmp,
                capacity=np.array(self.capacity, dtype=np.int64),
                frame_shape=np.asarray(self.frame_shape, dtype=np.int64),
                stack_size=np.array(self.stack_size, dtype=np.int64),
                frames=self.frames[:n],
                actions=self.actions[:n],
                rewards=self.rewards[:n],
                dones=self.dones[:n],
                frame_ids=self.frame_ids[:n],
                episode_ids=self.episode_ids[:n],
                episode_steps=self.episode_steps[:n],
                valid_transition=self.valid_transition[:n],
                next_slot=np.array(self.next_slot, dtype=np.int64),
                size=np.array(self.size, dtype=np.int64),
                global_frame_id=np.array(self.global_frame_id, dtype=np.int64),
                episode_counter=np.array(self.episode_counter, dtype=np.int64),
                rng_state=np.array(json.dumps(self.rng.bit_generator.state)),
            )
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def load(self, path: str | Path) -> None:
        """Ripristina un replay buffer salvato con save()."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            saved_capacity = int(data["capacity"])
            saved_frame_shape = tuple(int(x) for x in data["frame_shape"].tolist())
            saved_stack_size = int(data["stack_size"])

            if saved_capacity != self.capacity:
                raise ValueError(
                    f"Replay incompatibile: capacity {saved_capacity} != {self.capacity}."
                )
            if saved_frame_shape != self.frame_shape:
                raise ValueError(
                    f"Replay incompatibile: frame_shape {saved_frame_shape} "
                    f"!= {self.frame_shape}."
                )
            if saved_stack_size != self.stack_size:
                raise ValueError(
                    f"Replay incompatibile: stack_size {saved_stack_size} "
                    f"!= {self.stack_size}."
                )

            n = len(data["frames"])
            if n > self.capacity:
                raise ValueError("Replay salvato piu' grande della capacity corrente.")

            # Ripulisce gli slot non presenti nel file (caso buffer non pieno).
            self.frames.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.dones.fill(False)
            self.frame_ids.fill(-1)
            self.episode_ids.fill(-1)
            self.episode_steps.fill(-1)
            self.valid_transition.fill(False)

            self.frames[:n] = data["frames"]
            self.actions[:n] = data["actions"]
            self.rewards[:n] = data["rewards"]
            self.dones[:n] = data["dones"]
            self.frame_ids[:n] = data["frame_ids"]
            self.episode_ids[:n] = data["episode_ids"]
            self.episode_steps[:n] = data["episode_steps"]
            self.valid_transition[:n] = data["valid_transition"]

            self.next_slot = int(data["next_slot"])
            self.size = int(data["size"])
            self.global_frame_id = int(data["global_frame_id"])
            self.episode_counter = int(data["episode_counter"])
            self.rng.bit_generator.state = json.loads(str(data["rng_state"].item()))

        # Un checkpoint puo' essere stato scritto durante un'interruzione.
        # Al resume si parte sempre da un nuovo reset dell'ambiente, quindi non
        # si tenta di continuare una partita a meta'.
        self.current_idx = None
        self.current_episode_id = None
        self.current_episode_step = 0


def build_q_network(
    input_shape: tuple[int, int, int],
    num_actions: int,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """CNN compatta, senza BatchNorm, adatta sia al BC sia al DQN.

    La BatchNorm e' evitata perche' le demo arrivano in sequenze temporali molto
    correlate e il DQN vede una distribuzione non stazionaria. In questi casi le
    moving statistics possono produrre un forte divario tra training=True e
    training=False. Il terzo strato usa stride 2 per ridurre drasticamente il
    numero di parametri nel Dense senza perdere completamente l'informazione
    spaziale.
    """
    regularizer = keras.regularizers.l2(1e-5)

    inputs = keras.Input(shape=input_shape, dtype=tf.uint8, name="frames")
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    x = keras.layers.Conv2D(
        32, kernel_size=8, strides=4, activation="relu",
        kernel_initializer="he_normal", kernel_regularizer=regularizer,
        name="conv1",
    )(x)
    x = keras.layers.Conv2D(
        64, kernel_size=4, strides=2, activation="relu",
        kernel_initializer="he_normal", kernel_regularizer=regularizer,
        name="conv2",
    )(x)
    x = keras.layers.Conv2D(
        64, kernel_size=3, strides=2, activation="relu",
        kernel_initializer="he_normal", kernel_regularizer=regularizer,
        name="conv3",
    )(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(
        512, activation="relu", kernel_initializer="he_normal",
        kernel_regularizer=regularizer, name="dense1",
    )(x)
    # Nel BC usa 0.20; nel DQN il valore di default resta 0.0.
    x = keras.layers.Dropout(float(dropout_rate), name="dropout1")(x)

    outputs = keras.layers.Dense(int(num_actions), name="q_values")(x)
    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="avoid_hands_double_dqn_compact",
    )


class DQNLearner:
    """Learner Double DQN con Huber loss e clipping globale dei gradienti."""

    def __init__(
        self,
        online_network: keras.Model,
        target_network: keras.Model,
        learning_rate: float,
        gamma: float,
        max_grad_norm: float,
        target_soft_tau: float,
    ):
        self.online = online_network
        self.target = target_network
        self.target_soft_tau = tf.constant(
            float(target_soft_tau), dtype=tf.float32
        )
        self.hard_update_target()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-7)
        self.td_loss_fn = keras.losses.Huber()
        self.bc_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=keras.losses.Reduction.NONE,
        )
        num_actions = int(self.online.output_shape[-1])
        self.bc_action_weights = tf.Variable(
            tf.ones((num_actions,), dtype=tf.float32),
            trainable=False,
            name="bc_action_weights",
        )
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.max_grad_norm = float(max_grad_norm)
        self.update_count = 0

    @tf.function
    def _train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        demo_count,
        demo_bc_mask,
        bc_loss_weight,
    ):
        # Double DQN: online sceglie l'azione, target la valuta.
        next_online_q = self.online(next_states, training=False)
        next_actions = tf.argmax(next_online_q, axis=1, output_type=tf.int32)
        next_target_q_all = self.target(next_states, training=False)
        next_target_q = tf.gather(
            next_target_q_all, next_actions, axis=1, batch_dims=1
        )
        targets = rewards + (1.0 - dones) * self.gamma * next_target_q
        targets = tf.stop_gradient(targets)

        demo_count = tf.cast(demo_count, tf.int32)
        demo_bc_mask = tf.cast(demo_bc_mask, tf.float32)
        bc_loss_weight = tf.cast(bc_loss_weight, tf.float32)

        with tf.GradientTape() as tape:
            all_q = self.online(states, training=True)
            selected_q = tf.gather(all_q, actions, axis=1, batch_dims=1)

            # 1) TD loss su tutto il minibatch: demo + esperienze dell'agente.
            td_loss = self.td_loss_fn(targets, selected_q)

            # 2) Behavior Cloning ausiliario solo sui primi demo_count elementi.
            # sample_protected_mixed_batch concatena infatti demo prima di agent.
            # I Q-value vengono trattati come logits: la cross-entropy mantiene
            # alta la preferenza relativa per l'azione scelta dall'umano.
            def compute_bc_loss():
                demo_actions = actions[:demo_count]
                per_example_loss = self.bc_loss_fn(
                    demo_actions,
                    all_q[:demo_count],
                )
                # Mantiene lo stesso bilanciamento per classe usato nel BC e
                # annulla gli esempi terminali/pre-terminali esclusi dal BC.
                sample_weights = (
                    tf.gather(self.bc_action_weights, demo_actions)
                    * demo_bc_mask[:demo_count]
                )
                return tf.math.divide_no_nan(
                    tf.reduce_sum(per_example_loss * sample_weights),
                    tf.reduce_sum(sample_weights),
                )

            bc_loss = tf.cond(
                demo_count > 0,
                compute_bc_loss,
                lambda: tf.constant(0.0, dtype=tf.float32),
            )

            total_loss = td_loss + bc_loss_weight * bc_loss

        gradients = tape.gradient(total_loss, self.online.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(gradients, self.online.trainable_variables))

        # Soft/Polyak update dopo ogni gradient step. La target si avvicina
        # gradualmente alla online, senza sostituzioni brusche ogni N step.
        tau = self.target_soft_tau
        for target_variable, online_variable in zip(
            self.target.weights, self.online.weights
        ):
            target_variable.assign_add(
                tau * (online_variable - target_variable)
            )

        return (
            total_loss,
            td_loss,
            bc_loss,
            tf.reduce_sum(demo_bc_mask[:demo_count]),
            tf.reduce_mean(selected_q),
            tf.reduce_mean(targets),
            grad_norm,
        )

    def train(
        self,
        batch,
        demo_count: int,
        demo_bc_mask: np.ndarray,
        bc_loss_weight: float,
    ):
        states, actions, rewards, next_states, dones = batch
        result = self._train_step(
            tf.convert_to_tensor(states, dtype=tf.uint8),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.uint8),
            tf.convert_to_tensor(dones, dtype=tf.float32),
            tf.convert_to_tensor(demo_count, dtype=tf.int32),
            tf.convert_to_tensor(demo_bc_mask, dtype=tf.float32),
            tf.convert_to_tensor(bc_loss_weight, dtype=tf.float32),
        )
        self.update_count += 1
        return tuple(float(x.numpy()) for x in result)

    def set_bc_action_weights(self, action_weights: np.ndarray) -> None:
        action_weights = np.asarray(action_weights, dtype=np.float32)
        expected_shape = tuple(self.bc_action_weights.shape)
        if action_weights.shape != expected_shape:
            raise ValueError(
                f"BC action weights shape {action_weights.shape}; "
                f"attesa {expected_shape}."
            )
        self.bc_action_weights.assign(action_weights)

    def hard_update_target(self) -> None:
        """Copia completa, usata solo all'inizializzazione o nel resume legacy."""
        self.target.set_weights(self.online.get_weights())


def linear_epsilon(
    step: int,
    cfg: DQNConfig,
    epsilon_start: float | None = None,
) -> float:
    start = cfg.epsilon_start if epsilon_start is None else float(epsilon_start)
    fraction = min(max(step, 0) / cfg.epsilon_decay_steps, 1.0)
    return start + fraction * (cfg.epsilon_end - start)


def linear_demo_fraction(step: int, cfg: DQNConfig) -> float:
    """Quota di ogni minibatch estratta dal replay demo protetto."""
    decay_steps = max(1, int(cfg.demo_batch_fraction_decay_steps))
    fraction = min(max(int(step), 0) / decay_steps, 1.0)
    value = (
        float(cfg.demo_batch_fraction_start)
        + fraction
        * (
            float(cfg.demo_batch_fraction_end)
            - float(cfg.demo_batch_fraction_start)
        )
    )
    return float(np.clip(value, 0.0, 1.0))


def linear_bc_aux_weight(step: int, cfg: DQNConfig) -> float:
    """Peso della BC loss, contato dall'inizio effettivo degli update DQN."""
    decay_steps = max(1, int(cfg.bc_aux_loss_decay_steps))
    steps_after_warmup = max(int(step) - int(cfg.learning_starts), 0)
    fraction = min(steps_after_warmup / decay_steps, 1.0)
    value = (
        float(cfg.bc_aux_loss_weight_start)
        + fraction
        * (
            float(cfg.bc_aux_loss_weight_end)
            - float(cfg.bc_aux_loss_weight_start)
        )
    )
    return float(max(value, 0.0))


def mixed_batch_sizes(
    batch_size: int,
    demo_fraction: float,
    demo_available: int,
) -> tuple[int, int]:
    """Calcola quanti campioni demo/agente usare nel minibatch."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size deve essere positivo.")

    if demo_available <= 0:
        return 0, batch_size

    demo_count = int(round(batch_size * float(demo_fraction)))
    demo_count = min(max(demo_count, 0), batch_size, int(demo_available))
    return demo_count, batch_size - demo_count


def concatenate_replay_batches(*batches):
    """Concatena minibatch replay mantenendo il formato di FrameReplayBuffer."""
    batches = tuple(batch for batch in batches if batch is not None)
    if not batches:
        raise ValueError("Nessun batch da concatenare.")
    if len(batches) == 1:
        return batches[0]

    return tuple(
        np.concatenate([batch[column] for batch in batches], axis=0)
        for column in range(5)
    )


def build_demo_bc_eligible_mask(
    demo_replay: FrameReplayBuffer,
    exclude_pre_terminal: int,
) -> np.ndarray:
    """Marca le transizioni demo utilizzabili dalla loss supervisionata.

    Il replay DQN conserva anche terminale e pre-terminale per la TD loss. La
    BC loss invece replica il filtro del pretraining ed evita di imitare azioni
    compiute subito prima della collisione.
    """
    exclude_pre_terminal = max(0, int(exclude_pre_terminal))
    eligible = np.zeros(demo_replay.capacity, dtype=np.bool_)
    candidates = np.flatnonzero(demo_replay.valid_transition)

    if candidates.size == 0:
        return eligible

    for episode_id in np.unique(demo_replay.episode_ids[candidates]):
        episode_indices = candidates[
            demo_replay.episode_ids[candidates] == episode_id
        ]
        terminal_indices = episode_indices[demo_replay.dones[episode_indices]]
        if terminal_indices.size == 0:
            continue

        terminal_step = int(
            np.max(demo_replay.episode_steps[terminal_indices])
        )
        episode_steps = demo_replay.episode_steps[episode_indices]
        eligible[episode_indices] = (
            episode_steps < terminal_step - exclude_pre_terminal
        )

    return eligible


def sample_protected_mixed_batch(
    demo_replay: FrameReplayBuffer,
    agent_replay: FrameReplayBuffer,
    demo_bc_eligible_mask: np.ndarray,
    batch_size: int,
    demo_fraction: float,
):
    """Campiona un minibatch misto da due replay fisicamente separati.

    Il replay demo non riceve mai esperienze dell'agente e non viene quindi
    sovrascritto. Il replay agente contiene solo transizioni prodotte dal DQN.
    """
    demo_count, agent_count = mixed_batch_sizes(
        batch_size=batch_size,
        demo_fraction=demo_fraction,
        demo_available=demo_replay.num_transitions,
    )

    if agent_replay.num_transitions < agent_count:
        raise RuntimeError(
            "Replay agente non ancora sufficiente: "
            f"{agent_replay.num_transitions} < {agent_count}."
        )

    if demo_count > 0:
        demo_batch, demo_indices = demo_replay.sample(
            demo_count,
            return_indices=True,
        )
        sampled_demo_bc_mask = demo_bc_eligible_mask[demo_indices].astype(
            np.float32,
            copy=True,
        )
    else:
        demo_batch = None
        sampled_demo_bc_mask = np.empty((0,), dtype=np.float32)

    agent_batch = agent_replay.sample(agent_count) if agent_count > 0 else None
    mixed_batch = concatenate_replay_batches(demo_batch, agent_batch)
    return mixed_batch, demo_count, agent_count, sampled_demo_bc_mask


def load_training_origin(cfg: DQNConfig) -> dict:
    path = Path(cfg.training_origin_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def save_training_origin(cfg: DQNConfig, initialized_from_bc: bool) -> None:
    path = Path(cfg.training_origin_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "initialized_from_behavior_cloning": bool(initialized_from_bc),
        "bc_weights": str(cfg.bc_pretrained_weights_path) if initialized_from_bc else None,
        "demo_dir": str(cfg.demo_dir),
        "epsilon_start": (
            float(cfg.epsilon_start_after_bc)
            if initialized_from_bc
            else float(cfg.epsilon_start)
        ),
        "replay_layout": "protected_demo_plus_agent_v1",
        "demo_replay_capacity": int(cfg.demo_replay_capacity),
        "demo_replay_max_transitions": int(cfg.demo_replay_max_transitions),
        "target_update_strategy": "soft_polyak_after_gradient_step",
        "target_soft_tau": float(cfg.target_soft_tau),
        "agent_replay_capacity": int(cfg.replay_capacity),
        "demo_batch_fraction_start": float(cfg.demo_batch_fraction_start),
        "demo_batch_fraction_end": float(cfg.demo_batch_fraction_end),
        "demo_batch_fraction_decay_steps": int(
            cfg.demo_batch_fraction_decay_steps
        ),
        "bc_aux_loss": "sparse_categorical_crossentropy_on_demo_samples",
        "bc_aux_loss_weight_start": float(cfg.bc_aux_loss_weight_start),
        "bc_aux_loss_weight_end": float(cfg.bc_aux_loss_weight_end),
        "bc_aux_loss_decay_steps": int(cfg.bc_aux_loss_decay_steps),
        "bc_aux_exclude_pre_terminal": int(cfg.bc_aux_exclude_pre_terminal),
    }
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def select_action(
    network: keras.Model,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(num_actions))
    q_values = network(np.expand_dims(state, axis=0), training=False)
    return int(tf.argmax(q_values[0]).numpy())


def make_training_env(cfg: DQNConfig):
    from src.game.game_env import GameEnv

    env = GameEnv(render_mode="rgb_array")
    if cfg.record_video:
        try:
            from gymnasium.wrappers import RecordVideo
        except ImportError:
            from gym.wrappers import RecordVideo
        env = RecordVideo(
            env,
            video_folder=cfg.video_folder,
            episode_trigger=lambda episode_id: episode_id % cfg.video_every_episodes == 0,
            name_prefix="avoid-hands-training",
        )
    return env


def evaluate_greedy_policy(
    env,
    model: keras.Model,
    cfg: DQNConfig,
) -> dict[str, float]:
    """Valuta la policy con epsilon=0 su una banca di seed fissi.

    Usa l'ambiente gia' esistente, ma non aggiunge transizioni ai replay e non
    esegue aggiornamenti della rete. Lo stato RNG globale di Python e NumPy
    viene ripristinato alla fine, per non cambiare casualmente il training.
    """
    if cfg.greedy_eval_episodes <= 0:
        raise ValueError("greedy_eval_episodes deve essere positivo.")

    # Se il training usa un wrapper video, la valutazione lavora direttamente
    # sull'ambiente sottostante e non altera i contatori del wrapper.
    eval_env = env.unwrapped if hasattr(env, "unwrapped") else env

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()

    rewards: list[float] = []
    scores: list[float] = []
    lengths: list[int] = []
    times_passed: list[float] = []

    try:
        for eval_index in range(int(cfg.greedy_eval_episodes)):
            eval_seed = int(cfg.greedy_eval_seed_base) + eval_index
            reset_result = eval_env.reset(seed=eval_seed)
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            stacker = FrameStack(cfg.stack_size)
            frame = preprocess_frame(obs, cfg.frame_height, cfg.frame_width)
            state = stacker.reset(frame)

            episode_reward = 0.0
            episode_length = 0
            last_info: dict = {}

            while episode_length < cfg.max_steps_per_episode:
                # Greedy puro: nessuna estrazione casuale e nessuna epsilon.
                q_values = model(np.expand_dims(state, axis=0), training=False)
                action = int(tf.argmax(q_values[0]).numpy())

                next_obs, reward, terminated, truncated, last_info = repeated_step(
                    eval_env,
                    action,
                    cfg.action_repeat,
                )
                episode_reward += float(reward)
                episode_length += 1

                next_frame = preprocess_frame(
                    next_obs,
                    cfg.frame_height,
                    cfg.frame_width,
                )
                state = stacker.append(next_frame)

                if terminated or truncated:
                    break

            rewards.append(float(episode_reward))
            scores.append(float(last_info.get("score", 0.0)))
            lengths.append(int(episode_length))
            times_passed.append(
                float(
                    last_info.get(
                        "time_passed",
                        episode_length * cfg.action_repeat,
                    )
                )
            )
    finally:
        # GameEnv.reset(seed=...) usa random.seed/np.random.seed globali.
        # Ripristinandoli, la valutazione non cambia la sequenza del training.
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_length": float(np.mean(lengths)),
        "mean_time_passed": float(np.mean(times_passed)),
    }


def run_greedy_evaluation(
    env,
    model: keras.Model,
    cfg: DQNConfig,
    best_checkpoint: BestModelCheckpoint,
    history: dict,
    episode: int,
    total_agent_steps: int,
) -> tuple[dict[str, float], bool]:
    """Esegue la valutazione e salva il best solo se migliora la media reward."""
    metrics = evaluate_greedy_policy(env, model, cfg)

    improved = best_checkpoint.update(
        # Best selezionato sulla media reward, non sul singolo score di training.
        score=metrics["mean_reward"],
        episode=episode,
        timestep=total_agent_steps,
        time_passed=metrics["mean_time_passed"],
        extra_metadata={
            "selection_metric": BEST_SELECTION_METRIC,
            "greedy_epsilon": 0.0,
            "greedy_eval_episodes": int(cfg.greedy_eval_episodes),
            "greedy_eval_seed_base": int(cfg.greedy_eval_seed_base),
            "greedy_mean_reward": metrics["mean_reward"],
            "greedy_std_reward": metrics["std_reward"],
            "greedy_min_reward": metrics["min_reward"],
            "greedy_max_reward": metrics["max_reward"],
            "greedy_mean_score": metrics["mean_score"],
            "greedy_std_score": metrics["std_score"],
            "greedy_mean_length": metrics["mean_length"],
        },
    )

    history["eval_episode"].append(int(episode))
    history["eval_agent_step"].append(int(total_agent_steps))
    history["eval_mean_reward"].append(float(metrics["mean_reward"]))
    history["eval_std_reward"].append(float(metrics["std_reward"]))
    history["eval_mean_score"].append(float(metrics["mean_score"]))
    history["eval_std_score"].append(float(metrics["std_score"]))
    history["eval_mean_length"].append(float(metrics["mean_length"]))
    history["best_eval_mean_reward"].append(float(best_checkpoint.best_score))

    status = "NUOVO BEST SALVATO" if improved else "nessun miglioramento"
    print(
        "GREEDY EVAL | "
        f"episodio {episode} | step {total_agent_steps} | "
        f"N={cfg.greedy_eval_episodes} | "
        f"mean reward {metrics['mean_reward']:.4f} "
        f"± {metrics['std_reward']:.4f} | "
        f"mean score {metrics['mean_score']:.2f} "
        f"± {metrics['std_score']:.2f} | "
        f"best mean reward {best_checkpoint.best_score:.4f} | "
        f"{status}"
    )

    return metrics, improved


def save_history(history: dict, path: str | Path = "checkpoints/dqn_history.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_history(path: str | Path = "checkpoints/dqn_history.json") -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def latest_metadata_path(weights_path: str | Path) -> Path:
    return Path(weights_path).with_suffix(".json")


def save_latest_checkpoint(
    model: keras.Model,
    cfg: DQNConfig,
    completed_episode: int,
    total_agent_steps: int,
) -> None:
    """Salva atomicamente gli ultimi pesi e lo stato minimo per il resume."""
    weights_path = Path(cfg.latest_weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_weights = weights_path.with_name(f".{weights_path.stem}.tmp.weights.h5")
    metadata_path = latest_metadata_path(weights_path)
    tmp_metadata = metadata_path.with_name(f".{metadata_path.stem}.tmp.json")

    metadata = {
        "algorithm": "dqn",
        "checkpoint_kind": "latest",
        "episode": int(completed_episode),
        "timestep": int(total_agent_steps),
        "frame_height": int(cfg.frame_height),
        "frame_width": int(cfg.frame_width),
        "stack_size": int(cfg.stack_size),
        "action_repeat": int(cfg.action_repeat),
        "preprocessing": "grayscale_bilinear_uint8",
        "network": "conv32-64-64_stride2_dense512_no_bn_dropout0",
        "gamma": float(cfg.gamma),
        "learning_rate": float(cfg.learning_rate),
        "target_update_strategy": "soft_polyak_after_gradient_step",
        "target_soft_tau": float(cfg.target_soft_tau),
        "bc_aux_loss": "sparse_categorical_crossentropy_on_demo_samples",
        "bc_aux_loss_weight_start": float(cfg.bc_aux_loss_weight_start),
        "bc_aux_loss_weight_end": float(cfg.bc_aux_loss_weight_end),
        "bc_aux_loss_decay_steps": int(cfg.bc_aux_loss_decay_steps),
        "bc_aux_exclude_pre_terminal": int(cfg.bc_aux_exclude_pre_terminal),
    }

    try:
        model.save_weights(tmp_weights)
        tmp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp_weights.replace(weights_path)
        tmp_metadata.replace(metadata_path)
    finally:
        if tmp_weights.exists():
            tmp_weights.unlink()
        if tmp_metadata.exists():
            tmp_metadata.unlink()


def load_resume_checkpoint(
    model: keras.Model, cfg: DQNConfig
) -> tuple[str | None, int, int]:
    """Carica il checkpoint PRIMA del training. Priorità: latest -> best -> nuovo."""
    if not cfg.resume_if_available:
        return None, 0, 0

    latest_path = Path(cfg.latest_weights_path)
    if latest_path.exists():
        model.load_weights(latest_path)
        metadata = {}
        metadata_path = latest_metadata_path(latest_path)
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                metadata = {}
        episode = int(metadata.get("episode", 0) or 0)
        timestep = int(metadata.get("timestep", 0) or 0)
        # Compatibilità con checkpoint creati dalla versione precedente, che
        # salvava i pesi latest ma non il relativo JSON di stato.
        if episode == 0 or timestep == 0:
            previous_history = load_history()
            agent_steps = previous_history.get("agent_step", [])
            scores = previous_history.get("episode_score", [])
            if episode == 0 and isinstance(scores, list):
                episode = len(scores)
            if timestep == 0 and isinstance(agent_steps, list) and agent_steps:
                timestep = int(agent_steps[-1])
        return str(latest_path), episode, timestep

    best_path = Path(cfg.best_model_path)
    if cfg.resume_fallback_to_best and best_path.exists():
        metadata = {}
        metadata_path = best_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                metadata = {}

        # Non usare come resume un vecchio "best" scelto dal massimo score di
        # un singolo episodio epsilon-greedy: ha una semantica incompatibile.
        if metadata.get("selection_metric") != BEST_SELECTION_METRIC:
            print(
                "ATTENZIONE: checkpoint best ignorato per il resume: "
                "non e' stato selezionato tramite greedy mean reward."
            )
            return None, 0, 0

        loaded = keras.models.load_model(best_path, compile=False)
        if tuple(loaded.input_shape[1:]) != tuple(model.input_shape[1:]):
            raise ValueError(
                f"Checkpoint best incompatibile: input {loaded.input_shape[1:]} "
                f"!= {model.input_shape[1:]}"
            )
        model.set_weights(loaded.get_weights())
        episode = int(metadata.get("episode", 0) or 0)
        timestep = int(metadata.get("timestep", 0) or 0)
        return str(best_path), episode, timestep

    return None, 0, 0


def full_state_metadata_path(cfg: DQNConfig) -> Path:
    return Path(cfg.full_state_dir) / "state.json"


def full_state_history_path(cfg: DQNConfig) -> Path:
    return Path(cfg.full_state_dir) / "history.json"


def save_full_training_state(
    checkpoint_manager: tf.train.CheckpointManager,
    demo_replay: FrameReplayBuffer,
    agent_replay: FrameReplayBuffer,
    learner: DQNLearner,
    cfg: DQNConfig,
    completed_episode: int,
    total_agent_steps: int,
    agent_rng: np.random.Generator,
    history: dict,
) -> None:
    """Salva uno snapshot coerente del training con replay separati.

    Viene serializzato soltanto il replay dell'agente, perché il replay demo è
    immutabile e viene ricostruito dai file .npz ad ogni avvio. Nel metadata
    viene comunque salvato lo stato RNG del replay demo, così un resume completo
    mantiene anche la sequenza di campionamento.
    """
    state_dir = Path(cfg.full_state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    episode_tag = max(int(completed_episode), 0)
    agent_replay_path = state_dir / (
        f"agent_replay_{episode_tag:08d}_{int(total_agent_steps):012d}.npz"
    )

    # 1) Salva solo il replay circolare dell'agente.
    agent_replay.save(agent_replay_path)

    # 2) Online + target + optimizer Adam.
    tf_checkpoint_path = checkpoint_manager.save(
        checkpoint_number=max(int(total_agent_steps), 0)
    )
    if tf_checkpoint_path is None:
        raise RuntimeError("Impossibile salvare il checkpoint TensorFlow completo.")

    # 3) History coerente con lo snapshot.
    save_history(history, full_state_history_path(cfg))

    metadata = {
        "algorithm": "dqn",
        "checkpoint_kind": "full_training_state",
        "replay_layout": "protected_demo_plus_agent_v1",
        "episode": int(completed_episode),
        "timestep": int(total_agent_steps),
        "learner_update_count": int(learner.update_count),
        "agent_replay_file": agent_replay_path.name,
        "tf_checkpoint": Path(tf_checkpoint_path).name,
        "agent_rng_state": agent_rng.bit_generator.state,
        "demo_replay_rng_state": demo_replay.rng.bit_generator.state,
        "demo_replay_num_transitions": int(demo_replay.num_transitions),
        "agent_replay_num_transitions": int(agent_replay.num_transitions),
        "frame_height": int(cfg.frame_height),
        "frame_width": int(cfg.frame_width),
        "stack_size": int(cfg.stack_size),
        "action_repeat": int(cfg.action_repeat),
        "demo_replay_capacity": int(cfg.demo_replay_capacity),
        "agent_replay_capacity": int(cfg.replay_capacity),
        "demo_replay_max_transitions": int(cfg.demo_replay_max_transitions),
        "target_update_strategy": "soft_polyak_after_gradient_step",
        "target_soft_tau": float(cfg.target_soft_tau),
        "bc_aux_loss": "sparse_categorical_crossentropy_on_demo_samples",
        "bc_aux_loss_weight_start": float(cfg.bc_aux_loss_weight_start),
        "bc_aux_loss_weight_end": float(cfg.bc_aux_loss_weight_end),
        "bc_aux_loss_decay_steps": int(cfg.bc_aux_loss_decay_steps),
        "bc_aux_exclude_pre_terminal": int(cfg.bc_aux_exclude_pre_terminal),
    }

    metadata_path = full_state_metadata_path(cfg)
    tmp_metadata = metadata_path.with_name(f".{metadata_path.stem}.tmp.json")
    tmp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_metadata.replace(metadata_path)

    # Conserva soltanto il replay agente indicato dal nuovo state.json.
    for old_replay in state_dir.glob("agent_replay_*.npz"):
        if old_replay != agent_replay_path:
            try:
                old_replay.unlink()
            except OSError:
                pass


def load_full_training_state(
    checkpoint: tf.train.Checkpoint,
    demo_replay: FrameReplayBuffer,
    agent_replay: FrameReplayBuffer,
    learner: DQNLearner,
    cfg: DQNConfig,
    agent_rng: np.random.Generator,
) -> tuple[str | None, int, int]:
    """Ripristina rete, optimizer e replay agente.

    I vecchi full checkpoint con un unico replay non vengono caricati, perché
    mescolavano demo e dati dell'agente e non possono garantire la nuova
    separazione protetta. In quel caso si prova il resume legacy latest/best.
    """
    if not cfg.resume_if_available:
        return None, 0, 0

    metadata_path = full_state_metadata_path(cfg)
    if not metadata_path.exists():
        return None, 0, 0

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ATTENZIONE: full checkpoint ignorato, state.json non valido: {exc}")
        return None, 0, 0

    if metadata.get("replay_layout") != "protected_demo_plus_agent_v1":
        print(
            "ATTENZIONE: full checkpoint con replay unico ignorato; "
            "la nuova versione richiede replay demo/agente separati. "
            "Provo il resume legacy latest/best."
        )
        return None, 0, 0

    state_dir = Path(cfg.full_state_dir)
    agent_replay_path = state_dir / str(metadata.get("agent_replay_file", ""))
    tf_checkpoint_name = str(metadata.get("tf_checkpoint", ""))
    tf_checkpoint_path = state_dir / "tf" / tf_checkpoint_name

    if (
        not agent_replay_path.exists()
        or not Path(str(tf_checkpoint_path) + ".index").exists()
    ):
        print(
            "ATTENZIONE: full checkpoint incompleto; provo il resume legacy "
            "latest/best."
        )
        return None, 0, 0

    expected = {
        "frame_height": int(cfg.frame_height),
        "frame_width": int(cfg.frame_width),
        "stack_size": int(cfg.stack_size),
        "action_repeat": int(cfg.action_repeat),
        "demo_replay_capacity": int(cfg.demo_replay_capacity),
        "agent_replay_capacity": int(cfg.replay_capacity),
        "demo_replay_max_transitions": int(cfg.demo_replay_max_transitions),
    }
    for key, value in expected.items():
        saved_value = metadata.get(key)
        if saved_value is not None and int(saved_value) != value:
            raise ValueError(
                f"Full checkpoint incompatibile: {key}={saved_value} != {value}."
            )

    saved_target_strategy = metadata.get("target_update_strategy")
    if (
        saved_target_strategy is not None
        and saved_target_strategy != "soft_polyak_after_gradient_step"
    ):
        print(
            "ATTENZIONE: il full checkpoint proviene da una strategia target "
            f"diversa ({saved_target_strategy}); la target ripristinata verra' "
            "aggiornata con soft update da questo momento."
        )

    saved_tau = metadata.get("target_soft_tau")
    if saved_tau is not None and not np.isclose(
        float(saved_tau), float(cfg.target_soft_tau)
    ):
        print(
            "ATTENZIONE: target_soft_tau del checkpoint="
            f"{float(saved_tau):.6f}, configurazione corrente="
            f"{float(cfg.target_soft_tau):.6f}. Uso il valore corrente."
        )

    status = checkpoint.restore(str(tf_checkpoint_path))
    status.expect_partial()
    agent_replay.load(agent_replay_path)

    learner.update_count = int(metadata.get("learner_update_count", 0) or 0)

    rng_state = metadata.get("agent_rng_state")
    if isinstance(rng_state, dict):
        agent_rng.bit_generator.state = rng_state

    demo_rng_state = metadata.get("demo_replay_rng_state")
    if isinstance(demo_rng_state, dict):
        demo_replay.rng.bit_generator.state = demo_rng_state

    expected_demo_transitions = metadata.get("demo_replay_num_transitions")
    if (
        expected_demo_transitions is not None
        and int(expected_demo_transitions) != demo_replay.num_transitions
    ):
        print(
            "ATTENZIONE: il replay demo ricostruito contiene "
            f"{demo_replay.num_transitions} transizioni, mentre il checkpoint "
            f"ne usava {int(expected_demo_transitions)}. "
            "Controlla che demonstrations/train non sia cambiata."
        )

    episode = int(metadata.get("episode", 0) or 0)
    timestep = int(metadata.get("timestep", 0) or 0)
    return str(metadata_path), episode, timestep


def train_dqn(cfg: DQNConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if not (0.0 <= cfg.demo_batch_fraction_start <= 1.0):
        raise ValueError("demo_batch_fraction_start deve essere tra 0 e 1.")
    if not (0.0 <= cfg.demo_batch_fraction_end <= 1.0):
        raise ValueError("demo_batch_fraction_end deve essere tra 0 e 1.")
    if cfg.demo_replay_max_transitions >= cfg.demo_replay_capacity:
        raise ValueError(
            "demo_replay_capacity deve essere maggiore di "
            "demo_replay_max_transitions per contenere i frame di confine."
        )
    if cfg.greedy_eval_every_episodes <= 0:
        raise ValueError("greedy_eval_every_episodes deve essere positivo.")
    if cfg.greedy_eval_episodes <= 0:
        raise ValueError("greedy_eval_episodes deve essere positivo.")
    if not (0.0 < cfg.target_soft_tau <= 1.0):
        raise ValueError("target_soft_tau deve essere in (0, 1].")
    if cfg.bc_aux_loss_weight_start < 0.0:
        raise ValueError("bc_aux_loss_weight_start non puo' essere negativo.")
    if cfg.bc_aux_loss_weight_end < 0.0:
        raise ValueError("bc_aux_loss_weight_end non puo' essere negativo.")
    if cfg.bc_aux_loss_decay_steps <= 0:
        raise ValueError("bc_aux_loss_decay_steps deve essere positivo.")
    if cfg.bc_aux_exclude_pre_terminal < 0:
        raise ValueError("bc_aux_exclude_pre_terminal non puo' essere negativo.")

    env = make_training_env(cfg)
    num_actions = int(env.action_space.n)
    input_shape = (cfg.frame_height, cfg.frame_width, cfg.stack_size)

    online = build_q_network(input_shape, num_actions)
    target = build_q_network(input_shape, num_actions)
    learner = DQNLearner(
        online_network=online,
        target_network=target,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        max_grad_norm=cfg.max_grad_norm,
        target_soft_tau=cfg.target_soft_tau,
    )

    # Crea subito gli slot di Adam, cosi' tf.train.Checkpoint puo' ripristinare
    # anche momentum/variance dell'optimizer prima del primo train step.
    if hasattr(learner.optimizer, "build"):
        learner.optimizer.build(online.trainable_variables)

    # Replay 1: demo umane, caricato una volta e mai modificato dall'agente.
    demo_replay = FrameReplayBuffer(
        capacity=cfg.demo_replay_capacity,
        frame_shape=(cfg.frame_height, cfg.frame_width),
        stack_size=cfg.stack_size,
        seed=cfg.seed + 1,
    )

    # Replay 2: sole esperienze generate dall'agente, circolare.
    agent_replay = FrameReplayBuffer(
        capacity=cfg.replay_capacity,
        frame_shape=(cfg.frame_height, cfg.frame_width),
        stack_size=cfg.stack_size,
        seed=cfg.seed + 2,
    )
    stacker = FrameStack(cfg.stack_size)

    # Il replay demo viene ricostruito sempre, anche al resume. Non viene
    # salvato nei full checkpoint per evitare di duplicare centinaia di MB.
    if cfg.populate_replay_from_demos:
        demo_stats = populate_frame_replay_from_demos(
            replay=demo_replay,
            demo_dir=cfg.demo_dir,
            frame_height=cfg.frame_height,
            frame_width=cfg.frame_width,
            stack_size=cfg.stack_size,
            action_repeat=cfg.action_repeat,
            max_transitions=cfg.demo_replay_max_transitions,
            seed=cfg.seed,
        )
        print(
            "REPLAY DEMO PROTETTO: "
            f"{demo_stats['transitions']} transizioni da "
            f"{demo_stats['files']} episodi umani "
            f"(saltati {demo_stats['skipped']})."
        )
    else:
        print("REPLAY DEMO PROTETTO: disabilitato.")

    demo_bc_eligible_mask = build_demo_bc_eligible_mask(
        demo_replay,
        exclude_pre_terminal=cfg.bc_aux_exclude_pre_terminal,
    )
    eligible_actions = demo_replay.actions[demo_bc_eligible_mask]
    bc_action_counts = np.bincount(
        eligible_actions,
        minlength=num_actions,
    )
    bc_action_weights = compute_action_weights(bc_action_counts)
    learner.set_bc_action_weights(bc_action_weights)
    print(
        "AUX BC dataset: "
        f"{int(demo_bc_eligible_mask.sum())} transizioni eleggibili | "
        f"escluse terminale + {cfg.bc_aux_exclude_pre_terminal} precedenti | "
        f"action weights={np.round(bc_action_weights, 3).tolist()}"
    )

    full_checkpoint = tf.train.Checkpoint(
        online=online,
        target=target,
        optimizer=learner.optimizer,
    )
    full_tf_dir = Path(cfg.full_state_dir) / "tf"
    full_tf_dir.mkdir(parents=True, exist_ok=True)
    full_checkpoint_manager = tf.train.CheckpointManager(
        full_checkpoint,
        directory=str(full_tf_dir),
        max_to_keep=2,
    )

    # Priorita': full training state -> latest weights -> best -> nuovo.
    resume_source, completed_episodes, total_agent_steps = (
        load_full_training_state(
            full_checkpoint,
            demo_replay,
            agent_replay,
            learner,
            cfg,
            rng,
        )
    )
    full_resume = resume_source is not None

    if not full_resume:
        resume_source, completed_episodes, total_agent_steps = load_resume_checkpoint(
            online, cfg
        )
        # Nel resume legacy viene recuperata solo online: target sincronizzata,
        # optimizer nuovo e replay agente vuoto.
        learner.hard_update_target()

    if resume_source is not None:
        resume_kind = "COMPLETO" if full_resume else "LEGACY"
        print(
            f"RESUME {resume_kind}: caricato {resume_source} | "
            f"episodio={completed_episodes} | step={total_agent_steps} | "
            f"demo_replay={demo_replay.num_transitions} | "
            f"agent_replay={agent_replay.num_transitions}"
        )
    else:
        print("RESUME: nessun checkpoint trovato, inizializzazione da zero.")

    # ---------------------------------------------------------------
    # INIT DA BEHAVIOR CLONING (solo training nuovo).
    # Il replay demo e' separato e viene mantenuto anche durante i resume.
    # ---------------------------------------------------------------
    initialized_from_bc = False
    if resume_source is None:
        bc_path = Path(cfg.bc_pretrained_weights_path)
        if cfg.initialize_from_bc_if_available and bc_path.exists():
            online.load_weights(bc_path)
            learner.hard_update_target()
            initialized_from_bc = True
            print(f"INIT BC: caricati pesi supervised da {bc_path}")
        elif cfg.initialize_from_bc_if_available:
            print(
                f"INIT BC: {bc_path} non trovato; parto da pesi casuali. "
                "Per usare le demo esegui prima: python pretrain_from_demos.py"
            )

        save_training_origin(cfg, initialized_from_bc)
    else:
        origin = load_training_origin(cfg)
        initialized_from_bc = bool(
            origin.get("initialized_from_behavior_cloning", False)
        )

    epsilon_schedule_start = (
        cfg.epsilon_start_after_bc if initialized_from_bc else cfg.epsilon_start
    )
    print(
        f"EPSILON schedule: start={epsilon_schedule_start:.3f} -> "
        f"end={cfg.epsilon_end:.3f} in {cfg.epsilon_decay_steps} step"
    )
    print(
        "TARGET update: soft Polyak dopo ogni gradient step | "
        f"tau={cfg.target_soft_tau:.6f}"
    )
    print(
        "MIXED REPLAY: quota demo "
        f"{cfg.demo_batch_fraction_start:.2f} -> "
        f"{cfg.demo_batch_fraction_end:.2f} in "
        f"{cfg.demo_batch_fraction_decay_steps} step; "
        f"warm-up agente={cfg.learning_starts} transizioni."
    )
    print(
        "AUX BC loss: sparse cross-entropy sulle sole demo | peso "
        f"{cfg.bc_aux_loss_weight_start:.3f} -> "
        f"{cfg.bc_aux_loss_weight_end:.3f} in "
        f"{cfg.bc_aux_loss_decay_steps} step dopo il warm-up."
    )

    best_checkpoint = BestModelCheckpoint(
        model=online,
        model_path=cfg.best_model_path,
        algorithm="dqn",
        static_metadata={
            "agent": "Double DQN",
            "frame_height": cfg.frame_height,
            "frame_width": cfg.frame_width,
            "stack_size": cfg.stack_size,
            "action_repeat": cfg.action_repeat,
            "preprocessing": "grayscale_bilinear_uint8",
            "network": "conv32-64-64_stride2_dense512_no_bn_dropout0",
            "gamma": cfg.gamma,
            "learning_rate": cfg.learning_rate,
            "target_update_strategy": "soft_polyak_after_gradient_step",
            "target_soft_tau": cfg.target_soft_tau,
            "replay_layout": "protected_demo_plus_agent_v1",
            "demo_batch_fraction_start": cfg.demo_batch_fraction_start,
            "demo_batch_fraction_end": cfg.demo_batch_fraction_end,
            "bc_aux_loss": "sparse_categorical_crossentropy_on_demo_samples",
            "bc_aux_loss_weight_start": cfg.bc_aux_loss_weight_start,
            "bc_aux_loss_weight_end": cfg.bc_aux_loss_weight_end,
            "bc_aux_loss_decay_steps": cfg.bc_aux_loss_decay_steps,
            "bc_aux_exclude_pre_terminal": cfg.bc_aux_exclude_pre_terminal,
            "selection_metric": BEST_SELECTION_METRIC,
            "greedy_eval_every_episodes": cfg.greedy_eval_every_episodes,
            "greedy_eval_episodes": cfg.greedy_eval_episodes,
            "greedy_eval_seed_base": cfg.greedy_eval_seed_base,
        },
    )

    # Un best creato dalla vecchia logica (massimo score di un singolo
    # episodio epsilon-greedy) non deve bloccare le nuove medie reward, che
    # tipicamente sono negative e non confrontabili con quello score.
    existing_best_metadata = {}
    if best_checkpoint.metadata_path.exists():
        try:
            existing_best_metadata = json.loads(
                best_checkpoint.metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_best_metadata = {}
    if existing_best_metadata.get("selection_metric") != BEST_SELECTION_METRIC:
        if best_checkpoint.model_path.exists():
            print(
                "ATTENZIONE: il vecchio dqn_best.keras usa una metrica "
                "incompatibile; verra' sostituito dalla prima greedy eval."
            )
        best_checkpoint.best_score = float("-inf")
        best_checkpoint.best_episode = None

    empty_history = {
        "episode_reward": [],
        "episode_score": [],
        "episode_length": [],
        "loss": [],
        "td_loss": [],
        "bc_loss": [],
        "bc_loss_weight": [],
        "bc_eligible_samples": [],
        "mean_q": [],
        "mean_target": [],
        "grad_norm": [],
        "epsilon": [],
        "demo_batch_fraction": [],
        "agent_step": [],
        "eval_episode": [],
        "eval_agent_step": [],
        "eval_mean_reward": [],
        "eval_std_reward": [],
        "eval_mean_score": [],
        "eval_std_score": [],
        "eval_mean_length": [],
        "best_eval_mean_reward": [],
    }
    if full_resume and full_state_history_path(cfg).exists():
        history = load_history(full_state_history_path(cfg))
    else:
        history = load_history() if resume_source is not None else {}
    for key, default_value in empty_history.items():
        if not isinstance(history.get(key), list):
            history[key] = list(default_value)

    if cfg.greedy_eval_at_start:
        print(
            "Avvio valutazione greedy iniziale "
            f"su {cfg.greedy_eval_episodes} episodi a seed fissi..."
        )
        run_greedy_evaluation(
            env=env,
            model=online,
            cfg=cfg,
            best_checkpoint=best_checkpoint,
            history=history,
            episode=int(completed_episodes),
            total_agent_steps=int(total_agent_steps),
        )
        save_history(history)

    # L'ambiente non viene serializzato: ogni resume riparte da un nuovo
    # episodio. Solo il replay dell'agente continua dal checkpoint.
    reset_result = env.reset(seed=cfg.seed)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    frame = preprocess_frame(obs, cfg.frame_height, cfg.frame_width)
    state = stacker.reset(frame)
    agent_replay.start_episode(frame)

    latest_weights_path = Path(cfg.latest_weights_path)
    latest_weights_path.parent.mkdir(parents=True, exist_ok=True)
    last_completed_episode = int(completed_episodes)

    if total_agent_steps >= cfg.max_agent_steps:
        print(
            f"Il checkpoint è già a step {total_agent_steps}, che raggiunge/supera "
            f"max_agent_steps={cfg.max_agent_steps}. Aumenta max_agent_steps per continuare."
        )

    try:
        for episode in range(completed_episodes + 1, cfg.num_episodes + 1):
            if total_agent_steps >= cfg.max_agent_steps:
                break

            episode_reward = 0.0
            episode_length = 0
            last_info = {}

            while episode_length < cfg.max_steps_per_episode:
                epsilon = linear_epsilon(
                    total_agent_steps,
                    cfg,
                    epsilon_schedule_start,
                )
                action = select_action(online, state, epsilon, num_actions, rng)

                next_obs, reward, terminated, truncated, last_info = repeated_step(
                    env, action, cfg.action_repeat
                )
                episode_length += 1
                total_agent_steps += 1

                time_limit_reached = episode_length >= cfg.max_steps_per_episode
                done = bool(terminated or truncated or time_limit_reached)

                next_frame = preprocess_frame(
                    next_obs,
                    cfg.frame_height,
                    cfg.frame_width,
                )

                # Solo il replay agente riceve le nuove esperienze.
                agent_replay.append(action, reward, done, next_frame)

                next_state = stacker.append(next_frame)
                episode_reward += reward

                demo_fraction = linear_demo_fraction(total_agent_steps, cfg)
                demo_batch_size, agent_batch_size = mixed_batch_sizes(
                    batch_size=cfg.batch_size,
                    demo_fraction=demo_fraction,
                    demo_available=demo_replay.num_transitions,
                )

                can_train = (
                    agent_replay.num_transitions
                    >= max(cfg.learning_starts, agent_batch_size)
                    and demo_replay.num_transitions >= demo_batch_size
                )

                if can_train and total_agent_steps % cfg.train_every == 0:
                    for _ in range(cfg.gradient_steps):
                        (
                            batch,
                            sampled_demo_count,
                            _,
                            sampled_demo_bc_mask,
                        ) = sample_protected_mixed_batch(
                            demo_replay=demo_replay,
                            agent_replay=agent_replay,
                            demo_bc_eligible_mask=demo_bc_eligible_mask,
                            batch_size=cfg.batch_size,
                            demo_fraction=demo_fraction,
                        )
                        bc_loss_weight = linear_bc_aux_weight(
                            total_agent_steps,
                            cfg,
                        )
                        (
                            loss,
                            td_loss,
                            bc_loss,
                            bc_eligible_samples,
                            mean_q,
                            mean_target,
                            grad_norm,
                        ) = learner.train(
                            batch,
                            demo_count=sampled_demo_count,
                            demo_bc_mask=sampled_demo_bc_mask,
                            bc_loss_weight=bc_loss_weight,
                        )
                        history["loss"].append(loss)
                        history["td_loss"].append(td_loss)
                        history["bc_loss"].append(bc_loss)
                        history["bc_loss_weight"].append(bc_loss_weight)
                        history["bc_eligible_samples"].append(
                            bc_eligible_samples
                        )
                        history["mean_q"].append(mean_q)
                        history["mean_target"].append(mean_target)
                        history["grad_norm"].append(grad_norm)

                state = next_state
                if done or total_agent_steps >= cfg.max_agent_steps:
                    break

            score = float(last_info.get("score", 0.0))
            epsilon_now = linear_epsilon(
                total_agent_steps,
                cfg,
                epsilon_schedule_start,
            )
            demo_fraction_now = linear_demo_fraction(total_agent_steps, cfg)

            history["episode_reward"].append(float(episode_reward))
            history["episode_score"].append(score)
            history["episode_length"].append(int(episode_length))
            history["epsilon"].append(float(epsilon_now))
            history["demo_batch_fraction"].append(float(demo_fraction_now))
            history["agent_step"].append(int(total_agent_steps))

            # Il best NON viene piu' aggiornato usando questo episodio di
            # training, perche' contiene esplorazione epsilon-greedy.
            # Viene aggiornato esclusivamente dalla valutazione greedy periodica.
            if episode % cfg.greedy_eval_every_episodes == 0:
                run_greedy_evaluation(
                    env=env,
                    model=online,
                    cfg=cfg,
                    best_checkpoint=best_checkpoint,
                    history=history,
                    episode=episode,
                    total_agent_steps=total_agent_steps,
                )
                save_history(history)

            if episode % cfg.log_every_episodes == 0 or episode == 1:
                window = min(20, len(history["episode_reward"]))
                mean_score = float(np.mean(history["episode_score"][-window:]))
                recent_loss = (
                    float(np.mean(history["loss"][-100:]))
                    if history["loss"]
                    else float("nan")
                )
                recent_td_loss = (
                    float(np.mean(history["td_loss"][-100:]))
                    if history["td_loss"]
                    else float("nan")
                )
                recent_bc_loss = (
                    float(np.mean(history["bc_loss"][-100:]))
                    if history["bc_loss"]
                    else float("nan")
                )
                bc_weight_now = linear_bc_aux_weight(total_agent_steps, cfg)
                recent_bc_eligible = (
                    float(np.mean(history["bc_eligible_samples"][-100:]))
                    if history["bc_eligible_samples"]
                    else float("nan")
                )
                demo_batch_size, agent_batch_size = mixed_batch_sizes(
                    batch_size=cfg.batch_size,
                    demo_fraction=demo_fraction_now,
                    demo_available=demo_replay.num_transitions,
                )
                print(
                    f"Episode {episode:5d} | step {total_agent_steps:7d} | "
                    f"reward {episode_reward:7.2f} | score {score:6.1f} | "
                    f"mean20 score {mean_score:6.2f} | "
                    f"loss T/TD/BC {recent_loss:.4f}/{recent_td_loss:.4f}/"
                    f"{recent_bc_loss:.4f} | bc_w {bc_weight_now:.3f} | "
                    f"bc_n {recent_bc_eligible:.1f} | eps {epsilon_now:.3f} | "
                    f"batch D/A {demo_batch_size:2d}/{agent_batch_size:2d} | "
                    f"demo {demo_replay.num_transitions:6d} | "
                    f"agent {agent_replay.num_transitions:6d} | "
                    f"best eval reward {best_checkpoint.best_score:.4f}"
                )

            last_completed_episode = episode

            if episode % cfg.latest_every_episodes == 0:
                save_latest_checkpoint(
                    online,
                    cfg,
                    last_completed_episode,
                    total_agent_steps,
                )
                save_history(history)
                print(
                    f"Checkpoint latest salvato: {cfg.latest_weights_path} "
                    f"(episodio {last_completed_episode}, step {total_agent_steps})"
                )

            if episode % cfg.full_state_every_episodes == 0:
                save_full_training_state(
                    full_checkpoint_manager,
                    demo_replay,
                    agent_replay,
                    learner,
                    cfg,
                    last_completed_episode,
                    total_agent_steps,
                    rng,
                    history,
                )
                print(
                    f"Checkpoint COMPLETO salvato in {cfg.full_state_dir} "
                    f"(episodio {last_completed_episode}, step {total_agent_steps}, "
                    f"demo {demo_replay.num_transitions}, "
                    f"agent {agent_replay.num_transitions})"
                )

            if total_agent_steps >= cfg.max_agent_steps:
                print("Raggiunto max_agent_steps.")
                break

            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            frame = preprocess_frame(obs, cfg.frame_height, cfg.frame_width)
            state = stacker.reset(frame)
            agent_replay.start_episode(frame)

    finally:
        save_latest_checkpoint(
            online,
            cfg,
            last_completed_episode,
            total_agent_steps,
        )
        save_history(history)
        try:
            save_full_training_state(
                full_checkpoint_manager,
                demo_replay,
                agent_replay,
                learner,
                cfg,
                last_completed_episode,
                total_agent_steps,
                rng,
                history,
            )
            print(
                f"Checkpoint COMPLETO finale salvato in {cfg.full_state_dir} "
                f"(episodio {last_completed_episode}, step {total_agent_steps}, "
                f"demo {demo_replay.num_transitions}, "
                f"agent {agent_replay.num_transitions})"
            )
        finally:
            env.close()

    # Mantiene la firma storica: il replay restituito è quello dell'agente.
    return online, target, agent_replay, history


if __name__ == "__main__":
    cfg = DQNConfig()
    print(json.dumps(asdict(cfg), indent=2))
    train_dqn(cfg)
