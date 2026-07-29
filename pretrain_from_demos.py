#!/usr/bin/env python3
"""Behavior Cloning sulle demo umane, prima del training DQN.

Uso:
    python pretrain_from_demos.py

Input:
    demonstrations/demo_*.npz

Output:
    checkpoints/dqn_bc_pretrained.weights.h5

La rete e' ESATTAMENTE la stessa build_q_network usata dal DQN. Durante questa
fase i 5 output vengono trattati come logits per classificare l'azione umana.
Nel DQN gli stessi 5 output verranno poi adattati a Q(s,a).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from demo_utils import (
    compute_action_weights,
    make_demo_batch_generator,
    summarize_demos,
)
from train_dqn import DQNConfig, build_q_network


@dataclass
class BCConfig:
    train_demo_dir: str = "demonstrations/train"
    validation_demo_dir: str = "demonstrations/validation"
    output_weights: str = "checkpoints/dqn_bc_pretrained.weights.h5"
    metadata_path: str = "checkpoints/dqn_bc_pretrained.json"

    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 8

    # Mescola singole transizioni provenienti da episodi diversi. 512 stati
    # uint8 occupano circa 50 MB con input 220x110x4.
    shuffle_buffer: int = 1024
    dropout_rate: float = 0.15

    # Pesi moderati per compensare dataset sbilanciati (es. molto NOOP).
    balance_actions: bool = False
def augment_batch(states, actions, sample_weights):
    """
    Flip orizzontale casuale del 50% dei campioni.

    states:
        [batch, height, width, stack_size]

    IMPORTANTE:
    tutti e 4 i frame dello stack vengono flippati insieme.

    Mapping azioni:
        0 NOOP  -> 0 NOOP
        1 LEFT  -> 2 RIGHT
        2 RIGHT -> 1 LEFT
        3 UP    -> 3 UP
        4 DOWN  -> 4 DOWN
    """

    batch_size = tf.shape(states)[0]

    # Decide indipendentemente per ogni elemento del batch
    flip_mask = tf.random.uniform(
        shape=(batch_size,),
        minval=0.0,
        maxval=1.0,
    ) < 0.5

    # Flip sulla dimensione WIDTH.
    # states ha shape:
    # [batch, height, width, stack]
    flipped_states = tf.reverse(states, axis=[2])

    # Applica il flip solo ai campioni selezionati
    states = tf.where(
        flip_mask[:, None, None, None],
        flipped_states,
        states,
    )

    # LEFT <-> RIGHT
    action_map = tf.constant(
        [0, 2, 1, 3, 4],
        dtype=tf.int32,
    )

    flipped_actions = tf.gather(action_map, actions)

    actions = tf.where(
        flip_mask,
        flipped_actions,
        actions,
    )

    return states, actions, sample_weights

def make_tf_dataset(
    files,
    cfg: BCConfig,
    dqn_cfg: DQNConfig,
    action_weights,
    shuffle: bool,
):
    generator_factory = make_demo_batch_generator(
        files=files,
        batch_size=cfg.batch_size,
        frame_height=dqn_cfg.frame_height,
        frame_width=dqn_cfg.frame_width,
        stack_size=dqn_cfg.stack_size,
        action_repeat=dqn_cfg.action_repeat,
        action_weights=action_weights,
        shuffle=shuffle,
        seed=cfg.seed + (0 if shuffle else 10_000),
        exclude_pre_terminal=18,
    )

    output_signature = (
        tf.TensorSpec(
            shape=(None, dqn_cfg.frame_height, dqn_cfg.frame_width, dqn_cfg.stack_size),
            dtype=tf.uint8,
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        generator_factory,
        output_signature=output_signature,
    )

    # Il generatore compatto produce batch temporalmente consecutivi. Li
    # spacchettiamo, mescoliamo con un buffer limitato e ricreiamo i batch:
    # cosi' ogni batch contiene stati meno correlati e le statistiche di train
    # sono molto piu' rappresentative.
    dataset = dataset.unbatch()
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=cfg.shuffle_buffer,
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(cfg.batch_size, drop_remainder=shuffle)

    # La validation NON viene augmentata.
    if shuffle:
        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def pretrain_from_demos(cfg: BCConfig) -> keras.Model:
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    dqn_cfg = DQNConfig()
    num_actions = 5

    # ------------------------------------------------------------
    # TRAIN DEMOS
    # ------------------------------------------------------------
    train_summary = summarize_demos(
        cfg.train_demo_dir,
        num_actions=num_actions,
        frame_height=dqn_cfg.frame_height,
        frame_width=dqn_cfg.frame_width,
        stack_size=dqn_cfg.stack_size,
        action_repeat=dqn_cfg.action_repeat,
    )

    # ------------------------------------------------------------
    # VALIDATION DEMOS
    # ------------------------------------------------------------
    val_summary = summarize_demos(
        cfg.validation_demo_dir,
        num_actions=num_actions,
        frame_height=dqn_cfg.frame_height,
        frame_width=dqn_cfg.frame_width,
        stack_size=dqn_cfg.stack_size,
        action_repeat=dqn_cfg.action_repeat,
    )

    train_files = train_summary.files
    val_files = val_summary.files

    # IMPORTANTISSIMO:
    # i pesi delle classi vengono calcolati SOLO dal TRAIN,
    # senza usare informazioni della validation.
    action_weights = (
        compute_action_weights(train_summary.action_counts)
        if cfg.balance_actions
        else np.ones(num_actions, dtype=np.float32)
    )

    print(f"Demo TRAIN: {len(train_files)}")
    print(f"Transizioni TRAIN: {train_summary.transitions}")

    print(f"Demo VALIDATION: {len(val_files)}")
    print(f"Transizioni VALIDATION: {val_summary.transitions}")

    print(
        "Azioni TRAIN [NOOP, LEFT, RIGHT, UP, DOWN]: "
        f"{train_summary.action_counts.tolist()}"
    )

    print(
        "Azioni VALIDATION [NOOP, LEFT, RIGHT, UP, DOWN]: "
        f"{val_summary.action_counts.tolist()}"
    )

    print(f"Pesi azioni TRAIN: {[round(float(x), 3) for x in action_weights]}")
    input_shape = (
        dqn_cfg.frame_height,
        dqn_cfg.frame_width,
        dqn_cfg.stack_size,
    )
    model = build_q_network(
        input_shape,
        num_actions,
        dropout_rate=cfg.dropout_rate,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
            epsilon=1e-7,
            clipnorm=dqn_cfg.max_grad_norm,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="action_accuracy")],
    )

    train_ds = make_tf_dataset(
        train_files,
        cfg,
        dqn_cfg,
        action_weights,
        shuffle=True,
    )
    val_ds = (
        # La loss di validation resta non pesata: in questo modo il monitor
        # misura la generalizzazione reale, non la funzione obiettivo corretta
        # con i pesi calcolati sul train.
        make_tf_dataset(val_files, cfg, dqn_cfg, None, shuffle=False)
        if val_files
        else None
    )

    output_path = Path(cfg.output_weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.TerminateOnNaN(),
    ]
    monitor = "val_loss" if val_ds is not None else "loss"
    callbacks.extend(
        [
            keras.callbacks.ModelCheckpoint(
                filepath=str(output_path),
                monitor=monitor,
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=1,
            ),
        ]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint ha gia' salvato il best; questa save garantisce che il
    # file esista anche in configurazioni/callback anomale.
    model.save_weights(output_path)

    metadata = {
        "stage": "behavior_cloning",
        "bc_config": asdict(cfg),
        "dqn_input_shape": list(input_shape),
        "num_actions": num_actions,
        "train_demo_files": [p.name for p in train_files],
        "validation_demo_files": [p.name for p in val_files],

        "train_transitions": int(train_summary.transitions),
        "validation_transitions": int(val_summary.transitions),

        "train_action_counts": train_summary.action_counts.tolist(),
        "validation_action_counts": val_summary.action_counts.tolist(),
        "action_weights": action_weights.tolist(),
        "final_metrics": {
            key: float(values[-1])
            for key, values in history.history.items()
            if values
        },
    }
    metadata_path = Path(cfg.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nPretraining completato. Pesi salvati in: {output_path}")
    print("Ora puoi avviare: python -u train_dqn.py")
    return model


if __name__ == "__main__":
    cfg = BCConfig()
    print(json.dumps(asdict(cfg), indent=2))
    pretrain_from_demos(cfg)
