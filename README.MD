# Avoid Hands — Behavior Cloning + Double DQN

[Versione italiana](README.IT.MD)

A reinforcement-learning project built around a Pygame/Gymnasium survival game. The agent first learns from human demonstrations through Behavior Cloning, then improves with Double DQN while continuing to sample protected demonstration data.

![Avoid Hands gameplay](./best_model_videos/model-run-episode-13.gif)

## Overview

| Item | Default |
|---|---|
| Actions | `NOOP`, left, right, up, down |
| Input | `220 × 110` grayscale, stack of 4 frames |
| Action repeat | 2 environment steps per agent decision |
| Frame pooling | Pixel-wise maximum over the last two repeated frames |
| Reward | `+0.001` per survived environment step, `-2.0` on collision |
| Demonstrations | One compressed `.npz` file per episode |

```text
Human demonstrations → Behavior Cloning → Double DQN
```

The DQN stage uses two physically separate replay buffers:

- a protected, immutable replay populated from human demonstrations;
- a circular replay containing only new agent experience.

With the default batch size of 32 and demo fraction `0.50`, every training minibatch contains exactly **16 demonstration transitions and 16 agent transitions** whenever demonstrations are available.

## Project structure

```text
.
├── collect_human_demo.py       # record human demonstrations
├── pretrain_from_demos.py      # Behavior Cloning pretraining
├── train_dqn.py                # hybrid Double DQN training and resume
├── record_best_run.py          # record greedy evaluation videos
├── demo_utils.py               # demonstration loading and filtering
├── rl_common.py                # preprocessing, frame stack and checkpoints
├── demonstrations/
│   ├── train/
│   └── validation/
├── checkpoints/
├── best_model_videos/
└── dont_touch_my_presents/     # Pygame/Gymnasium environment
```

The `.npz`, `.h5`, and `.keras` files are managed through **Git LFS**.

## Installation

Python 3.11 is recommended. A GPU is optional.

```bash
git lfs install
git clone <repository-url>
cd <repository-directory>
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate       # Linux/macOS/Git Bash
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that LFS objects were downloaded:

```bash
git lfs ls-files
```

A file containing only text beginning with `version https://git-lfs.github.com/spec/v1` is still an LFS pointer; run `git lfs pull` again.

## Quick start

### Record a greedy run

```bash
python record_best_run.py --episodes 1
```

The script loads `dqn_latest.weights.h5` when available and otherwise falls back to `dqn_best.keras`. To force the best model:

```bash
python record_best_run.py --checkpoint best --episodes 1
```

Videos are written to `best_model_videos/`. Add `--show-window` to display the game while recording.

### Run Behavior Cloning

```bash
python pretrain_from_demos.py
```

Main output:

```text
checkpoints/dqn_bc_pretrained.weights.h5
```

### Run or resume Double DQN

```bash
python -u train_dqn.py
```

Resume priority:

1. full state in `checkpoints/dqn_training_state/`;
2. `checkpoints/dqn_latest.weights.h5`;
3. compatible `checkpoints/dqn_best.keras`;
4. Behavior Cloning weights for a new run;
5. random initialization if no previous state is available.

## Training pipeline

### 1. Behavior Cloning

The five network outputs are treated as action logits and optimized with sparse categorical cross-entropy.

Default settings:

- batch size 32;
- Adam, learning rate `1e-4`;
- up to 40 epochs;
- early stopping with patience 8;
- dropout `0.15`;
- random horizontal flips on training samples, with `LEFT ↔ RIGHT`;
- action balancing disabled by default;
- validation data is neither augmented nor used to calculate class weights.

For each demonstration episode, the imitation-learning dataset excludes:

- the terminal collision transition;
- the 18 transitions immediately before it.

The final **19 transitions** are therefore excluded from Behavior Cloning targets. They remain available to the DQN TD loss, where their negative rewards are useful.

### 2. Double DQN with protected demonstrations

Default settings:

- Double DQN targets with `gamma=0.99`;
- Huber TD loss on the complete mixed minibatch;
- gradient clipping with global norm 10;
- learning rate `1e-5`;
- one gradient update every 4 agent decisions;
- soft Polyak target update after every gradient step, `tau=0.0005`;
- agent replay capacity: 50,000 transitions;
- protected demo replay capacity: 65,000 slots, up to 60,000 demo transitions;
- learning starts after 5,000 **agent-generated** transitions;
- batch size 32, split **16 demo / 16 agent** by default.

The auxiliary BC loss is applied only to eligible demo samples inside the mixed batch. Its weight decreases from `0.50` to `0.30` over 200,000 post-warm-up steps. The TD loss still uses all sampled demo transitions, including terminal and pre-terminal failures.

When initialized from Behavior Cloning, epsilon starts at `0.15` and decays to `0.05` over 100,000 agent steps. Without BC initialization, it starts at `1.0`.

### Evaluation and checkpoints

The best model is selected using a separate greedy evaluation, not a noisy epsilon-greedy training episode:

- evaluation every 50 training episodes;
- 10 episodes with fixed seeds;
- selection metric: mean episode reward;
- an initial greedy evaluation is run before training continues.

Checkpoint types:

```text
checkpoints/dqn_best.keras                 # best greedy-evaluation model
checkpoints/dqn_latest.weights.h5          # latest online-network weights
checkpoints/dqn_training_state/            # online, target, Adam and agent replay
checkpoints/dqn_history.json               # training and evaluation history
checkpoints/dqn_training_origin.json       # initialization and replay metadata
```

The protected demo replay is rebuilt from `demonstrations/train/` at startup. Full checkpoints serialize only the mutable agent replay, plus the model, target network, optimizer and random-number-generator state.

## Recording demonstrations

Run the collector in a graphical session:

```bash
python collect_human_demo.py
```

| Key | Action |
|---|---|
| Arrow keys | Move |
| No key | `NOOP` |
| `R` | Discard and restart the current episode |
| `ESC` | Quit |
| `SPACE` | Start another episode after death |

New episodes are initially saved as:

```text
demonstrations/demo_XXXX.npz
```

Move reviewed files to:

```text
demonstrations/train/       # BC training and protected DQN replay
demonstrations/validation/  # BC validation only
```

Compact demonstrations contain the initial frame, subsequent frames, actions, rewards, terminal flags and scores. Four-frame states are reconstructed when loading the files.

## Git LFS workflow

After changing demonstrations or checkpoints:

```bash
git add demonstrations checkpoints .gitattributes
git commit -m "Update demonstrations and checkpoints"
git push
```

On another machine:

```bash
git pull
git lfs pull
```
