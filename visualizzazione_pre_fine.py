import numpy as np
import matplotlib.pyplot as plt

DEMO = "demonstrations/demo_0005.npz"

# -1 = ultima decisione della partita
INDEX = -1

data = np.load(DEMO)

print("Campi nel file:", data.files)

# ============================================================
# NUOVO FORMATO COMPATTO
# ============================================================
if "initial_frame" in data and "next_frames" in data:

    initial_frame = data["initial_frame"]
    next_frames = data["next_frames"]
    actions = data["actions"]

    stack_size = int(data["stack_size"]) if "stack_size" in data else 4

    n = len(actions)

    # Converte indice negativo
    i = INDEX if INDEX >= 0 else n + INDEX

    if i < 0 or i >= n:
        raise IndexError(f"INDEX={INDEX} non valido. La demo ha {n} decisioni.")

    # Stato iniziale: stesso frame ripetuto 4 volte,
    # esattamente come nel DQN.
    state = np.repeat(
        initial_frame[..., None],
        stack_size,
        axis=-1
    )

    # Ricostruiamo lo stato che la rete vede PRIMA della decisione i.
    for k in range(i):
        state = np.concatenate(
            [
                state[..., 1:],
                next_frames[k][..., None]
            ],
            axis=-1
        )

# ============================================================
# VECCHIO FORMATO
# ============================================================
elif "states" in data:

    states = data["states"]
    i = INDEX if INDEX >= 0 else len(states) + INDEX
    state = states[i]

else:
    raise RuntimeError("Formato demo non riconosciuto.")


print("Decisione visualizzata:", i)
print("Shape data alla rete:", state.shape)

if "actions" in data:
    action_names = {
        0: "NOOP",
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }

    action = int(data["actions"][i])
    print("Azione scelta dall'umano:", action_names.get(action, action))


# ============================================================
# VISUALIZZAZIONE DEI 4 FRAME
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(12, 6))

for j in range(state.shape[-1]):
    axes[j].imshow(
        state[:, :, j],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )

    axes[j].set_title(f"t-{state.shape[-1] - 1 - j}")
    axes[j].axis("off")

plt.suptitle(
    f"Input reale della rete - decisione {i}",
    fontsize=14
)

plt.tight_layout()
plt.show()