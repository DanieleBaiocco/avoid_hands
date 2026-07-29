# Avoid Hands — Behavior Cloning + Double DQN

[English version](README.MD)

Progetto di reinforcement learning basato su un gioco survival realizzato con Pygame e Gymnasium. L’agente impara inizialmente dalle dimostrazioni umane tramite Behavior Cloning, quindi migliora con Double DQN continuando a campionare dati dimostrativi protetti.

![Gameplay di Avoid Hands](./best_model_videos/model-run-episode-13.gif)

## Panoramica

| Voce | Valore predefinito |
|---|---|
| Azioni | `NOOP`, sinistra, destra, su, giù |
| Input | immagini grayscale `220 × 110`, stack di 4 frame |
| Action repeat | 2 step dell’ambiente per decisione dell’agente |
| Frame pooling | massimo pixel per pixel sugli ultimi due frame ripetuti |
| Reward | `+0.001` per ogni step sopravvissuto, `-2.0` alla collisione |
| Dimostrazioni | un file `.npz` compresso per episodio |

```text
Dimostrazioni umane → Behavior Cloning → Double DQN
```

La fase DQN usa due replay buffer fisicamente separati:

- un replay protetto e immutabile, popolato con le dimostrazioni umane;
- un replay circolare contenente esclusivamente le nuove esperienze dell’agente.

Con il batch predefinito di 32 e una quota demo pari a `0.50`, ogni minibatch di training contiene esattamente **16 transizioni demo e 16 transizioni dell’agente**, quando le dimostrazioni sono disponibili.

## Struttura del progetto

```text
.
├── collect_human_demo.py       # registra dimostrazioni umane
├── pretrain_from_demos.py      # pretraining con Behavior Cloning
├── train_dqn.py                # training ibrido Double DQN e resume
├── record_best_run.py          # registra video di valutazione greedy
├── demo_utils.py               # caricamento e filtraggio delle demo
├── rl_common.py                # preprocessing, frame stack e checkpoint
├── demonstrations/
│   ├── train/
│   └── validation/
├── checkpoints/
├── best_model_videos/
└── dont_touch_my_presents/     # ambiente Pygame/Gymnasium
```

I file `.npz`, `.h5` e `.keras` sono gestiti tramite **Git LFS**.

## Installazione

È consigliato Python 3.11. La GPU è facoltativa.

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

Verifica che gli oggetti LFS siano stati scaricati:

```bash
git lfs ls-files
```

Un file che contiene soltanto testo a partire da `version https://git-lfs.github.com/spec/v1` è ancora un puntatore LFS: esegui nuovamente `git lfs pull`.

## Avvio rapido

### Registrare una partita greedy

```bash
python record_best_run.py --episodes 1
```

Lo script carica `dqn_latest.weights.h5` quando disponibile, altrimenti usa `dqn_best.keras`. Per forzare il modello best:

```bash
python record_best_run.py --checkpoint best --episodes 1
```

I video vengono salvati in `best_model_videos/`. Aggiungi `--show-window` per mostrare il gioco durante la registrazione.

### Eseguire il Behavior Cloning

```bash
python pretrain_from_demos.py
```

Output principale:

```text
checkpoints/dqn_bc_pretrained.weights.h5
```

### Avviare o riprendere Double DQN

```bash
python -u train_dqn.py
```

Ordine di priorità del resume:

1. stato completo in `checkpoints/dqn_training_state/`;
2. `checkpoints/dqn_latest.weights.h5`;
3. `checkpoints/dqn_best.keras`, se compatibile;
4. pesi del Behavior Cloning per un nuovo training;
5. inizializzazione casuale se non esiste alcuno stato precedente.

## Pipeline di training

### 1. Behavior Cloning

I cinque output della rete vengono trattati come logits delle azioni e ottimizzati con sparse categorical cross-entropy.

Configurazione predefinita:

- batch size 32;
- Adam con learning rate `1e-4`;
- massimo 40 epoche;
- early stopping con patience 8;
- dropout `0.15`;
- flip orizzontali casuali sui campioni di training, con `LEFT ↔ RIGHT`;
- bilanciamento delle azioni disattivato per impostazione predefinita;
- la validation non viene aumentata né usata per calcolare eventuali pesi di classe.

Per ogni episodio dimostrativo, il dataset di imitation learning esclude:

- la transizione terminale della collisione;
- le 18 transizioni immediatamente precedenti.

Le ultime **19 transizioni** non vengono quindi usate come target del Behavior Cloning. Rimangono invece disponibili per la TD loss del DQN, dove i reward negativi sono informativi.

### 2. Double DQN con dimostrazioni protette

Configurazione predefinita:

- target Double DQN con `gamma=0.99`;
- TD loss Huber sull’intero minibatch misto;
- gradient clipping con norma globale 10;
- learning rate `1e-5`;
- un aggiornamento dei gradienti ogni 4 decisioni dell’agente;
- soft update Polyak della target dopo ogni gradient step, con `tau=0.0005`;
- capacità del replay agente: 50.000 transizioni;
- capacità del replay demo protetto: 65.000 slot, fino a 60.000 transizioni demo;
- inizio del learning dopo 5.000 transizioni generate **dall’agente**;
- batch size 32, diviso di default in **16 demo / 16 agente**.

La loss BC ausiliaria viene applicata soltanto ai campioni demo eleggibili nel batch misto. Il suo peso scende da `0.50` a `0.30` nell’arco di 200.000 step successivi al warm-up. La TD loss continua invece a usare tutte le transizioni demo campionate, comprese quelle terminali e pre-terminali.

Quando la rete viene inizializzata dal Behavior Cloning, epsilon parte da `0.15` e scende a `0.05` in 100.000 step dell’agente. Senza inizializzazione BC parte da `1.0`.

### Valutazione e checkpoint

Il modello best viene scelto tramite una valutazione greedy separata, non usando un singolo episodio epsilon-greedy del training:

- valutazione ogni 50 episodi di training;
- 10 episodi con seed fissi;
- metrica di selezione: reward medio per episodio;
- valutazione greedy iniziale prima di proseguire il training.

Tipi di checkpoint:

```text
checkpoints/dqn_best.keras                 # miglior modello in valutazione greedy
checkpoints/dqn_latest.weights.h5          # ultimi pesi della rete online
checkpoints/dqn_training_state/            # online, target, Adam e replay agente
checkpoints/dqn_history.json               # cronologia di training e valutazione
checkpoints/dqn_training_origin.json       # origine del training e metadata replay
```

Il replay demo protetto viene ricostruito da `demonstrations/train/` a ogni avvio. I checkpoint completi serializzano soltanto il replay modificabile dell’agente, insieme a rete online, target network, optimizer e stato dei generatori casuali.

## Registrazione delle dimostrazioni

Avvia il collector in una sessione grafica:

```bash
python collect_human_demo.py
```

| Tasto | Azione |
|---|---|
| Frecce | Movimento |
| Nessun tasto | `NOOP` |
| `R` | Scarta e riavvia l’episodio corrente |
| `ESC` | Esce |
| `SPACE` | Avvia un altro episodio dopo la morte |

I nuovi episodi vengono inizialmente salvati come:

```text
demonstrations/demo_XXXX.npz
```

Sposta manualmente i file revisionati in:

```text
demonstrations/train/       # training BC e replay DQN protetto
demonstrations/validation/  # sola validation BC
```

Le dimostrazioni compatte contengono frame iniziale, frame successivi, azioni, reward, flag terminali e punteggi. Gli stati completi da quattro frame vengono ricostruiti durante il caricamento.

## Workflow Git LFS

Dopo aver modificato dimostrazioni o checkpoint:

```bash
git add demonstrations checkpoints .gitattributes
git commit -m "Update demonstrations and checkpoints"
git push
```

Su un’altra macchina:

```bash
git pull
git lfs pull
```
