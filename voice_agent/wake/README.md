# Wake-word — réveiller Reachy avec "Billou"

Reachy n'écoute la pipeline **que** quand il a entendu son mot de réveil.
Évite qu'il réponde à n'importe quel bruit.

Moteur : [openWakeWord](https://github.com/dscripka/openWakeWord) — petit
modèle ONNX qui tourne en continu à faible CPU. Pas de cloud, pas de clé API.

Toutes les commandes ci-dessous se lancent depuis `voice_agent/`. Ce dossier a
son propre venv (`.venv_wake`) pour les outils : score en direct, labellisation,
entraînement du verifier. L'agent, lui, tourne dans `.venv_supertonic`.

## Installation

```bash
cd voice_agent
uv venv .venv_wake --python 3.13
source .venv_wake/bin/activate
uv pip install -r wake/requirements.txt
# Télécharge les modèles pré-entraînés (une fois) :
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Tester tout de suite (modèle pré-entraîné anglais)

Le mot "billou" n'est pas pré-entraîné — il faut l'entraîner
(voir plus bas). Mais on peut valider tout le pipeline immédiatement avec un
mot pré-entraîné comme **"Hey Jarvis"** :

```bash
source .venv_wake/bin/activate

# Détection + capture de la phrase qui suit
python wake/wake_word.py --model hey_jarvis

# Avec le score affiché en continu (pratique pour régler le seuil)
python wake/wake_word.py --model hey_jarvis --meter

# Juste détecter le mot, sans capturer la phrase
python wake/wake_word.py --model hey_jarvis --no-phrase

# Sauver les phrases captées pour les écouter
python wake/wake_word.py --model hey_jarvis --save-dir /tmp/wake_captures
```

Dis **"Hey Jarvis"** → tu verras `🔔 RÉVEILLÉ`. Si tu enchaînes une phrase,
elle est captée. Si tu fais une pause puis parles, elle est aussi captée
(le script attend jusqu'à 1,5 s qu'une parole démarre).

Modèles pré-entraînés disponibles : `hey_jarvis`, `alexa`, `hey_mycroft`,
`hey_rhasspy`, `timer`, `weather`.

## Régler le seuil

```bash
# Plus strict (moins de faux positifs, mais faut bien articuler) :
python wake/wake_word.py --model hey_jarvis --threshold 0.7

# Plus permissif (déclenche plus facilement) :
python wake/wake_word.py --model hey_jarvis --threshold 0.3
```

Lance avec `--meter` et observe le score quand tu parles vs quand il y a du
bruit ambiant. Choisis un seuil entre les deux.

## Entraîner "Billou" (~1h, gratuit)

openWakeWord génère des données synthétiques (des milliers d'exemples du mot
prononcé par différentes voix TTS) puis entraîne un petit classifieur.

1. Ouvre le notebook officiel de training :
   https://github.com/dscripka/openWakeWord
   → `notebooks/automatic_model_training.ipynb` (lien "Open in Colab")

2. Dans le notebook, mets le mot cible : `target_word = "billou"`. Le
   français marche : le générateur utilise des voix multilingues.

3. Lance toutes les cellules (GPU Colab gratuit, ~30–60 min). Ça produit un
   fichier **`billou.onnx`**.

4. Télécharge-le et place-le dans `voice_agent/wake/models/billou.onnx`.

5. Vérifie-le avec le score en direct :

   ```bash
   python wake/wake_word.py --model wake/models/billou.onnx --meter
   ```

> Astuce : commence par valider tout le flux avec `hey_jarvis`. Une fois que
> le comportement te plaît (seuil, capture de phrase), entraîne ton mot et
> remplace juste `--model`.

## Améliorer la détection avec un verifier custom (quelques minutes)

Le modèle de base `.onnx` est entraîné sur des voix synthétiques génériques.
Pour l'adapter à **ta** voix et réduire les faux déclenchements, openWakeWord
permet d'ajouter un petit **verifier** de second étage : quand le modèle de
base se déclenche, le verifier vérifie le clip contre ton profil vocal.

C'est une régression logistique sur les embeddings audio — ça s'entraîne en
quelques secondes, ne demande que `scikit-learn` (pas de GPU, pas de torch),
et ça **réduit les faux positifs** tout en **améliorant la détection de ta
voix**. Tu réutilises directement les clips que l'agent enregistre à l'usage.

### 1. Collecter des clips en utilisant l'agent

Lance l'agent avec `--record-wake DIR` et sers-t'en normalement. Chaque
détection est sauvegardée, et tu peux signaler les ratés depuis le web UI
(bouton « raté » → le clip part dans `misses/`).

```bash
.venv_supertonic/bin/python agent.py --no-robot --wake wake/models/billou.onnx \
    --record-wake ~/reachy_wake_data --webui
```

(`./launch.sh` le fait déjà.) Les clips atterrissent dans :

```
~/reachy_wake_data/detections/   le mot a déclenché (vrai ? ou faux positif ?)
~/reachy_wake_data/nearmiss/     score proche du seuil mais pas déclenché
~/reachy_wake_data/misses/       tu as cliqué « raté » dans le web UI
```

### 2. Labelliser les clips (positif / négatif)

```bash
source .venv_wake/bin/activate

# Interactif : joue chaque clip, tu tapes p (c'est le mot) / n (faux/bruit)
python wake/label_recordings.py /Users/ewann/reachy_wake_data

# Juste voir les compteurs
python wake/label_recordings.py /Users/ewann/reachy_wake_data --stats
```

- **positif** = c'est bien toi qui dis « billou »
- **négatif** = faux déclenchement, bruit, autre parole

Vise ~10+ de chaque pour un bon verifier (ça marche dès 3, en moins bien).

### 3. Exporter les jeux d'entraînement

```bash
python wake/label_recordings.py /Users/ewann/reachy_wake_data --export
```

Ça construit deux dossiers plats :

```
~/reachy_wake_data/_train_positive/
~/reachy_wake_data/_train_negative/
```

### 4. Entraîner le verifier (~secondes)

```bash
python wake/train_verifier.py /Users/ewann/reachy_wake_data
# → wake/models/billou_verifier.joblib

# Pour un autre modèle de base (chemin relatif à wake/) :
python wake/train_verifier.py ~/reachy_wake_data --model models/autre_mot.onnx
```

### 5. L'utiliser

Passe le `.joblib` à l'agent avec `--wake-verifier` (en plus de `--wake`) :

```bash
.venv_supertonic/bin/python agent.py --no-robot --wake wake/models/billou.onnx \
    --wake-verifier wake/models/billou_verifier.joblib --webui
```

> Boucle d'amélioration : continue de tourner avec `--record-wake`, signale les
> ratés, re-labellise les nouveaux clips, ré-exporte et ré-entraîne. Le verifier
> s'affine à chaque passe sans jamais retoucher le modèle `.onnx` de base.

## Dans l'agent

Le wake-word est intégré dans `agent.py` via `--wake`. Tant que le mot n'est
pas entendu, rien d'autre ne tourne. Quand il déclenche, la phrase captée part
vers Whisper → le LM → le TTS, puis Bilou reste à l'écoute pendant
`--conversation-timeout` secondes pour les relances.

```bash
# Lancer l'agent en mode "endormi" jusqu'au mot de réveil
.venv_supertonic/bin/python agent.py --no-robot --wake wake/models/billou.onnx --webui

# Régler la sensibilité
.venv_supertonic/bin/python agent.py --no-robot --wake wake/models/billou.onnx --wake-threshold 0.6
```

`openwakeword` est déjà dans le venv de l'agent (`voice_agent/requirements.txt`) ;
ses modèles de preprocessing se téléchargent une fois par venv :

```bash
.venv_supertonic/bin/python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Fichiers

```
wake/
├── README.md
├── requirements.txt        deps pour .venv_wake (openwakeword + onnxruntime + scikit-learn)
├── wake_word.py            prototype de détection + capture
├── label_recordings.py     labelliser/exporter les clips enregistrés (--record-wake)
├── train_verifier.py       entraîner le verifier custom (logistic regression)
└── models/                 tes .onnx custom + les *_verifier.joblib
```
