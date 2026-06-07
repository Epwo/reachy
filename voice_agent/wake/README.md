# Wake-word — réveiller Reachy avec "Bilou" / "Reachy"

Prototype isolé : Reachy n'écoute la pipeline LLM **que** quand il a entendu
son mot de réveil. Évite qu'il réponde à n'importe quel bruit.

Moteur : [openWakeWord](https://github.com/dscripka/openWakeWord) — petit
modèle ONNX qui tourne en continu à faible CPU. Pas de cloud, pas de clé API.

## Installation

```bash
cd voice_agent
uv venv .venv_wake
source .venv_wake/bin/activate
uv pip install -r wake/requirements.txt
# Télécharge les modèles pré-entraînés (une fois) :
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Tester tout de suite (modèle pré-entraîné anglais)

Les mots "bilou"/"reachy" ne sont pas pré-entraînés — il faut les entraîner
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

## Entraîner "Bilou" / "Reachy" (~1h, gratuit)

openWakeWord génère des données synthétiques (des milliers d'exemples du mot
prononcé par différentes voix TTS) puis entraîne un petit classifieur.

1. Ouvre le notebook officiel de training :
   https://github.com/dscripka/openWakeWord
   → `notebooks/automatic_model_training.ipynb` (lien "Open in Colab")

2. Dans le notebook, mets le mot cible, par ex. `target_word = "bilou"`
   (et un second run pour `"reachy"`). Le français marche : le générateur
   utilise des voix multilingues.

3. Lance toutes les cellules (GPU Colab gratuit, ~30–60 min). Ça produit un
   fichier **`bilou.onnx`**.

4. Télécharge-le et place-le ici :
   ```
   voice_agent/wake/models/bilou.onnx
   voice_agent/wake/models/reachy.onnx
   ```

5. Utilise-le :
   ```bash
   python wake/wake_word.py --model wake/models/bilou.onnx --threshold 0.5
   ```

   Pour écouter les deux mots en même temps, on adaptera le script pour
   passer plusieurs modèles (`wakeword_models=[bilou, reachy]`) — dis-le
   moi quand tu auras les .onnx.

> Astuce : commence par valider tout le flux avec `hey_jarvis`. Une fois que
> le comportement te plaît (seuil, capture de phrase), entraîne tes mots et
> remplace juste `--model`.

## Brancher sur la pipeline (FAIT)

Le wake-word est intégré dans `agent.py` via `--wake`. Tant que le mot n'est
pas entendu, le LM et le TTS ne tournent pas. Quand il déclenche, la phrase
captée part vers Gemma puis le TTS.

```bash
# Depuis le venv de l'agent (.venv_supertonic par défaut)
source ../.venv_supertonic/bin/activate

# Une fois par venv : télécharger les modèles de preprocessing openWakeWord
python -c "import openwakeword; openwakeword.utils.download_models()"

# Lancer l'agent en mode "endormi" jusqu'au mot de réveil
python ../agent.py --no-robot --wake wake/models/billou.onnx --webui

# Régler la sensibilité
python ../agent.py --no-robot --wake wake/models/billou.onnx --wake-threshold 0.6
```

Pré-requis : `openwakeword` doit être installé dans le venv qui lance
`agent.py` (pur ONNX, pas de conflit avec Supertonic/Kokoro) :

```bash
uv pip install openwakeword
```

## Fichiers

```
wake/
├── README.md
├── requirements.txt    deps pour .venv_wake (openwakeword + onnxruntime)
├── wake_word.py        prototype de détection + capture
└── models/             (tes .onnx custom une fois entraînés)
```
