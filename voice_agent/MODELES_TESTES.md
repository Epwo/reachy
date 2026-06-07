# Modèles explorés pour le voice agent

Ce document trace l'historique des modèles qu'on a évalués pour le pipeline
voix de Reachy Mini, et pourquoi chacun ne convenait pas (ou conviendrait
plus tard). Servir de mémoire pour ne pas refaire les mêmes essais.

## Contexte / contraintes

- **Hardware** : Mac mini M4 base — 10 cœurs GPU, **12 Go de RAM unifiée**
- **Cible** : conversation temps réel en français avec Reachy Mini
- **Préférence forte** : tout local, pas de cloud (sauf fallback)
- **Performance attendue** : RTF < 1 pour TTS, ~2 s de latence par tour total
- **Accélération MLX/Metal** souhaitée pour bénéficier du GPU
- **Tool calling** souhaité à terme pour piloter le robot

---

## 1. Rêve initial : modèles end-to-end speech-to-speech

L'idée de départ : un seul modèle qui prend l'audio en entrée et sort de
l'audio en sortie, comme dans la démo originale de Voxtral. **Aucun n'a
fonctionné sur le Mac M4 base.**

### Voxtral Mini 3B (mistralai/Voxtral-Mini-3B-2507)

- **Modalités** : audio in → texte out (et pas audio out malgré le nom)
- **Pourquoi essayé** : openAI-compatible, format Mistral, support cloud + local
- **Pourquoi rejeté** :
  - vLLM ne supporte pas le backend Metal sur Mac → CPU only → **5–15 s par tour**
  - Pas de portage MLX existant
  - Première erreur : `ImportError: cannot import name 'AudioChunk'`
    (résolue avec `mistral_common[audio]>=1.5.4`)
- **Verdict** : utilisable uniquement via le cloud Mistral, pas en local Mac

### Voxtral Mini 4B Realtime (mistralai/Voxtral-Mini-4B-Realtime-2602)

- **Modalités** : ASR streaming temps réel (audio → texte uniquement)
- **Pourquoi essayé** : "realtime" semblait prometteur
- **Pourquoi rejeté** : c'est de l'**ASR streaming**, pas du audio-to-audio.
  Le mot "realtime" désigne la latence de transcription (<500 ms), pas la
  génération de voix. Mistral a un Voxtral-TTS séparé pour la sortie audio.

### PersonaPlex 7B (nvidia/personaplex-7b-v1)

- **Modalités** : speech-to-speech complet, full-duplex, basé sur Moshi
- **Pourquoi essayé** : 438k téléchargements, M-series mentionné
- **Pourquoi rejeté** :
  - Requiert NVIDIA Ampere/Hopper GPU + Linux (pas Mac)
  - **Anglais uniquement** — pas de français
  - Pas de tool calling natif

### Moshi / Moshika (kyutai/moshika-mlx-q4)

- **Modalités** : speech-to-speech full-duplex
- **Pourquoi essayé** : `moshi-mlx` est un vrai package PyPI, MLX, fonctionne
  sur Apple Silicon
- **Pourquoi rejeté** :
  - **Anglais uniquement** (confirmé sur la model card)
  - **Trop lourd pour M4 base** : Kyutai annonce RTF ~0.87 sur **M2 Max**
    (38 cœurs GPU). M4 base = 10 cœurs GPU = ~25 % de cette puissance.
    Mesure réelle : RTF ~3, lag cumulé qui grossit sans fin
  - Crash avec `reached max-steps 4005` après quelques minutes
  - Pas de tool calling natif (hack possible via parsing du texte)

### MoshiVis (kyutai/moshika-vis-mlx)

- **Modalités** : audio + image → audio (vision en bonus !)
- **Pourquoi essayé** : MLX disponible, donnerait à Reachy une vraie vision
- **Pourquoi rejeté** :
  - Même problème de perf que Moshi (architecture identique)
  - Repo séparé (`kyutai-labs/moshivis`), pas dans le package `moshi_mlx`
    standard, install plus complexe (`kyuteye_mlx` server, `uv sync`, etc.)
  - Anglais uniquement

### Qwen3-Omni (Qwen/Qwen3-Omni)

- **Modalités** : audio + texte + image + vidéo → audio + texte, multilingue
  (français inclus), tool calling natif — **le rêve sur papier**
- **Pourquoi essayé** : meilleure intégration vue (au moins en théorie)
- **Pourquoi rejeté** :
  - **Aucun portage MLX disponible** pour les variantes 3B/7B (j'ai
    inventé `mlx-community/Qwen3-Omni-7B-Instruct-MLX-4bit` qui n'existe
    pas — leçon apprise)
  - Le seul portage existant (`pherber3/Qwen3-Omni-30B-A3B-Instruct-4bit-mlx`)
    est **text-out only**, l'encodeur/décodeur audio n'est pas implémenté
  - Trop gros (30B) pour le M4 12 Go de toute façon

### Qwen2.5-Omni

- **Modalités** : audio + vision + texte → audio + texte
- **Pourquoi essayé** : alternative plus mature à Qwen3-Omni
- **Pourquoi rejeté** : voix de sortie en chinois et anglais uniquement ;
  parle français phonétiquement mais avec un accent étrange

### LiquidAI LFM2.5-Audio-1.5B

- **Modalités** : audio in + audio out, 1.5B
- **Pourquoi essayé** : très petit (1.5B), pourrait tenir sur M4
- **Pourquoi rejeté** : tool calling faible, support français incertain,
  abandonné après mieux trouvé ailleurs

### Mini-Omni 2

- **Modalités** : speech-to-speech, 0.5B
- **Pourquoi rejeté** : démo-grade, pas de tool calling, pas de français

### Step-Audio, Fish-Agent, GLM-4-Voice, Hibiki

- Pas testés en détail : tous présentent au moins un blocant majeur
  (anglais only, pas de MLX, trop gros, ou pas de support français
  pour la sortie audio)

### Conclusion sur l'end-to-end

**Aucun modèle speech-to-speech ne coche toutes les cases simultanément** :
- français en entrée ET sortie
- tournant localement sur M4 base 12 Go
- avec tool calling

Pivot stratégique : **pipeline découpé** (audio→texte) puis (texte→audio)
avec deux modèles spécialisés. C'est ce que Mistral recommande d'ailleurs
dans leur propre architecture "speech-to-speech assistant".

---

## 2. Étape 1 — Audio → Texte (compréhension)

### Whisper (mlx-community/whisper-large-v3-turbo)

- **Modalités** : ASR uniquement, ~10× temps réel sur M4
- **Pourquoi essayé** : référence de l'ASR, MLX disponible
- **Verdict** : **fonctionne très bien** mais nécessite un LLM séparé
  derrière pour le raisonnement. Décision : préférer un modèle multimodal
  qui fait audio → texte ET raisonne dans la même passe.

### Gemma 3n E2B-it (mlx-community/gemma-3n-E2B-it-4bit)

- **Modalités** : audio + image + texte → texte, multilingue (140+ langues)
- **Pourquoi essayé** : design "mobile-first", supporté par mlx-vlm
- **Pourquoi écarté pour Gemma 4** : sortie en avril 2025 ;
  Gemma 4 est mieux notée et plus récente

### Gemma 4 E2B-it-4bit (mlx-community/gemma-4-e2b-it-4bit) ✅

- **Modalités** : audio + image + texte → texte, 140+ langues
- **Pourquoi essayé** : 2B activé, ~3 Go, tient à l'aise en 12 Go
- **Statut** : utilisé en intermédiaire avant le passage à E4B.
  Bonne qualité mais déflectif sur les questions ouvertes.

### Gemma 4 E4B-it-4bit (mlx-community/gemma-4-e4b-it-4bit) ✅ **RETENU**

- **Modalités** : pareil qu'E2B mais 4B paramètres activés
- **Pourquoi retenu** :
  - ~5 Go en mémoire, tient en 12 Go une fois Kyutai TTS retiré
  - Bien meilleur raisonnement que E2B (réelle conversation, pas que
    "Je suis là, je t'écoute")
  - Audio compris correctement en français
  - Support du tool calling (à brancher plus tard)
- **Bugs croisés en chemin** :
  - Bug audio MLX-VLM sur Gemma 4 réglé par PR #931 (`mlx-vlm>=0.4.4`)
  - Faux modèle hallucinated par moi (`gemma-4n`) — pas existe ;
    c'était bien `gemma-4` sans le `n`

### Phi-4-Multimodal

- **Modalités** : audio + image + texte → texte, ~5.6 B
- **Pourquoi écarté** : plus gros que Gemma 4 E4B sans gain de qualité
  notable pour notre cas, install plus complexe

### MiniCPM-o 2.6

- **Modalités** : audio + vision → texte, ~8 B
- **Pourquoi écarté** : trop gros pour 12 Go avec un TTS chargé en plus

### Qwen2-Audio-7B-Instruct

- **Modalités** : audio + texte → texte
- **Pourquoi écarté** : support MLX peu mature au moment où on a regardé,
  Gemma 4 plus avancé sur le marché

---

## 3. Étape 2 — Texte → Audio (TTS)

### macOS `say` (intégré)

- **Modalités** : texte → audio, voix françaises (Thomas, Jacques, Amélie...)
- **Pourquoi essayé** : zéro install, instantané
- **Pourquoi écarté** : voix robotique des années 90 ; OK comme fallback
  mais inacceptable pour vraie conversation

### Voxtral-TTS (Mistral cloud)

- **Modalités** : texte → audio, multilingue
- **Pourquoi essayé** : ~70 ms latence modèle, qualité excellente
- **Pourquoi écarté** : cloud only, et l'utilisateur veut tout local

### Kyutai TTS 1.6B (kyutai/tts-1.6b-en_fr)

- **Modalités** : texte → audio, bilingue anglais + français
- **Pourquoi essayé** : streaming, qualité expressive, MLX via `moshi_mlx`
- **Pourquoi écarté pour la pipeline par défaut** :
  - **Trop lent sur M4 base** : RTF ~6 en bf16, RTF ~1.5 en q8
    (alors qu'il faut < 1 pour temps réel)
  - Quantization buggée dans moshi_mlx 0.3.0 (matmul shape mismatch
    dans cross-attention)
  - 3.6 Go en bf16 — combiné à Gemma E4B (5 Go), on était en swap permanent
- **Conservé** comme alternative A/B (`test_tts.py --engine kyutai`)

### Pocket TTS (kyutai/pocket-tts)

- **Modalités** : texte → audio, ~100 M paramètres
- **Pourquoi essayé** : minuscule, CPU-friendly
- **Pourquoi écarté** : **anglais uniquement** (au moment de l'évaluation —
  l'extension à 5 langues a été annoncée mais pas vérifiée pour le français)

### Kokoro-82M (mlx-community/Kokoro-82M-bf16) ✅

- **Modalités** : texte → audio, 8 langues dont français (`ff_siwis`)
- **Pourquoi essayé** : 82 M paramètres, MLX, ultra-rapide
- **Mesure** : **RTF ~0.10 sur M4 base** — bien sub-realtime
- **Statut** : utilisé en TTS par défaut un moment, **conservé comme
  alternative**
- **Bugs en chemin** : chaîne de dépendances pénible (`mlx-audio` →
  `misaki` → `phonemizer` → `espeak-ng` → `espeakng-loader`).
  Plusieurs `ImportError` consécutifs avant que tout fonctionne.

### Supertonic 3 (Supertone/supertonic-3) ⭐ **RETENU PAR DÉFAUT**

- **Modalités** : texte → audio, **31 langues**, expressions
  (`<laugh>`, `<breath>`, `<sigh>`), 99 M paramètres
- **Pourquoi essayé** : annoncé 167× temps réel sur M4 Pro, dep chain
  beaucoup plus propre (`pip install supertonic`, pas d'espeak)
- **Mesure** : **RTF ~0.25 sur M4 base** avec `total_steps=11`
  (en bf16, pas de Metal). Légèrement plus lent que Kokoro mais
  fait largement le job sub-realtime.
- **Pourquoi retenu** :
  - **Pas de conflit MLX** (utilise ONNX Runtime) — futur merge possible
    dans le venv LM
  - Plus de langues
  - Tags d'expression utiles plus tard
- **Limitations** :
  - Voix par défaut limitées (M1-M5, F1-F5)
  - Voix custom payantes via Voice Builder

### Qwen3-TTS-MLX (suckerfish/qwen3-tts-mlx)

- **Modalités** : texte → audio, 10 langues dont français
- **Pourquoi pas testé en profondeur** : portage communautaire, qualité
  supérieure à Kokoro probable mais plus lent ; Supertonic gagnait sur
  la simplicité d'install. À évaluer si on a besoin de meilleure qualité
  française.

### F5-TTS, Marvis TTS, Chatterbox, Dia

- Modèles cités dans l'écosystème `mlx-audio`. Pas testés faute de besoin
  après Supertonic. À reprendre si on veut explorer le voice cloning
  (Dia est dialogue-focused, Chatterbox supporte 16 langues).

---

## 4. Architecture finale (à ce stade)

```
mic → VAD → Gemma 4 E4B (audio→texte, .venv_lm)
        → Supertonic 3 (texte→audio, .venv_supertonic)
        → speaker
```

**Pourquoi ce choix au final** :
- Gemma 4 E4B = meilleur ratio qualité / mémoire / vitesse pour 12 Go
- Supertonic 3 = bonne qualité de voix + sub-realtime + dep chain propre
- Kokoro reste branchable (`--tts kokoro`) si on veut RTF encore plus bas
- Kyutai reste branchable (`--tts kyutai`) si on veut tester la qualité

## Voies futures à explorer

| Quand | Modèle à tester |
|---|---|
| Si M4 Pro / Max disponible | Moshi-MLX pour le "natural" de la conversation full-duplex |
| Si Qwen3-Omni MLX sort en taille 3B/7B | Replacer Gemma + TTS par un seul modèle |
| Si on veut une voix française premium | Acheter une voix custom Supertonic ou évaluer Qwen3-TTS |
| Si tool calling devient prioritaire | Tester Phi-4-Multimodal qui a un meilleur support outils |

## Leçons apprises

1. **"Run on Apple Silicon" ne veut pas dire vite sur n'importe quel M-chip.**
   La plupart des benchmarks sont faits sur M2 Max / M4 Pro. M4 base a
   25 % de la puissance GPU d'un M2 Max — RTF triplé pour le même modèle.

2. **vLLM ≠ Metal sur Mac.** vLLM utilise le CPU. Pour avoir le GPU, il
   faut MLX (ou Ollama / llama.cpp avec Metal, ou MLC LLM).

3. **Les conflits de pin `mlx` cassent tout.** `mlx-audio` (≥0.31) vs
   `moshi_mlx` (<0.27) sont incompatibles. Solution : un venv par modèle,
   communication via JSON-over-pipes.

4. **End-to-end speech-to-speech multilingue local n'existe pas encore
   en mai 2026** (pour M4 base avec français). Pipeline découpé pour
   l'instant.

5. **Vérifier les modèles sur HuggingFace AVANT de les recommander.**
   J'ai halluciné plusieurs noms (`mlx-community/Qwen3-Omni-7B-Instruct-MLX-4bit`,
   `gemma-4n`) — leçon retenue : `WebSearch` + `WebFetch` la première fois
   qu'on cite un repo précis.
