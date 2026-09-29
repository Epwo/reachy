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

### Whisper (mlx-community/whisper-large-v3-turbo) ⭐ **RETENU (juin 2026)**

- **Modalités** : ASR uniquement, ~10× temps réel sur M4
- **Premier verdict (mai)** : fonctionne très bien, mais on a d'abord
  préféré un audio-LLM (Gemma 4) qui comprend ET répond en une passe.
- **Pourquoi repris** : l'audio-LLM impose un gros modèle multimodal juste
  pour transcrire ; en passant à Gemma E2B pour gagner en vitesse, la
  compréhension s'est effondrée (voir ci-dessous). Découper en cascade
  STT → LM texte → TTS laisse chaque étape faire son vrai métier.
- **Mesures** : ~0.8 s par phrase (médiane 817 ms en usage réel), transcription
  française quasi parfaite. Pas besoin de ffmpeg : on décode avec soundfile
  et on passe un tableau numpy à mlx-whisper.
- **Piège** : sur du silence pur, Whisper hallucine des génériques de
  sous-titres (« Sous-titrage ST' 501 ») — sans conséquence, la VAD n'envoie
  que de la vraie parole.

### Parakeet TDT 0.6B v3 (mlx-community/parakeet-tdt-0.6b-v3) — option `--stt parakeet` (sept. 2026)

- **Modalités** : ASR NVIDIA, 25 langues européennes dont le français, langue
  détectée automatiquement ; port MLX `parakeet-mlx`. Décodeur non
  autorégressif → très rapide.
- **Mesures** (12 phrases françaises synthétiques bruitées, 2 voix) :
  **0.12 s** par phrase contre 0.82 s pour Whisper, WER 5.2 % contre 4.3 %
  (une seule vraie faute en plus : « pattes » pour « pâtes »).
- **Statut** : option, Whisper reste par défaut en attendant une comparaison
  sur la vraie voix d'Ewann au micro de Reachy (`stats.py` compare les deux).
- **Piège** : comme Whisper, son chargeur de fichiers exige ffmpeg → on lui
  donne directement un tableau 16 kHz.

### Autres ASR évalués (juin 2026, sur papier)

| Modèle | Verdict |
|---|---|
| Voxtral Small 24B | ~14 Go même en 4-bit → ne tient pas en 12 Go |
| Voxtral Mini 3B | très bon français, ports MLX communautaires — meilleure alternative si Whisper déçoit |
| Nemotron / Canary (« 80 ms ») | écosystème NVIDIA NeMo/CUDA. (Parakeet v3, lui, est multilingue et a un port MLX : voir ci-dessus — j'avais tort en juin.) |
| Qwen3-ASR | surtout une API (Alibaba), pas de poids MLX locaux mûrs |
| SenseVoice-Small (FunAudioLLM) | ultra rapide (non autorégressif) mais PyTorch/ONNX, pas MLX ; français un cran sous Whisper |
| Qwen2.5-Omni 7B | audio-LLM trop gros et orienté EN/ZH |
| pyannote speaker-diarization-3.1 | **pas un ASR** : diarisation (qui parle quand), aucun texte |
| VibeVoice (Microsoft) | **pas un ASR** : c'est un TTS long format |

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

### Gemma 4 E4B-it-4bit (mlx-community/gemma-4-e4b-it-4bit) — utilisé mai → juin 2026

- **Modalités** : pareil qu'E2B mais 4B paramètres activés
- **Remplacé** en juin 2026 par la cascade Whisper + LM de chat (section 3).
- **Pourquoi retenu à l'époque** :
  - ~5 Go en mémoire, tient en 12 Go une fois Kyutai TTS retiré
  - Bien meilleur raisonnement que E2B (réelle conversation, pas que
    "Je suis là, je t'écoute")
  - Audio compris correctement en français
  - Support du tool calling (à brancher plus tard)
- **Bugs croisés en chemin** :
  - Bug audio MLX-VLM sur Gemma 4 réglé par PR #931 (`mlx-vlm>=0.4.4`)
  - Faux modèle hallucinated par moi (`gemma-4n`) — pas existe ;
    c'était bien `gemma-4` sans le `n`

### Gemma 4 E2B QAT (mlx-community/gemma-4-E2B-it-qat-4bit) — essai vitesse, abandonné

- **Pourquoi essayé** : ~2× plus rapide que E4B pour réduire la latence.
- **Pourquoi abandonné** : compréhension de la parole nettement moins bonne.
  E2B et E4B partagent le même encodeur audio (E2B est un sous-réseau
  MatFormer d'E4B) : c'est le **modèle de langue** qui interprète l'audio,
  donc un LM plus petit comprend moins bien. C'est ce qui a motivé la cascade.
- **Streaming aussi abandonné** : pour streamer la réponse vers le TTS il
  fallait mettre la réponse *avant* la ligne `[heard]`, et Gemma comprenait
  alors beaucoup moins bien. (Avec la cascade, `[heard]` n'existe plus — le
  streaming redevient possible, voir « Voies futures ».)

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

## 3. Étape 2 — LM de chat (texte + image → réponse)

Depuis la cascade, le LM n'a plus qu'à *écrire* 1–2 phrases (et des lignes
`[tool:…]`) à partir du texte de Whisper.

### Qwen2.5-3B-Instruct (mlx-community/Qwen2.5-3B-Instruct-4bit) — transition

- Texte seul via `mlx-lm` (pas d'encodeurs audio/vision chargés pour rien).
- ~1.6 s par réponse, français naturel, garde le contexte. Bien, mais pas de vision.

### Qwen3.5-4B (mlx-community/Qwen3.5-4B-MLX-4bit) ⭐ **RETENU**

- **VLM** (MoE, ~2.9 Go) via `mlx-vlm >= 0.6.3` : chat en français **et** vision,
  donc Bilou peut décrire ce que voit sa caméra (`[tool:vision]`).
- Raisonnement (`<think>`) **désactivé par défaut** sur le 4B — indispensable
  pour la latence ; on passe quand même `enable_thinking=False` et on retire
  tout bloc `<think>` par sécurité.
- **Mesures (juin)** : ~2.6 s en test isolé, ~6.8 s avec une image ; **médiane
  4.3 s en usage réel** (prompt système + doc des outils + historique).

### Accélérer le même modèle (benchmark sept. 2026) ⭐

Benchmark sur une vraie conversation de 8 tours (prompt système réel + doc des
outils), mlx-vlm 0.7.4 :

| Config | Temps / réponse | Verdict |
|---|---|---|
| 4B tel quel | 3.43 s | ~2.7 s passées à relire les ~1100 tokens de prompt à ~420 tok/s |
| 4B + `PromptCacheState` de mlx-vlm | 3.45 s | aucun effet (voir ci-dessous) |
| **4B + snapshot du prompt système** | **1.17 s** | ⭐ **retenu** — même qualité, moins de mémoire (3.7 vs 4.2 Go) |
| + décodage spéculatif MTP (`Qwen3.5-4B-MTP-4bit`) | 1.25 s | pas de gain : réponses trop courtes (20–40 tokens) |
| + décodage spéculatif DFlash (`z-lab/Qwen3.5-4B-DFlash`) | 1.67 s | plus lent, +2 Go |
| Qwen3.5-2B + snapshot | 0.98 s | ❌ mauvais outils, météo inventée, boucle sur la même phrase |

- **Pourquoi le cache standard ne marche pas** : Qwen3.5 est hybride — 3 couches
  sur 4 sont du Gated DeltaNet (attention linéaire) avec un état récurrent qu'on
  ne peut pas « rembobiner ». mlx-vlm ne réutilise le cache que si le nouveau
  prompt prolonge *exactement* les tokens en cache ; sinon il recalcule tout
  sans rien dire.
- **La solution** : pré-remplir une fois le prompt système (persona + outils,
  ~960 tokens), garder cet état, et en restaurer une copie à chaque tour.
- **Conséquence** : l'heure ne peut plus être dans le prompt système (il
  changerait chaque minute) → étiquette `[heure : …]` à la fin de chaque message.

### Autres candidats écartés

| Modèle | Pourquoi pas |
|---|---|
| Qwen3-4B (texte) | très bon, MLX officiel, thinking désactivable — mais pas de vision |
| Ministral 3 3B (Mistral) | probablement le meilleur français idiomatique, mais VLM sans chemin texte `mlx-lm` propre |
| Gemma 3 4B | VLM aussi, rien de mieux que Qwen3.5 pour nous |
| Llama 3.1 8B | bon français mais 2× plus lent, connaissances 2023 |
| SmolLM3 3B | orienté anglais |
| Qwen3.5-2B | testé (tableau ci-dessus) : trop bête pour les outils |
| K2 Horizon 3.7B (IFM, sept. 2026) | texte seul (perd la vision), cartes centrées anglais, MLX communautaire 4.1 Go |
| Qwen 3.6 / 3.7 / 3.8 | seulement en 27B et plus |
| Qwen3.8-Flash-Next | 125B (+51B d'embeddings), ~70 Go en 4-bit — aperçu de l'archi Qwen4, à surveiller en petites tailles |
| GLM-5.3-Flash | 320B au total (18B actifs) |
| DeepSeek V4.1 Flash | 552B au total (8–16B actifs) |

> **Règle pour les MoE** : seuls quelques experts *calculent* par token, mais
> **tous doivent tenir en mémoire**. Sur 12 Go partagés avec STT + TTS, le
> plafond est ~6–8B de paramètres *au total* en 4-bit. « Flash » = pas cher par
> token sur un serveur, pas petit.

---

## 4. Étape 3 — Texte → Audio (TTS)

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
- **Code supprimé** (sept. 2026) : gardé un temps comme alternative A/B,
  jamais réutilisé.

### Pocket TTS (kyutai/pocket-tts)

- **Modalités** : texte → audio, ~100 M paramètres
- **Pourquoi essayé** : minuscule, CPU-friendly
- **Pourquoi écarté** : **anglais uniquement** (au moment de l'évaluation —
  l'extension à 5 langues a été annoncée mais pas vérifiée pour le français)

### Kokoro-82M (mlx-community/Kokoro-82M-bf16) ✅

- **Modalités** : texte → audio, 8 langues dont français (`ff_siwis`)
- **Pourquoi essayé** : 82 M paramètres, MLX, ultra-rapide
- **Mesure** : **RTF ~0.10 sur M4 base** — bien sub-realtime
- **Statut** : utilisé en TTS par défaut un moment, puis remplacé par
  Supertonic. **Code supprimé** (sept. 2026) pour ne garder qu'un moteur.
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

## 5. Architecture actuelle (sept. 2026)

```
mot de réveil (openWakeWord + verifier)
  → Whisper large-v3-turbo  (parole → texte, .venv_stt)        ~0.8 s
     ou Parakeet v3 (--stt parakeet)                            ~0.12 s
  → Qwen3.5-4B VLM + snapshot du prompt système (.venv_lm)      ~1.2 s
  → Supertonic 3            (texte → parole, .venv_supertonic) ~0.7 s
  → haut-parleur
```

Latence avant que Bilou parle : **~6 s en juin → ~2.7 s avec Whisper, ~2 s
avec Parakeet** (estimation à partir des benchmarks ; à confirmer par `stats.py`).

**Pourquoi ce choix** :
- Whisper = transcription française fiable, indépendante du LM ; Parakeet en
  option pour la vitesse
- Qwen3.5-4B = bon français + vision + outils, sans raisonnement coûteux ;
  aucun modèle plus récent n'est à la fois plus malin et assez petit
- Supertonic 3 = bonne voix, sub-realtime, ONNX (aucun conflit MLX)

## Voies futures à explorer

| Idée | Pourquoi |
|---|---|
| Streamer la réponse du LM phrase par phrase vers le TTS | Bilou commencerait à parler dès la 1re phrase ; sans `[heard]` la cascade n'a plus le problème qui avait fait échouer le streaming avec Gemma |
| Petits modèles Qwen4 quand ils sortiront | Qwen3.8-Flash-Next en préfigure l'architecture ; à re-benchmarker avec `bench_lm` |
| Voxtral Mini 3B à la place de Whisper | si la transcription française déçoit |
| Si M4 Pro / Max disponible | Moshi-MLX pour le naturel du full-duplex |
| Si un omni-modèle français tient en 12 Go en MLX | remplacer toute la cascade par un seul modèle |
| Voix française premium | voix custom Supertonic ou Qwen3-TTS |

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

6. **Dans un audio-LLM, c'est le LM qui « comprend ».** Réduire la taille du
   LM (E4B → E2B) dégrade la compréhension même avec le même encodeur audio.
   Un ASR dédié + un petit LM texte est plus robuste et plus rapide.

7. **Vérifier la fonction d'un modèle, pas seulement son nom.** pyannote fait
   de la diarisation, VibeVoice du TTS : aucun des deux ne transcrit.
