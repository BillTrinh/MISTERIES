# BNVIP — Sign-to-Prompt / Sign Conversation

Accessibility tool: talk to an AI by **signing** instead of typing. A webcam
reads sign-language gestures, a model turns hand keypoints into words, and a
local LLM cleans them into a sentence **and replies** — a live conversation.
The LLM step also masks recognition errors and ASL-style grammar.

Two tracks:

- **Dynamic word-signs** (main): each gesture = a word; a GRU classifies short
  keypoint sequences; the LLM assembles keywords into a sentence and answers.
- **Static fingerspelling** (earlier MVP): each static hand pose = a letter.

## Setup

Uses the shared MediaPipe venv (Python 3.11) + PyTorch + scikit-learn:

```bash
PY=../hybrid_model/.venv-mp/bin/python   # run from BNVIP/
```

Local LLM (for prompt cleanup + conversation):

```bash
# Ollama installed via `brew install ollama`; service auto-starts.
ollama pull llama3.1:8b     # already pulled
```

Without Ollama the app still runs with fallback replies.

## Dynamic word-sign workflow (main)

1. **Collect** clips per word (~30-50 each; press space to record ~1.3s while
   signing). Suggested vocab: HELLO I WANT COFFEE WHERE SICK TIRED HELP
   ```bash
   $PY src/collect_sequences.py HELLO
   $PY src/collect_sequences.py WANT
   # ... one per word
   ```
2. **Train** the GRU:
   ```bash
   $PY src/train_lstm.py
   ```
3. **Converse**:
   ```bash
   $PY src/app_dynamic.py
   ```
   space = record a sign (adds a word to the turn), r = send the turn (clean +
   reply), b = backspace word, c = clear, n = new conversation, q = quit.

## Static fingerspelling workflow (earlier MVP)

```bash
$PY src/collect_data.py A        # per letter
$PY src/train_classifier.py
$PY src/app.py
```

## Files

| File | Role |
|------|------|
| `src/landmarks.py` | MediaPipe hand tracking (1-hand static / 2-hand dynamic) |
| `src/sequence_utils.py` | Normalize + resample keypoint sequences |
| `src/sign_net.py` | GRU model definition (PyTorch) |
| `src/collect_sequences.py` | Record dynamic word-sign clips |
| `src/train_lstm.py` | Train GRU, report accuracy + latency, save model |
| `src/sign_model.py` | Load GRU, classify a sequence into a word |
| `src/chat.py` | Conversational agent via Ollama (with memory) |
| `src/refine_prompt.py` | Clean raw text into a sentence via Ollama |
| `src/app_dynamic.py` | Real-time sign -> conversation loop + UI |
| `src/collect_data.py`, `train_classifier.py`, `classifier.py`, `text_buffer.py`, `app.py` | Static fingerspelling track |

## Roadmap

- [x] Static fingerspelling MVP
- [x] Dynamic word-sign pipeline (GRU) + LLM conversation
- [ ] Collect data + train on chosen vocab
- [ ] Auto sign segmentation (motion-gated, no key press)
- [ ] Whisper speech -> caption (reverse direction)
- [ ] Evaluation: confusion matrix, per-class accuracy, latency
