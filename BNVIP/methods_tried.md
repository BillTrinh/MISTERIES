# Continuous Sign Recognition — Methods Tried

Notes on every recognition approach explored for multi-word / continuous signing in BNVIP.  
Shared data: `data/signs/*.pkl` (~40–50 clips per word, 8 words). Shared hand features: MediaPipe two-hand keypoints (126-D per frame).

---

## 1. Isolated GRU (baseline)

| | |
|---|---|
| **Entry** | `python src/app_dynamic.py` |
| **Core** | `sign_model.py` + `models/sign_gru.pt` |
| **Train** | `python src/train_lstm.py` from `data/signs/*.pkl` |
| **Idea** | User presses space → record ~40 frames → one word |
| **Similarity?** | No. Softmax class confidence from a small GRU |
| **Uses `.pkl` at inference?** | No (baked into checkpoint) |
| **Pros** | Fast, simple UX for one sign at a time |
| **Cons** | Not continuous; user must segment each word |

**Architecture:** `SignGRU` — `GRU(126 → 128, 1 layer)` → Dropout → Linear → 8 classes. Input resampled to `T=32`.

---

## 2b. DTW + cosine local cost (continuous)

| | |
|---|---|
| **Entry** | `python src/app_dtw_cosine.py` |
| **Core** | `dtw_match.py` with `metric="cosine"` |
| **Idea** | Same sliding-window DTW as #2, but each frame pair costs `1 - cosine` instead of L2 |
| **Similarity?** | Yes — DTW path over cosine distances → `exp(-dist / 0.35)` |
| **Uses `.pkl` at inference?** | Yes |
| **Pros** | Nonlinear alignment + direction-based frame match |
| **Cons** | Still slower than Fast/FAST+; threshold may need tuning vs L2 |

`app_dtw_continuous.py` keeps the original **L2** DTW (`metric="l2"` default).

---

## 3. Fast resample + cosine (continuous)

| | |
|---|---|
| **Entry** | `python src/app_fast_continuous.py` |
| **Core** | `fast_match.py` |
| **Idea** | Normalize + resample window/template to fixed length; cosine similarity; multi-scale windows |
| **Similarity?** | Yes — cosine on flattened sequences |
| **Uses `.pkl` at inference?** | Yes |
| **Pros** | Much faster than DTW; no neural net needed |
| **Cons** | Alignment is only linear stretch (no nonlinear path); can duplicate / miss words |

---

## 4. Sliding-window GRU (continuous)

| | |
|---|---|
| **Entry** | `python src/app_gru_continuous.py` |
| **Core** | `gru_match.py` |
| **Idea** | Same `sign_gru.pt`; slide multi-scale windows; classify each window |
| **Similarity?** | No — class confidence |
| **Uses `.pkl` at inference?** | No |
| **Pros** | Very fast (~0.07 s in smoke tests) |
| **Cons** | Duplicates and false positives; less precise than templates |

---

## 5. Stable sliding-window GRU (continuous)

| | |
|---|---|
| **Entry** | `python src/app_gru_stable.py` |
| **Core** | `gru_match_stable.py` |
| **Idea** | Same GRU, but stricter decoding: higher confidence (`~0.72`), fewer scales, motion gate, peak picking, same-word merge |
| **Similarity?** | No — class confidence |
| **Uses `.pkl` at inference?** | No |
| **Pros** | Still fast; fewer duplicates than raw GRU windows |
| **Cons** | Still a classifier, not template similarity; accuracy limited by training data size |

---

## 6. FAST+ (continuous, current similarity-focused build)

| | |
|---|---|
| **Entry** | `python src/app_fast_plus.py` |
| **Core** | `fast_match_plus.py` |
| **Idea** | Like Fast (#3), plus motion similarity and small lag search (±frames), with peak picking / merge |
| **Similarity?** | Yes — `0.65 * pose_cosine + 0.35 * motion` |
| **Uses `.pkl` at inference?** | Yes |
| **Pros** | Template-based; faster than DTW; motion helps beyond pure pose cosine |
| **Cons** | Still not full dance-style joint-angle scoring; lag search ≠ full DTW |

**Score breakdown printed live:** `pose`, `motion`, `lag`.

---

## 7. Dance-style scoring on FAST+ (tried, then reverted)

| | |
|---|---|
| **Status** | **Reverted** — no longer in the tree |
| **Idea** | Replace cosine with dance-scorer-style metrics: weighted landmark distance Gaussian, finger/palm joint angles, landmark motion direction/magnitude |
| **Why tried** | User preferred dance-like similarity over cosine |
| **Result** | Correct self-similarity and multi-word smoke after tuning, but early versions were too slow / noisy; user asked to roll back to FAST+ (#6) |

---

## Quick comparison

| Method | Continuous? | Template / model | Speed (rough) | Notes |
|--------|-------------|------------------|---------------|--------|
| Isolated GRU | No | `sign_gru.pt` | Fast | One word per space |
| DTW | Yes | `.pkl` | Slow | Best nonlinear alignment |
| Fast cosine | Yes | `.pkl` | Fast | Linear time warp via resample |
| GRU windows | Yes | `sign_gru.pt` | Fastest | Duplicate-prone |
| GRU stable | Yes | `sign_gru.pt` | Fast | Cleaner GRU decode |
| FAST+ | Yes | `.pkl` | Fast | Cosine + motion + lag |
| Dance-style FAST+ | Yes (reverted) | `.pkl` | Was slow | Angle/coord like dance scorer |

---

## Shared pieces (not separate “models”)

| File | Role |
|------|------|
| `sequence_utils.py` | Normalize + resample (+ augment for training) |
| `landmarks.py` | MediaPipe hand tracking |
| `collect_sequences.py` | Record word clips → `.pkl` |
| `train_lstm.py` | Train → `models/sign_gru.pt` |
| `chat.py` / `refine_prompt.py` | LLM cleanup + reply (Ollama) |

---

## Suggested default for continuous demo

1. **Similarity path:** `app_fast_plus.py` (or `app_fast_continuous.py` if you want the simpler cosine-only version).  
2. **Classifier path:** `app_gru_stable.py`.  
3. **Strict alignment / offline:** `app_dtw_continuous.py` / `recognize_video_dtw.py`.

All continuous apps share the same UX: **space** start/stop recording → decode word list → **r** send to LLM.
