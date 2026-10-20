# How do you sound? · Real-time speech emotion recognition

Talk into your microphone and a fine-tuned **wav2vec2** model guesses the emotion in your voice, live, **entirely in your
browser**: no server, and your audio never leaves your device.

**▶ Live demo: _DEMO_URL_**

<!-- GIF: record the demo and save it as docs/demo.gif -->

| | Accuracy | Macro-F1 |
|---|---|---|
| Original CNN, original evaluation (leaky) | 63.5% | 63.1% |
| Logistic regression, unseen speakers | 33.6% | 32.2% |
| Original CNN, unseen speakers | 35.8% | 33.9% |
| **Fine-tuned wav2vec2, unseen speakers** | **_W2V_ACC_** | **_W2V_F1_** |

<sub>RAVDESS speech, 8 emotions (chance = 12.5%). "Unseen speakers" = 6 actors held out from training (360 clips).</sub>

## What this project is

It started as a course mini-project ([Trilha UFPB](LICENSE)): a Keras CNN on hand-crafted audio features with a Streamlit
front-end. Revisiting it, I found that it didn't work the way its README said, and rebuilt it:

| Problem in the original | What it caused | Fix |
|---|---|---|
| Training extracted features at 22 kHz from a 2.5 s crop with 20 MFCCs; the app used 16 kHz, the full clip and 40 MFCCs, truncated to fit | The app fed the model features it had never seen: its predictions were close to meaningless | One model that reads **raw audio**, so there is no feature pipeline to drift; the exported model is tested against the PyTorch one |
| Augmented copies of every clip were made **before** a random train/test split, and every actor appeared on both sides | 67% accuracy that was really "recognise this clip / this actor"; the same CNN scores **36%** on new speakers | Fixed **speaker-independent split** (by actor, gender-balanced), augmentation on the training set only |
| Absolute paths, model not in the repo, unpinned dependencies | Nobody else could run it | `uv` project with a lockfile, scripts that download the data, model published on the Hugging Face Hub |
| Streamlit app | Can't be hosted on Vercel; needs a server | Static Next.js site with in-browser inference |

## How it works

```
 mic ──► AudioWorklet ──► 3 s ring buffer ──► every 0.5 s: silence gate ──► resample to 16 kHz
                                                                                   │
           UI (smoothed bars) ◄── probabilities ◄── Web Worker: wav2vec2 int8 ONNX ◄┘
                                                    (transformers.js / onnxruntime-web, WASM)
```

- **Model:** `facebook/wav2vec2-base` (pretrained on 960 h of unlabelled speech) with a classification head, fine-tuned on
  random 3-second windows of RAVDESS with noise augmentation. Its transformer matmuls are quantised to int8 for the browser
  (_INT8_MB_ MB download, cached after the first visit).
- **Real-time loop:** the page keeps the last 3 s of microphone audio. Every 500 ms, if the last second is louder than a
  silence threshold, the window goes to a Web Worker. If the worker is still busy the tick is dropped rather than queued, so
  the display never lags behind. Probabilities are smoothed with an exponential moving average.
- **Files and samples:** uploads are decoded and resampled in the browser and classified in overlapping 3 s windows, with a
  per-window timeline.
- **Why in the browser:** Vercel serverless functions can't hold PyTorch or a 95 MB model comfortably, and a GPU server would
  cost money and add latency. Client-side inference is free to host, private, and fast enough on a laptop.

## Results

Notebook [2-models.ipynb](notebooks/2-models.ipynb) has the full comparison, per-emotion F1 and confusion matrices. In short:

- The fine-tuned wav2vec2 roughly **_W2V_GAIN_** the accuracy of the original CNN on unseen speakers.
- Remaining errors are mostly between acoustically close emotions: calm ↔ neutral ↔ sad, and happy ↔ surprise ↔ fear.
- The int8 browser model agrees with the PyTorch model on _AGREE_ of unseen-speaker clips.

## Limitations

- **RAVDESS is acted, scripted, studio-quality North American English**: 24 actors, two sentences. Real conversations, other
  languages and cheap microphones are much harder, and the demo will often say *neutral* or *calm* to natural speech.
- It hears **tone, not words**.
- Licensing: RAVDESS is CC BY-NC-SA 4.0, so the model and demo are non-commercial.

## Repository

```
emotion/            Python package: data + split, baselines, fine-tuning, ONNX export, inference
tests/              pytest: the exported ONNX model's predict() contract and a quality floor
notebooks/          1-eda.ipynb (data analysis) · 2-models.ipynb (model comparison)
results/            metrics written by the scripts (read by the notebooks and the website)
fixtures/           8 short RAVDESS clips from held-out actors, used by both test suites
web/                Next.js static site; lib/inference.ts is the worker's model code (vitest)
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Node 22.

```bash
uv sync --group dev
uv run python -m emotion.data         # downloads RAVDESS speech (~200 MB) and checks the split
uv run python -m emotion.baselines    # logistic regression + original CNN (leaky and honest)
uv run python -m emotion.finetune     # wav2vec2 fine-tuning (~1.5 h on an M4 MacBook, Apple GPU)
uv run python -m emotion.export       # ONNX fp32 + int8, parity check, size/latency → artifacts/onnx
uv run pytest

cd web && npm ci
npm test                              # worker tests against the local export
mkdir -p public/models && ln -s ../../../artifacts/onnx public/models/speech-emotion
NEXT_PUBLIC_MODEL_LOCAL=1 npm run dev # or leave the variable out to load the model from the Hub
```

## Credits

- Data: Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).*
  PLoS ONE 13(5). [Zenodo](https://zenodo.org/records/1188976), CC BY-NC-SA 4.0.
- Pretrained model: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base).
- Original project template and course: Trilha UFPB.
