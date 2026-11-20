# SoundScope

SoundScope is a local audio intelligence workbench for inspecting recordings with transparent signal processing, optional pretrained sound classification, silence intervals, comparison and persistent analysis history. It is designed as an analysis aid; features and model scores do not establish the identity or source of a sound.

## Features

- Upload and validate WAV, FLAC and MP3 recordings (25 MB, 120 seconds).
- Decode to mono and resample the analysis signal to 16 kHz; preserve the original for playback.
- RMS, zero crossing, spectral centroid, bandwidth, rolloff and 13 MFCC coefficients.
- Downsampled STFT magnitude spectrogram in decibels.
- Frame RMS low-energy intervals and chunk-level AST predictions (top three per 10-second chunk).
- Compare mean signal feature vectors, and store/reopen/delete analyses using SQLite.

## Run locally

Requires Python 3.11+, Node.js 20+, and a C/C++ build toolchain where Python wheels are unavailable. AST inference downloads model weights on first use; CPU inference is supported and can take time. Signal analysis remains available if model initialization fails.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. API docs are at http://localhost:8000/docs.

## Configuration

Copy `backend/.env.example` to `backend/.env` and configure the environment before starting the API. Audio and SQLite data are kept in `backend/uploads` and `backend/soundscope.db` by default. Delete recordings from Analysis History when no longer needed. `.env` files are ignored by Git.

## API

See [docs/API.md](docs/API.md). The browser uploads a recording, then requests its analysis. Classification failure is reported separately while signal processing remains available.

## Architecture

```text
React/Vite ── /api ── FastAPI ── validation / preprocessing / DSP
                               ├── optional lazy AST pipeline
                               └── SQLite metadata + local audio files
```

## Model and limitations

Model: `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face Transformers. Its outputs describe ten-second chunk classifications, not frame-level sound event detection or speech recognition. Confidence scores are not guarantees. No speaker identification is provided. Low-energy interval detection is threshold-based and may classify quiet content as silence. Similarity compares normalized average signal features and does not prove a common recording source. No noise reduction is applied.

No benchmark metrics are claimed. Evaluation needs a labeled, held-out dataset and a documented class mapping; see [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md).

## Security

Uploaded bytes are size-limited and decoded before acceptance. Stored filenames are generated UUIDs; original display filenames never form storage paths. Audio is untrusted, local data and is not exposed by filesystem path. The local API is intended for trusted development environments; production deployments should add authentication, quotas and managed retention.

## Tests

```powershell
cd backend
pytest
```

Tests cover DSP calculations, silence intervals, endpoint validation, persistence and comparisons without downloading model weights.
