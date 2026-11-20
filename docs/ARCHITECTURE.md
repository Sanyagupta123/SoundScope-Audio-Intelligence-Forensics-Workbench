# Architecture

The React application uses a single Axios API client and keeps the current analysis in the Analyze view. FastAPI validates and decodes uploads before persisting them under UUID filenames. A small SQLite repository stores metadata and serialized analysis results. Signal processing is isolated in `backend/app/services.py`; model loading is lazy and cached once per process. The AST path is optional at runtime, and errors appear beside usable DSP results.

Audio is analyzed as mono at 16 kHz. The original upload is retained for playback. Requests are synchronous for this local MVP and capped at two minutes; a production service should move inference to a worker queue.
