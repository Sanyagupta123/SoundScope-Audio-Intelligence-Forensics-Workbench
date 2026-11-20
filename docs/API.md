# API reference

All endpoints use JSON except the multipart upload. Errors use FastAPI's `{ "detail": "..." }` form.

- `GET /api/health` — API status and model identifier.
- `POST /api/audio/upload` — multipart `file`; returns audio ID and validated metadata.
- `POST /api/audio/{audio_id}/analyze` — extracts DSP, spectrogram, silence intervals and optional AST chunk predictions.
- `GET /api/audio/{audio_id}/results` — saved analysis JSON.
- `GET /api/audio/{audio_id}/stream` — original recording for playback.
- `GET /api/audio/{audio_id}/spectrogram` — spectrogram data.
- `POST /api/compare` — JSON `{ "audio_a": "uuid", "audio_b": "uuid" }`; both must be analyzed.
- `GET /api/history` and `GET /api/history/{analysis_id}` — list and detail.
- `DELETE /api/history/{analysis_id}` — deletes DB entry and stored audio.
