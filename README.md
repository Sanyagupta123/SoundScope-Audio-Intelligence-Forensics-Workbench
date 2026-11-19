# SceneMind

SceneMind turns an image into YOLO object detections, a geometry-based spatial scene graph, grounded descriptions, graph-backed questions, and approximate two-image change comparisons. The frontend and API are separate applications.

## Run locally

Requirements: Python 3.11+, Node.js 20+. From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The first analysis downloads the pretrained `yolo11n.pt` weights through Ultralytics (network access is required once); the model is cached by Ultralytics and loaded once per server process. Set `SCENEMIND_MODEL` to a compatible local Ultralytics weight path. No weights or user images are stored by this application.

API docs: http://localhost:8000/docs. Health: http://localhost:8000/api/health.

## API

* `POST /api/analyze` — multipart `file`, optional `confidence` from 0 to 1.
* `POST /api/ask` — JSON `{ "question": "...", "scene": <analysis response> }`.
* `POST /api/compare` — multipart `image_a`, `image_b`, and optional `confidence`.

Accepted image formats are content-validated JPEG, PNG and WEBP. Uploads are limited to 5 MB, 8192 pixels on either side and 30 megapixels. Files are read in memory and are not exposed or persisted. Relationship scores are geometry heuristics, not model probabilities; boxes cannot establish actions such as holding or sitting on. Image comparison matches labels and normalized positions approximately.

## Architecture

`backend/app/detector.py` lazily loads and reuses pretrained YOLO. `scene.py` validates image content, calculates normalized features and IoU, infers evidenced geometric edges, builds a NetworkX graph, serializes it, and answers questions. `main.py` exposes validated FastAPI endpoints. `frontend/src/App.jsx` coordinates upload, confidence threshold, synchronized object selection, image overlays, Cytoscape graph, questions, and comparison. Vite proxies API calls to the local backend.

## Tests

```powershell
cd backend
pip install -r requirements.txt
pytest
```

Unit tests exercise geometry, graph edge cases, and grounded question handling without loading YOLO. Useful images can be any locally available JPEG, PNG, or WEBP under 5 MB; no sample imagery is redistributed.
