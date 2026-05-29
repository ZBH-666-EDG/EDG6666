# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Elderly fall-detection microservice using YOLOv8-pose + InsightFace + DeepSeek AI. A Flask + SocketIO web app with multi-camera support, real-time MJPEG streaming, multi-person IoU-based tracking, probabilistic fall scoring, face recognition, and optional AI-powered incident analysis.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time database init (app.py also auto-initializes on start)
python init_db.py

# Start the server
python app.py
# or on Windows: double-click run.bat
```

The app binds to `http://0.0.0.0:5001`. There are no tests or linters configured in this project.

## Architecture

### Single-file backend (`app.py`)

All server logic lives in a single ~1500-line `app.py`. Key subsystems:

**Thread model (per camera):**
```
Camera capture thread (in generate_frames)
    → frame_queue (Queue, max 60)
        → detection_worker thread (runs YOLO every 3rd frame, face rec every 30th frame)
            → SSE push + WebSocket broadcast to clients
            → ThreadPoolExecutor (max 2) for async AI analysis
```

**Core globals:**
- `CAMERAS` — list of `{id, source}` loaded from `cameras.json` at startup. `source` is an int (USB index) or string (RTSP URL).
- `tracked_persons` — dict of `{track_id: {bbox, kp, kp_conf, hip_history, angle_history, fall_counter, name, color}}`, managed by a simple IoU-based multi-person tracker.
- Per-camera dicts: `frame_queues`, `latest_detections`, `detection_locks`, `current_fps_list`, `person_count_list`, `last_p_fall_list`, `camera_enabled`.

**Fall detection (`check_fall`, line 429):** Multi-feature fusion — torso angle (35%), vertical velocity (25%), aspect ratio (20%), angular acceleration (12%), head-foot Y diff (8%). Each feature feeds a sigmoid, weighted, then re-sigmoided to produce `P_FALL` (0–1).

**Alert levels:**
- Level 1 (warning): `0.55 ≤ P_FALL < 0.75` — yellow overlay, short beep
- Level 2 (fall): `P_FALL ≥ 0.75` for 2 consecutive frames — red overlay, continuous alarm, screenshot saved, event written to DB, AI analysis queued

**Face recognition:** InsightFace `buffalo_sc` on CPU. CLAHE preprocessing, cosine similarity against stored embeddings in SQLite. Auto-learns new embeddings on high-confidence matches (≥0.70). Name held for 90 frames after last match.

**Database (`faces.db`):** SQLite with 3 tables — `persons`, `face_embeddings` (FK to persons, stores numpy blobs), `events` (fall logs with optional AI reports and `permanent` flag). Schema auto-created and migrated on startup. Events older than 7 days are auto-purged (unless `permanent=1`).

**AI analysis (`config.py`):** Supports DeepSeek (default) and Gemini. Sends a text-only prompt to an LLM with event context (name, P_FALL, timestamp). Toggled on/off via `/api/toggle_ai` (off by default even if API key is set).

### Templates (Jinja2 server-rendered)

| Template | Purpose |
|---|---|
| `index.html` | Main monitoring dashboard with multi-camera grid, alerts overlay, audio |
| `register.html` | 3-step face registration wizard |
| `manage.html` | Person/face management |
| `history.html` | Fall event history with AI reports |
| `cameras.html` | Camera config (add/delete/rename/ONVIF scan) |
| `test.html` | Upload video for offline testing |

### External integration

- **iframe embed:** `GET /video_feed/<cam_id>` returns MJPEG stream
- **WebSocket:** SocketIO on `/` — emits `alert` events (type, level, name, confidence, timestamp, screenshot, event_id)
- **SSE:** `GET /events` — same payload as WebSocket
- **REST API:** health check, event CRUD, face registration/management, camera management — see README.md for full table

### Key configuration constants (top of `app.py`)

`FALL_PROB_THRESHOLD=0.75`, `WARN_PROB_THRESHOLD=0.55`, `FALL_CONSECUTIVE_FRAMES=2`, `FACE_RECOGNITION_INTERVAL=30`, `FACE_SIMILARITY_THRESHOLD=0.50`, `YOLO_IMGSZ=256`, `TRACK_MAX_LOST=30`, `IOU_MATCH_MIN=0.3`.

### Important patterns

- Camera config changes (`/api/cameras` POST/DELETE) require server restart to take effect — they only modify `cameras.json`.
- The test mode (`/test`) disables all live cameras before playback and restores them on reset.
- The `recognized_name` global reflects the named track with the highest `P_FALL` — this is what gets recorded when a fall triggers.
- ONVIF camera auto-discovery (`/api/cameras/scan`) requires `onvif-zeep` and tries common Hikvision default credentials.
