"""
Fall Detection Microservice — Multi-level alerts, SSE + WebSocket, REST API.
Supports iframe embedding and external system integration.

v1.0 — 单人追踪版
  仅处理画面中最大的人体（primary_person），适合单老人居家场景。
  多人监测请见后续版本。
"""
import cv2
import numpy as np
import math
import sqlite3
import os
import json
import time
import queue
import threading
import base64
import io
import requests
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO

import config as cfg

# ============================================================
# Configuration
# ============================================================
FALL_PROB_THRESHOLD = 0.75       # P_FALL above this = confirmed fall (level 2)
WARN_PROB_THRESHOLD = 0.55       # P_FALL above this = unstable warning (level 1)
FALL_CONSECUTIVE_FRAMES = 2
WARN_HOLD_FRAMES = 5             # hold warning state for N frames
DETECTION_INTERVAL = 3
FALL_COOLDOWN_SECONDS = 5

FACE_RECOGNITION_INTERVAL = 30
FACE_SIMILARITY_THRESHOLD = 0.50
FACE_DET_SCORE_THRESHOLD = 0.3
FACE_NAME_HOLD_FRAMES = 90
INSIGHTFACE_CTX_ID = -1

YOLO_IMGSZ = 256
JPEG_QUALITY = 75

# ============================================================
# Flask + SocketIO Setup
# ============================================================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

os.makedirs('static/falls', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# ============================================================
# Models & Camera config
# ============================================================
model = YOLO('yolov8n-pose.pt')

# Camera config — persisted to cameras.json, editable via /cameras page
CAMERA_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'cameras.json')


def _load_camera_config():
    """Load camera list + names from JSON. Returns (cameras, names)."""
    if os.path.isfile(CAMERA_CONFIG_FILE):
        try:
            with open(CAMERA_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('cameras', [{'id': 0, 'source': 0}]), data.get('names', {})
        except Exception:
            pass
    return [{'id': 0, 'source': 0}], {}


def _save_camera_config(cameras, names):
    with open(CAMERA_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump({'cameras': cameras, 'names': names}, f, ensure_ascii=False, indent=2)


CAMERAS, camera_names = _load_camera_config()

# Each camera gets its own pipeline
pipelines = {}
frame_queues = {}
latest_detections = {}
detection_locks = {}
current_fps_list = {}
person_count_list = {}
last_p_fall_list = {}
camera_enabled = {}  # id -> bool, whether camera is actively streaming
test_video_path = None
test_video_lock = threading.Lock()
test_saved_cam_states = {}  # save camera states before test, restore after

face_app = None
try:
    import insightface
    face_app = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=INSIGHTFACE_CTX_ID)
    print("[InsightFace] buffalo_sc loaded (CPU)")
except Exception as e:
    print(f"[InsightFace] WARNING: {e}")

# ============================================================
# Database
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faces.db')


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_v2(conn):
    rows = conn.execute('SELECT name, embedding_blob, photo_path, created_at FROM faces').fetchall()
    for row in rows:
        cur = conn.execute('INSERT INTO persons (name, created_at) VALUES (?, ?)',
                           (row['name'], row['created_at']))
        conn.execute('INSERT INTO face_embeddings (person_id, embedding_blob, photo_path, created_at) '
                     'VALUES (?, ?, ?, ?)',
                     (cur.lastrowid, row['embedding_blob'], row['photo_path'], row['created_at']))
    conn.execute('DROP TABLE IF EXISTS faces')
    conn.commit()
    print(f"[DB] Migrated {len(rows)} records to v2")


def init_db():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                 'name TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.execute('CREATE TABLE IF NOT EXISTS face_embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                 'person_id INTEGER NOT NULL, embedding_blob BLOB NOT NULL, '
                 'photo_path TEXT, det_score REAL DEFAULT 0.0, '
                 'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, '
                 'FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_emb_pid ON face_embeddings(person_id)')
    conn.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                 'elder_name TEXT DEFAULT "陌生人", confidence REAL, screenshot TEXT, '
                 'report TEXT DEFAULT "", permanent INTEGER DEFAULT 0, '
                 'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    # Add permanent column if missing (migration for existing DB)
    cols = [c[1] for c in conn.execute('PRAGMA table_info(events)').fetchall()]
    if 'permanent' not in cols:
        conn.execute('ALTER TABLE events ADD COLUMN permanent INTEGER DEFAULT 0')
    conn.commit()
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
    if cur.fetchone():
        _migrate_v2(conn)
    conn.close()
    print("[DB] faces.db initialized (v2)")


init_db()
ai_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='ai')
ai_toggle = False  # AI analysis off by default, user can toggle on via UI

# ---- Event cleanup: 7-day auto-delete, permanent events preserved ----
def cleanup_old_events():
    """Delete events older than 7 days that are not marked permanent."""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT id, screenshot FROM events WHERE permanent = 0 AND "
            "datetime(created_at) < datetime('now', '-7 days')"
        ).fetchall()
        for r in rows:
            if r['screenshot']:
                fp = os.path.join(os.path.dirname(__file__), r['screenshot'].lstrip('/'))
                if os.path.isfile(fp):
                    os.remove(fp)
        conn.execute(
            "DELETE FROM events WHERE permanent = 0 AND "
            "datetime(created_at) < datetime('now', '-7 days')"
        )
        conn.commit()
        conn.close()
        if len(rows) > 0:
            print(f"[Cleanup] Removed {len(rows)} old event(s)")
    except Exception as e:
        print(f"[Cleanup] Error: {e}")


def cleanup_loop():
    """Periodic cleanup every 6 hours."""
    cleanup_old_events()
    while alive.is_set():
        alive.wait(21600)  # 6 hours
        cleanup_old_events()

# ============================================================
# Global State
# ============================================================
fall_queue = queue.Queue(maxsize=10)
hip_history = deque(maxlen=5)
angle_history = deque(maxlen=5)
fall_counter = 0
last_fall_time = 0
current_fps = 0
person_count = 0
fps_timer = time.time()
fps_frame_count = 0
recognized_name = None
state_lock = threading.Lock()
last_face_match_frame = 0
held_face_name = None
alive = threading.Event()
alive.set()

_auto_learn_history = {}
auto_learn_lock = threading.Lock()

# ============================================================
# Skeleton Drawing Data
# ============================================================
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KP_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 255, 0), (255, 255, 0), (0, 255, 255), (0, 255, 255),
    (0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255),
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
]


# ============================================================
# Sigmoid helper
# ============================================================
def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ============================================================
# Face recognition helpers
# ============================================================
def extract_face_embedding(img_bgr):
    if face_app is None:
        return None, 0.0
    faces = face_app.get(img_bgr)
    if len(faces) == 0:
        return None, 0.0
    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < FACE_DET_SCORE_THRESHOLD:
        return None, float(best.det_score)
    return best.embedding, float(best.det_score)


def _auto_learn(person_id, embedding, det_score, frame_no):
    with auto_learn_lock:
        last = _auto_learn_history.get(person_id, 0)
        if frame_no - last < 90:
            return
        _auto_learn_history[person_id] = frame_no
    conn = get_db()
    conn.execute('INSERT INTO face_embeddings (person_id, embedding_blob, det_score) VALUES (?, ?, ?)',
                 (person_id, embedding.tobytes(), det_score))
    conn.commit()
    conn.close()
    print(f"[AutoLearn] New embedding for person_id={person_id} (det={det_score:.2f})")


def recognize_face(frame_bgr, frame_no=0):
    if face_app is None:
        return None, 0.0
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    faces = face_app.get(processed)
    if len(faces) == 0:
        return None, 0.0
    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < FACE_DET_SCORE_THRESHOLD:
        return None, 0.0

    embedding = best.embedding
    emb_norm = np.linalg.norm(embedding)
    if emb_norm < 1e-10:
        return None, 0.0

    conn = get_db()
    rows = conn.execute('''SELECT p.id AS person_id, p.name, e.embedding_blob
        FROM persons p JOIN face_embeddings e ON e.person_id = p.id''').fetchall()
    conn.close()
    if len(rows) == 0:
        return None, 0.0

    person_best = {}
    for row in rows:
        db_emb = np.frombuffer(row['embedding_blob'], dtype=np.float32)
        db_norm = np.linalg.norm(db_emb)
        if db_norm < 1e-10:
            continue
        sim = float(np.dot(embedding, db_emb) / (emb_norm * db_norm))
        pid = row['person_id']
        if pid not in person_best or sim > person_best[pid][1]:
            person_best[pid] = (row['name'], sim)

    if not person_best:
        return None, 0.0
    best_person = max(person_best.values(), key=lambda x: x[1])
    if best_person[1] >= FACE_SIMILARITY_THRESHOLD:
        if best_person[1] >= 0.70:
            best_pid = max(person_best, key=lambda k: person_best[k][1])
            _auto_learn(best_pid, embedding, best.det_score, frame_no)
        return best_person[0], best_person[1]
    return None, best_person[1]


# ============================================================
# YOLO helpers
# ============================================================
# ============================================================
# Multi-person tracker (simple IoU-based)
# ============================================================
TRACK_MAX_LOST = 30    # frames before removing a lost track
IOU_MATCH_MIN = 0.3    # minimum IoU to consider a match
next_track_id = 0
tracked_persons = {}   # id -> {bbox,last_seen,kp,kp_conf,hip_hist,angle_hist,fall_cnt,name,color}
tracker_lock = threading.Lock()

PERSON_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255),
    (255, 255, 0), (0, 255, 200), (200, 100, 255), (255, 200, 0),
]


def _iou(boxA, boxB):
    """Intersection-over-Union of two xyxy boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter)


def _match_or_create_tracks(detections, frame_no):
    """
    detections: list of (bbox_xyxy, kp_xy, kp_conf)
    Matches detections to existing tracks via IoU, creates new tracks as needed.
    """
    global next_track_id
    matched_tids = set()
    new_tracks = {}

    for det_bbox, kp, kp_conf in detections:
        best_tid, best_iou = None, 0
        for tid, t in tracked_persons.items():
            if tid in matched_tids:
                continue
            iou = _iou(det_bbox, t['bbox'])
            if iou > best_iou:
                best_iou = iou; best_tid = tid
        if best_tid is not None and best_iou >= IOU_MATCH_MIN:
            tid = best_tid; matched_tids.add(tid)
            t = tracked_persons[tid]
            t['bbox'] = det_bbox; t['last_seen'] = frame_no
            t['kp'] = kp; t['kp_conf'] = kp_conf
            new_tracks[tid] = t
        else:
            tid = next_track_id; next_track_id += 1
            color = PERSON_COLORS[tid % len(PERSON_COLORS)]
            new_tracks[tid] = {
                'bbox': det_bbox, 'last_seen': frame_no,
                'kp': kp, 'kp_conf': kp_conf,
                'hip_history': deque(maxlen=5), 'angle_history': deque(maxlen=5),
                'fall_counter': 0, 'name': None, 'color': color,
            }

    # Remove stale tracks
    stale = [tid for tid in tracked_persons if frame_no - tracked_persons[tid]['last_seen'] > TRACK_MAX_LOST]
    for tid in stale:
        del tracked_persons[tid]

    tracked_persons.clear()
    tracked_persons.update(new_tracks)
    return new_tracks


def all_persons(result):
    """Extract all detected persons from YOLO result. Returns list of (bbox, kp_xy, kp_conf)."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return []
    kps = result.keypoints.xy.cpu().numpy()
    confs = result.keypoints.conf.cpu().numpy()
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
    else:
        # Fallback: use keypoint min/max as approximate box
        boxes = np.array([[k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()] for k in kps])
    return [(tuple(boxes[i]), kps[i], confs[i]) for i in range(len(kps))]


def draw_skeleton(frame, kp_xy, kp_conf, is_fall=False):
    edge_color = (0, 0, 255) if is_fall else (0, 255, 0)
    for a, b in SKELETON_EDGES:
        if kp_conf[a] > 0.5 and kp_conf[b] > 0.5:
            cv2.line(frame, (int(kp_xy[a][0]), int(kp_xy[a][1])),
                     (int(kp_xy[b][0]), int(kp_xy[b][1])), edge_color, 2)
    for i in range(len(kp_xy)):
        if kp_conf[i] > 0.5:
            cx, cy = int(kp_xy[i][0]), int(kp_xy[i][1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255) if is_fall else KP_COLORS[i], -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)


# ============================================================
# Probabilistic Fall Detection (multi-feature fusion)
# ============================================================
def check_fall(kp_xy, kp_conf):
    def k(idx):
        return (kp_xy[idx], kp_conf[idx]) if kp_conf[idx] > 0.5 else None

    ls = k(5); rs = k(6); lh = k(11); rh = k(12)
    nose = k(0); l_ankle = k(15); r_ankle = k(16)

    shoulders = [p[0] for p in (ls, rs) if p is not None]
    hips = [p[0] for p in (lh, rh) if p is not None]
    if len(shoulders) < 1 or len(hips) < 1:
        return False, None

    sx = np.mean([p[0] for p in shoulders]); sy = np.mean([p[1] for p in shoulders])
    hx = np.mean([p[0] for p in hips]); hy = np.mean([p[1] for p in hips])

    # Feature 1: torso angle
    dx = abs(hx - sx); dy = abs(hy - sy)
    angle = math.degrees(math.atan2(dy, dx)) if (dx + dy) > 1e-6 else 90.0
    p_angle = _sigmoid((45 - angle) / 12.0)

    # Feature 2: vertical velocity
    hip_history.append(hy)
    velocity = 0.0
    if len(hip_history) >= 3:
        velocity = (hip_history[-1] - hip_history[0]) / (len(hip_history) - 1)
    p_vel = _sigmoid((velocity - 12) / 8.0)

    # Feature 3: aspect ratio
    all_x = [p[0] for p in shoulders + hips]; all_y = [p[1] for p in shoulders + hips]
    bw = max(all_x) - min(all_x) + 1; bh = max(all_y) - min(all_y) + 1
    ar = bw / bh if bh > 0 else 0.5
    p_ar = _sigmoid((ar - 0.8) * 6.0)

    # Feature 4: angular acceleration
    angle_history.append(angle)
    angle_accel = 0.0
    if len(angle_history) >= 4:
        angle_accel = abs(angle_history[-1] - angle_history[-4]) / 3.0
    p_accel = _sigmoid((angle_accel - 5.0) / 3.0)

    # Feature 5: head-foot Y diff
    p_hf = 0.5; hf_diff = 0.0
    if nose is not None:
        ankles = [p[0] for p in (l_ankle, r_ankle) if p is not None]
        if ankles:
            ankle_y = np.mean([p[1] for p in ankles])
            hf_diff = abs(ankle_y - nose[0][1])
            p_hf = _sigmoid((150 - hf_diff) / 60.0)

    P_FALL = (p_angle * 0.35 + p_vel * 0.25 + p_ar * 0.20 + p_accel * 0.12 + p_hf * 0.08)
    P_FALL = _sigmoid((P_FALL - 0.50) * 12.0)

    is_fall = P_FALL >= FALL_PROB_THRESHOLD
    return is_fall, {
        'p_fall': round(P_FALL, 3), 'angle': round(angle, 1), 'velocity': round(velocity, 1),
        'ar': round(ar, 2), 'angle_accel': round(angle_accel, 1),
        'p_angle': round(p_angle, 2), 'p_vel': round(p_vel, 2),
        'p_ar': round(p_ar, 2), 'p_accel': round(p_accel, 2), 'p_hf': round(p_hf, 2),
    }


# ============================================================
# Alert broadcast (SSE + WebSocket)
# ============================================================
def broadcast_alert(event_data):
    """Push to SSE queue and WebSocket broadcast."""
    try:
        fall_queue.put_nowait(event_data)
    except queue.Full:
        pass
    try:
        socketio.emit('alert', event_data)
    except Exception:
        pass


# ============================================================
# AI Analysis
# ============================================================
AI_PROMPT = """分析这张老年人跌倒图片，输出以下内容（用中文）：
1）可能原因（环境因素 / 健康因素 / 动作因素）
2）风险评估（高 / 中 / 低）
3）急救建议
4）是否需要呼叫家属（是 / 否，并说明理由）"""


def analyze_fall_image(screenshot_path, event_id):
    """Text-based AI analysis using event data (works with all LLMs including deepseek-chat)."""
    if not cfg.AI_ENABLED:
        print(f"[AI] Skipped (no API key) for event #{event_id}")
        return

    conn = get_db()
    row = conn.execute(
        'SELECT elder_name, confidence, created_at FROM events WHERE id = ?', (event_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return

    text_prompt = (
        f"你是一个老年人跌倒监测AI助手。刚刚发生了一起跌倒事件，请基于以下信息分析：\n\n"
        f"老人姓名：{row['elder_name']}\n"
        f"跌倒概率（P_FALL）：{row['confidence']:.0%}\n"
        f"时间：{row['created_at']}\n\n"
        f"请输出：\n"
        f"1）可能原因（环境因素 / 健康因素 / 动作因素）\n"
        f"2）风险评估（高 / 中 / 低）\n"
        f"3）急救建议\n"
        f"4）是否需要呼叫家属（是 / 否，并说明理由）\n"
        f"5）预防建议"
    )

    try:
        if cfg.AI_PROVIDER == 'gemini':
            payload = {'contents': [{'parts': [{'text': text_prompt}]}]}
            headers = {'Content-Type': 'application/json'}
            endpoint = cfg.get_endpoint()
        else:
            payload = {'model': cfg.AI_MODEL, 'messages': [{'role': 'user', 'content': text_prompt}],
                       'max_tokens': 800, 'temperature': 0.3}
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {cfg.AI_API_KEY}'}
            endpoint = cfg.get_endpoint()

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=cfg.AI_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        report = data['candidates'][0]['content']['parts'][0]['text'] if cfg.AI_PROVIDER == 'gemini' \
            else data['choices'][0]['message']['content']

        conn = get_db()
        conn.execute('UPDATE events SET report = ? WHERE id = ?', (report, event_id))
        conn.commit()
        conn.close()
        print(f"[AI] Report saved for event #{event_id}\n{report[:200]}")
    except requests.exceptions.Timeout:
        print(f"[AI] Timeout ({cfg.AI_TIMEOUT}s) for event #{event_id}")
    except Exception as e:
        print(f"[AI] Failed for event #{event_id}: {e}")


# ============================================================
# Fall Event Trigger
# ============================================================
def trigger_fall_event(frame, info):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join('static', 'falls', filename)
    cv2.imwrite(filepath, frame)

    p_fall = info.get('p_fall', 0.5) if info else 0.5
    with state_lock:
        fall_name = recognized_name if recognized_name else '陌生人'
    screenshot_url = f'/static/falls/{filename}'

    conn = get_db()
    cur = conn.execute('INSERT INTO events (elder_name, confidence, screenshot) VALUES (?, ?, ?)',
                       (fall_name, p_fall, screenshot_url))
    event_id = cur.lastrowid
    conn.commit()
    conn.close()

    if cfg.AI_ENABLED and ai_toggle:
        ai_executor.submit(analyze_fall_image, screenshot_url, event_id)

    event_data = {'type': 'fall', 'level': 2, 'name': fall_name, 'confidence': p_fall,
                  'timestamp': ts, 'screenshot': screenshot_url, 'event_id': event_id}
    broadcast_alert(event_data)

    if info:
        print(f"[跌倒检测到] {ts} 老人: {fall_name}  P_FALL={info.get('p_fall','?')}  "
              f"angle={info['angle']}° vel={info['velocity']}  ar={info['ar']}  "
              f"p=[a:{info['p_angle']} v:{info['p_vel']} r:{info['p_ar']} "
              f"aa:{info['p_accel']} hf:{info['p_hf']}]  event_id={event_id}")


# ============================================================
# Detection Worker Thread
# ============================================================
def detection_worker(cam_id):
    """Per-camera detection worker — reads from frame_queues[cam_id], updates per-camera state."""
    det_frame_count = 0
    warn_hold = 0
    fq = frame_queues[cam_id]
    dl = detection_locks[cam_id]
    ld = latest_detections[cam_id]
    global last_fall_time

    while alive.is_set():
        try:
            frame = fq.get(timeout=1)
        except queue.Empty:
            continue

        det_frame_count += 1

        # YOLO pose detection
        if det_frame_count % DETECTION_INTERVAL == 0:
            results = model(frame, imgsz=YOLO_IMGSZ, conf=0.5, verbose=False)
            result = results[0]
            detections = all_persons(result)

            with tracker_lock:
                tracks = _match_or_create_tracks(detections, det_frame_count)
                person_count_list[cam_id] = len(tracks)

            # Per-person face recognition
            if face_app is not None and det_frame_count % FACE_RECOGNITION_INTERVAL == 0:
                for tid, t in tracks.items():
                    if t.get('name') is None:
                        bx, by = max(0, int(t['bbox'][0])-20), max(0, int(t['bbox'][1])-40)
                        bw2 = min(frame.shape[1], int(t['bbox'][2])+20)
                        bh2 = min(frame.shape[0], int(t['bbox'][3])+10)
                        face_crop = frame[by:bh2, bx:bw2]
                        if face_crop.size > 0:
                            name, _ = recognize_face(face_crop, det_frame_count)
                            if name is not None:
                                t['name'] = name
                        break

            # Sync recognized_name from highest-confidence named track
            named_tracks = [(t.get('name'), t.get('last_p_fall', 0))
                            for t in tracks.values() if t.get('name')]
            with state_lock:
                if named_tracks:
                    recognized_name = max(named_tracks, key=lambda x: x[1])[0]

            # Process each tracked person
            max_p_fall = 0.0; any_is_fall = False
            for tid, t in tracks.items():
                kp, kp_conf = t['kp'], t['kp_conf']
                # Per-person histories — swap into globals temporarily
                sh = list(hip_history); sa = list(angle_history)
                hip_history.clear(); hip_history.extend(t['hip_history'])
                angle_history.clear(); angle_history.extend(t['angle_history'])
                is_fall_now, info = check_fall(kp, kp_conf)
                t['hip_history'] = deque(hip_history, maxlen=5)
                t['angle_history'] = deque(angle_history, maxlen=5)
                hip_history.clear(); hip_history.extend(sh)
                angle_history.clear(); angle_history.extend(sa)

                p_fall_val = info.get('p_fall', 0.0) if info else 0.0
                if p_fall_val > max_p_fall: max_p_fall = p_fall_val
                if is_fall_now: t['fall_counter'] += 1
                else: t['fall_counter'] = 0
                if t['fall_counter'] >= FALL_CONSECUTIVE_FRAMES:
                    any_is_fall = True; t['fall_counter'] = 0
                if info: t['last_p_fall'] = p_fall_val

            last_p_fall_list[cam_id] = max_p_fall

            # Level 1 warning
            if WARN_PROB_THRESHOLD <= max_p_fall < FALL_PROB_THRESHOLD:
                warn_hold = WARN_HOLD_FRAMES
            else:
                warn_hold = max(0, warn_hold - 1)
            if warn_hold > 0 and not any_is_fall:
                broadcast_alert({'type': 'warning', 'level': 1,
                                 'message': '姿态不稳', 'p_fall': max_p_fall, 'cam_id': cam_id})

            # Level 2: trigger per-person fall events
            now = time.time()
            for tid, t in tracks.items():
                if t['fall_counter'] >= FALL_CONSECUTIVE_FRAMES:
                    if (now - last_fall_time) > FALL_COOLDOWN_SECONDS:
                        last_fall_time = now
                        pname = t.get('name') or '陌生人'
                        with state_lock:
                            saved_name = recognized_name; recognized_name = pname
                        trigger_fall_event(frame, {'p_fall': t.get('last_p_fall', 0.75),
                                                    'angle': 0, 'velocity': 0, 'ar': 0,
                                                    'angle_accel': 0, 'p_angle': 0,
                                                    'p_vel': 0, 'p_ar': 0,
                                                    'p_accel': 0, 'p_hf': 0, 'cam_id': cam_id})
                        with state_lock:
                            recognized_name = saved_name
                        t['fall_counter'] = 0

            # Publish detection for drawing
            best_t = max(tracks.values(), key=lambda t: t.get('last_p_fall', 0)) if tracks else None
            with dl:
                ld['kp_xy'] = best_t['kp'] if best_t else None
                ld['kp_conf'] = best_t['kp_conf'] if best_t else None
                ld['is_fall'] = any_is_fall
                ld['tracks'] = tracks
        else:
            time.sleep(0.002)


# ============================================================
# MJPEG Stream Generator (reads camera, feeds detection, draws overlays)
# ============================================================
def open_camera(source):
    """Open a camera with DSHOW backend."""
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def generate_frames(cam_id):
    """MJPEG stream generator for a specific camera."""
    fq = frame_queues[cam_id]
    dl = detection_locks[cam_id]
    ld = latest_detections[cam_id]
    fc_start = time.time(); fc_count = 0
    cam = None

    while alive.is_set():
        # Disabled camera — show placeholder
        if not camera_enabled.get(cam_id, True):
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, f'{camera_names.get(str(cam_id), "Camera")} 已关闭', (40, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 30])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        if cam is None:
            cam = open_camera(CAMERAS[cam_id]['source'])

        success, frame = cam.read()
        if not success:
            time.sleep(0.1); continue

        fc_count += 1
        now = time.time(); elapsed = now - fc_start
        if elapsed >= 1.0:
            current_fps_list[cam_id] = round(fc_count / elapsed, 1)
            fc_count = 0; fc_start = now

        # Feed detection worker
        if fq.full():
            try: fq.get_nowait()
            except queue.Empty: pass
        fq.put(frame)

        # Draw all tracked persons
        with dl:
            tracks = ld.get('tracks', {})

        for tid, t in tracks.items():
            kp, kp_conf = t.get('kp'), t.get('kp_conf')
            if kp is None or kp_conf is None: continue
            is_fall_person = t.get('fall_counter', 0) > 0
            color = t.get('color', (0, 255, 0))
            pname = t.get('name') or f'ID:{tid}'
            p_fall_val = t.get('last_p_fall', 0)

            for a, b in SKELETON_EDGES:
                if kp_conf[a] > 0.5 and kp_conf[b] > 0.5:
                    c = (0, 0, 255) if is_fall_person else color
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), c, 2)
            for i in range(len(kp)):
                if kp_conf[i] > 0.5:
                    cx, cy = int(kp[i][0]), int(kp[i][1])
                    c = (0, 0, 255) if is_fall_person else color
                    cv2.circle(frame, (cx, cy), 4, c, -1)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

            bx, by = int(t['bbox'][0]), int(t['bbox'][1])
            label = f'{pname} | {p_fall_val:.2f}'
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = max(0, bx + int((t['bbox'][2] - t['bbox'][0] - lw) / 2))
            ly = max(20, by - 8)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # HUD
        fps = current_fps_list.get(cam_id, 0)
        pers = person_count_list.get(cam_id, 0)
        pf = last_p_fall_list.get(cam_id, 0)
        cam_name = camera_names.get(str(cam_id), f'摄像头{cam_id+1}')
        cv2.putText(frame, f"{cam_name} | FPS: {fps} | Persons: {pers}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        p_color = (0, 255, 0) if pf < WARN_PROB_THRESHOLD else (
            (0, 0, 255) if pf >= FALL_PROB_THRESHOLD else (0, 165, 255))
        cv2.putText(frame, f"P_FALL: {pf:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 1)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ============================================================
# Routes — Pages
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    if face_app is None:
        return jsonify({'ok': False, 'error': 'InsightFace 未加载'}), 500
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': '请输入姓名'}), 400
    photos = request.files.getlist('photo')
    photos = [f for f in photos if f and f.filename]
    if len(photos) == 0:
        return jsonify({'ok': False, 'error': '请上传至少一张照片'}), 400

    conn = get_db()
    row = conn.execute('SELECT id FROM persons WHERE name = ?', (name,)).fetchone()
    person_id = row['id'] if row else conn.execute(
        'INSERT INTO persons (name) VALUES (?)', (name,)).lastrowid
    saved = 0; errors = []
    for pf in photos:
        fn = pf.filename.lower()
        if not (fn.endswith('.jpg') or fn.endswith('.jpeg') or fn.endswith('.png')):
            errors.append(f'{pf.filename}: 格式不支持'); continue
        try:
            fb = pf.read(); nparr = np.frombuffer(fb, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: errors.append(f'{pf.filename}: 无法解码'); continue
            emb, score = extract_face_embedding(img)
            if emb is None: errors.append(f'{pf.filename}: 未检测到人脸 (det={score:.2f})'); continue
            sn = "".join(c for c in name if c.isalnum() or c in ('_', '-', '一-鿿'))
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            pfn = f"{sn}_{ts}_{saved}.jpg"; pp = os.path.join('static', 'uploads', pfn)
            cv2.imwrite(pp, img)
            conn.execute('INSERT INTO face_embeddings (person_id, embedding_blob, photo_path, det_score) '
                         'VALUES (?, ?, ?, ?)', (person_id, emb.tobytes(), f'/static/uploads/{pfn}', score))
            saved += 1
        except Exception as e:
            errors.append(f'{pf.filename}: {str(e)}')
    conn.commit(); conn.close()
    if saved == 0:
        return jsonify({'ok': False, 'error': f'全部失败: {"; ".join(errors[-3:])}'}), 400
    return jsonify({'ok': True, 'message': f'{name} 注册成功！已保存 {saved} 个面部嵌入',
                    'person_id': person_id, 'saved': saved, 'errors': errors[:3]})


@app.route('/manage')
def manage():
    return render_template('manage.html')


@app.route('/history')
def history():
    return render_template('history.html')


# ============================================================
# Routes — Streaming & WebSocket
# ============================================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/<int:cam_id>')
def video_feed_cam(cam_id):
    if cam_id < 0 or cam_id >= len(CAMERAS):
        return 'Camera not found', 404
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_frame')
def capture_frame():
    """Capture from camera 0 frame queue (used by registration)."""
    fq = frame_queues.get(0)
    if fq is None:
        return Response(b'', status=503)
    try:
        frame = fq.get(timeout=3)
    except queue.Empty:
        return Response(b'', status=503)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/camera/<int:cam_id>/toggle', methods=['POST'])
def api_toggle_camera(cam_id):
    camera_enabled[cam_id] = not camera_enabled.get(cam_id, True)
    return jsonify({'ok': True, 'enabled': camera_enabled[cam_id]})


@app.route('/api/camera/<int:cam_id>/rename', methods=['POST'])
def api_rename_camera(cam_id):
    data = request.get_json(force=True) or {}
    new_name = (data.get('name') or '').strip()
    if not new_name:
        return jsonify({'ok': False, 'error': '名称不能为空'}), 400
    camera_names[str(cam_id)] = new_name
    _save_camera_config(CAMERAS, camera_names)
    return jsonify({'ok': True, 'name': new_name})


@app.route('/cameras')
def cameras_page():
    return render_template('cameras.html')


@app.route('/api/cameras/scan', methods=['POST'])
def api_cameras_scan():
    """Auto-discover ONVIF cameras on the local network."""
    results = []

    # Common Hikvision default passwords
    DEFAULT_CREDS = [
        ('admin', 'admin12345'),
        ('admin', 'admin'),
        ('admin', '12345'),
        ('admin', 'Hik12345'),
        ('admin', 'hikvision'),
        ('admin', 'password'),
    ]

    try:
        from onvif import ONVIFCamera

        # WS-Discovery scan
        from wsdiscovery import WSDiscovery
        wsd = WSDiscovery()
        wsd.start()
        services = wsd.searchServices(timeout=5)
        wsd.stop()

        for svc in services:
            xaddrs = svc.getXAddrs()
            if not xaddrs:
                continue
            addr = xaddrs[0]
            ip = addr.split('://')[1].split(':')[0] if '://' in addr else addr.split(':')[0]

            # Try common credentials
            found_cred = None
            for user, pwd in DEFAULT_CREDS:
                try:
                    cam = ONVIFCamera(ip, 80, user, pwd)
                    info = cam.devicemgmt.GetDeviceInformation()
                    found_cred = (user, pwd)
                    mfr = info.Manufacturer
                    model = info.Model
                    # Get RTSP URL
                    profiles = cam.media.GetProfiles()
                    rtsp = None
                    if profiles:
                        try:
                            uri = cam.media.GetStreamUri({
                                'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                                'ProfileToken': profiles[0].token,
                            })
                            rtsp = uri.Uri.replace('//', f'//{user}:{pwd}@')
                        except:
                            pass
                    results.append({
                        'ip': ip, 'manufacturer': mfr, 'model': model,
                        'rtsp': rtsp, 'user': user, 'password': pwd,
                        'found_cred': True,
                    })
                    break
                except Exception:
                    continue

            if not found_cred:
                # Could discover but not login — still list it
                results.append({
                    'ip': ip, 'manufacturer': 'Unknown (need password)',
                    'model': '', 'rtsp': None, 'user': 'admin',
                    'password': '', 'found_cred': False,
                })

    except ImportError:
        return jsonify({'ok': False, 'error': 'ONVIF library not installed (onvif-zeep)', 'results': []}), 500
    except Exception as e:
        return jsonify({'ok': True, 'results': results, 'note': f'Partial scan: {str(e)[:200]}'})

    return jsonify({'ok': True, 'results': results, 'count': len(results)})


@app.route('/api/cameras', methods=['GET', 'POST', 'DELETE'])
def api_cameras():
    """GET: list all. POST: add. DELETE: remove (with id param)."""
    global CAMERAS
    if request.method == 'GET':
        return jsonify([{
            'id': c['id'], 'source': c['source'],
            'name': camera_names.get(str(c['id']), f'摄像头{c["id"]+1}'),
            'enabled': camera_enabled.get(c['id'], True),
        } for c in CAMERAS])

    elif request.method == 'POST':
        data = request.get_json(force=True) or {}
        source = data.get('source')
        if source is None:
            return jsonify({'ok': False, 'error': 'source is required (int for USB, str for RTSP)'}), 400
        # Find next available id
        used_ids = {c['id'] for c in CAMERAS}
        new_id = 0
        while new_id in used_ids:
            new_id += 1
        CAMERAS.append({'id': new_id, 'source': source})
        _save_camera_config(CAMERAS, camera_names)
        return jsonify({'ok': True, 'message': f'摄像头 {new_id} 已添加，重启后生效', 'id': new_id})

    elif request.method == 'DELETE':
        cam_id = request.args.get('id', type=int)
        if cam_id is None:
            return jsonify({'ok': False, 'error': '?id= required'}), 400
        idx = next((i for i, c in enumerate(CAMERAS) if c['id'] == cam_id), None)
        if idx is None:
            return jsonify({'ok': False, 'error': '摄像头不存在'}), 404
        CAMERAS.pop(idx)
        camera_names.pop(str(cam_id), None)
        _save_camera_config(CAMERAS, camera_names)
        return jsonify({'ok': True, 'message': f'摄像头 {cam_id} 已删除，重启后生效'})


@app.route('/events')
def sse_events():
    def stream():
        while True:
            try:
                data = fall_queue.get(timeout=1)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(stream(), mimetype='text/event-stream')


@socketio.on('connect')
def on_connect():
    print(f"[WS] Client connected")


@socketio.on('disconnect')
def on_disconnect():
    print(f"[WS] Client disconnected")


# ============================================================
# Routes — REST API
# ============================================================
@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'ok',
        'cameras': [{
            'id': c['id'],
            'name': camera_names.get(str(c['id']), '摄像头' + str(c['id'] + 1)),
            'fps': current_fps_list.get(c['id'], 0),
            'persons': person_count_list.get(c['id'], 0),
            'p_fall': last_p_fall_list.get(c['id'], 0),
        } for c in CAMERAS],
        'name': recognized_name,
        'ai_enabled': cfg.AI_ENABLED and ai_toggle,
        'ai_toggle': ai_toggle,
    })


@app.route('/api/toggle_ai', methods=['POST'])
def api_toggle_ai():
    global ai_toggle
    ai_toggle = not ai_toggle
    return jsonify({'ok': True, 'ai_toggle': ai_toggle})


@app.route('/api/events')
def api_events():
    limit = request.args.get('limit', 100, type=int)
    conn = get_db()
    rows = conn.execute(
        'SELECT id, elder_name, confidence, screenshot, report, permanent, created_at '
        'FROM events ORDER BY id DESC LIMIT ?', (min(limit, 500),)).fetchall()
    conn.close()
    return jsonify([{
        'id': r['id'], 'elder_name': r['elder_name'], 'confidence': r['confidence'],
        'screenshot': r['screenshot'],
        'report': r['report'] if r['report'] else '',
        'has_report': bool(r['report']),
        'report_summary': r['report'][:100] if r['report'] else '',
        'permanent': bool(r['permanent']),
        'created_at': r['created_at'],
    } for r in rows])


@app.route('/api/events/<int:event_id>', methods=['DELETE'])
def api_delete_event(event_id):
    conn = get_db()
    row = conn.execute('SELECT id, screenshot FROM events WHERE id = ?', (event_id,)).fetchone()
    if row is None: conn.close(); return jsonify({'ok': False, 'error': '事件不存在'}), 404
    if row['screenshot']:
        fp = os.path.join(os.path.dirname(__file__), row['screenshot'].lstrip('/'))
        if os.path.isfile(fp): os.remove(fp)
    conn.execute('DELETE FROM events WHERE id = ?', (event_id,))
    conn.commit(); conn.close()
    return jsonify({'ok': True, 'message': f'事件 #{event_id} 已删除'})


@app.route('/api/events/delete_all', methods=['POST'])
def api_delete_all_events():
    conn = get_db()
    rows = conn.execute('SELECT screenshot FROM events').fetchall()
    for r in rows:
        if r['screenshot']:
            fp = os.path.join(os.path.dirname(__file__), r['screenshot'].lstrip('/'))
            if os.path.isfile(fp): os.remove(fp)
    conn.execute('DELETE FROM events')
    conn.commit(); conn.close()
    return jsonify({'ok': True, 'message': '所有跌倒事件已清空'})


@app.route('/api/events/<int:event_id>/permanent', methods=['POST'])
def api_toggle_permanent(event_id):
    conn = get_db()
    row = conn.execute('SELECT id, permanent FROM events WHERE id = ?', (event_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({'ok': False, 'error': '事件不存在'}), 404
    new_val = 0 if row['permanent'] else 1
    conn.execute('UPDATE events SET permanent = ? WHERE id = ?', (new_val, event_id))
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'permanent': bool(new_val),
                    'message': '已标记为永久保存' if new_val else '已取消永久保存'})


@app.route('/api/events/<int:event_id>')
def api_event_detail(event_id):
    conn = get_db()
    row = conn.execute(
        'SELECT id, elder_name, confidence, screenshot, report, created_at '
        'FROM events WHERE id = ?', (event_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({'ok': False, 'error': '事件不存在'}), 404
    return jsonify({
        'id': row['id'], 'elder_name': row['elder_name'], 'confidence': row['confidence'],
        'screenshot': row['screenshot'],
        'report': row['report'] if row['report'] else 'AI 分析中...',
        'created_at': row['created_at'],
    })


@app.route('/api/latest_report')
def api_latest_report():
    conn = get_db()
    row = conn.execute(
        "SELECT id, elder_name, confidence, report, created_at FROM events "
        "WHERE report != '' ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        row2 = conn.execute(
            'SELECT id, elder_name, confidence, created_at FROM events ORDER BY id DESC LIMIT 1'
        ).fetchone()
        conn.close()
        if row2:
            return jsonify({'id': row2['id'], 'elder_name': row2['elder_name'],
                            'confidence': row2['confidence'], 'created_at': row2['created_at'],
                            'report': 'AI 分析中，请稍候...', 'pending': True})
        return jsonify(None)
    conn.close()
    return jsonify({'id': row['id'], 'elder_name': row['elder_name'],
                    'confidence': row['confidence'], 'report': row['report'],
                    'created_at': row['created_at'], 'pending': False})


@app.route('/api/register_face', methods=['POST'])
def api_register_face():
    """Register face via REST API: {"name":"李四","image_base64":"..."}"""
    if face_app is None:
        return jsonify({'ok': False, 'error': 'InsightFace 未加载'}), 500
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip()
    img_b64 = (data.get('image_base64') or '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'name 字段为空'}), 400
    if not img_b64:
        return jsonify({'ok': False, 'error': 'image_base64 字段为空'}), 400
    try:
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'ok': False, 'error': '无法解码 base64 图片'}), 400
        emb, score = extract_face_embedding(img)
        if emb is None:
            return jsonify({'ok': False, 'error': f'未检测到人脸 (det={score:.2f})'}), 400

        conn = get_db()
        row = conn.execute('SELECT id FROM persons WHERE name = ?', (name,)).fetchone()
        person_id = row['id'] if row else conn.execute(
            'INSERT INTO persons (name) VALUES (?)', (name,)).lastrowid
        sn = "".join(c for c in name if c.isalnum() or c in ('_', '-', '一-鿿'))
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pfn = f"{sn}_{ts}_api.jpg"; pp = os.path.join('static', 'uploads', pfn)
        cv2.imwrite(pp, img)
        conn.execute('INSERT INTO face_embeddings (person_id, embedding_blob, photo_path, det_score) '
                     'VALUES (?, ?, ?, ?)', (person_id, emb.tobytes(), f'/static/uploads/{pfn}', score))
        conn.commit(); conn.close()
        return jsonify({'ok': True, 'message': f'{name} 已注册', 'person_id': person_id,
                        'det_score': round(score, 3), 'photo_url': f'/static/uploads/{pfn}'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/faces')
def api_faces():
    conn = get_db()
    rows = conn.execute('''SELECT p.id, p.name, p.created_at, COUNT(e.id) AS embedding_count
        FROM persons p LEFT JOIN face_embeddings e ON e.person_id = p.id
        GROUP BY p.id ORDER BY p.id DESC''').fetchall()
    result = []
    for r in rows:
        photos = conn.execute(
            'SELECT photo_path FROM face_embeddings WHERE person_id = ? ORDER BY id', (r['id'],)
        ).fetchall()
        result.append({
            'id': r['id'], 'name': r['name'],
            'embedding_count': r['embedding_count'],
            'created_at': r['created_at'],
            'photos': [p['photo_path'] for p in photos],
        })
    conn.close()
    return jsonify(result)


@app.route('/api/faces/<int:face_id>', methods=['DELETE'])
def api_delete_face(face_id):
    conn = get_db()
    row = conn.execute('SELECT id, name FROM persons WHERE id = ?', (face_id,)).fetchone()
    if row is None: conn.close(); return jsonify({'ok': False, 'error': '记录不存在'}), 404
    photos = conn.execute('SELECT photo_path FROM face_embeddings WHERE person_id = ?',
                          (face_id,)).fetchall()
    for p in photos:
        if p['photo_path']:
            fp = os.path.join(os.path.dirname(__file__), p['photo_path'].lstrip('/'))
            if os.path.isfile(fp): os.remove(fp)
    conn.execute('DELETE FROM persons WHERE id = ?', (face_id,))
    conn.commit(); conn.close()
    return jsonify({'ok': True, 'message': f"已删除 {row['name']} 及其所有面部记录"})


@app.route('/api/faces/<int:face_id>/photo', methods=['PUT'])
def api_add_face_photo(face_id):
    """Add face photos to existing person (multi-file or multi-blob)."""
    conn = get_db()
    row = conn.execute('SELECT id, name FROM persons WHERE id = ?', (face_id,)).fetchone()
    if row is None: conn.close(); return jsonify({'ok': False, 'error': '人员记录不存在'}), 404
    if face_app is None: conn.close(); return jsonify({'ok': False, 'error': 'InsightFace 未加载'}), 500

    photos = request.files.getlist('photo')
    photos = [f for f in photos if f and f.filename]
    if len(photos) == 0:
        conn.close(); return jsonify({'ok': False, 'error': '请上传至少一张照片'}), 400

    saved = 0; errors = []
    for pf in photos:
        fn = pf.filename.lower()
        if not (fn.endswith('.jpg') or fn.endswith('.jpeg') or fn.endswith('.png')):
            errors.append(f'{pf.filename}: 格式不支持'); continue
        try:
            fb = pf.read(); nparr = np.frombuffer(fb, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: errors.append(f'{pf.filename}: 无法解码'); continue
            emb, score = extract_face_embedding(img)
            if emb is None: errors.append(f'{pf.filename}: 未检测到人脸 (det={score:.2f})'); continue
            sn = "".join(c for c in row['name'] if c.isalnum() or c in ('_', '-', '一-鿿'))
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            pfn = f"{sn}_{ts}_{saved}.jpg"
            pp = os.path.join('static', 'uploads', pfn); cv2.imwrite(pp, img)
            conn.execute('INSERT INTO face_embeddings (person_id, embedding_blob, photo_path, det_score) '
                         'VALUES (?, ?, ?, ?)', (face_id, emb.tobytes(), f'/static/uploads/{pfn}', score))
            saved += 1
        except Exception as e:
            errors.append(f'{pf.filename}: {str(e)}')
    conn.commit(); conn.close()
    if saved == 0:
        return jsonify({'ok': False, 'error': f'全部失败: {"; ".join(errors[-3:])}'}), 400
    msg = f"已为 {row['name']} 添加 {saved} 个面部嵌入"
    if errors: msg += f'（{len(errors)} 张跳过）'
    return jsonify({'ok': True, 'message': msg, 'saved': saved, 'errors': errors[:3]})


# ============================================================
# Test mode
# ============================================================
@app.route('/test', methods=['GET', 'POST'])
def test_page():
    global test_video_path
    if request.method == 'POST':
        vf = request.files.get('video')
        if not vf or vf.filename == '':
            return jsonify({'ok': False, 'error': '请选择视频文件'}), 400
        vp = os.path.join('static', 'test_video.mp4')
        vf.save(vp)
        with test_video_lock:
            test_video_path = vp
            # Save and disable all cameras
            test_saved_cam_states.clear()
            for c in CAMERAS:
                cid = c['id']
                test_saved_cam_states[cid] = camera_enabled.get(cid, True)
                camera_enabled[cid] = False
        print(f"[Test] Video uploaded, cameras disabled")
        return jsonify({'ok': True, 'message': '视频已加载，摄像头已关闭'})
    return render_template('test.html')


@app.route('/test/reset')
def test_reset():
    global test_video_path
    with test_video_lock:
        test_video_path = None
        # Restore saved camera states
        for cid, state in test_saved_cam_states.items():
            camera_enabled[cid] = state
        test_saved_cam_states.clear()
    print("[Test] Reset, cameras restored")
    return jsonify({'ok': True, 'message': '测试结束，摄像头已恢复'})


@app.route('/test_feed')
def test_feed():
    """MJPEG stream for uploaded test video with detection overlay."""
    def gen():
        global test_video_path
        while alive.is_set():
            vp = None
            with test_video_lock:
                vp = test_video_path
            if vp is None:
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(blank, '请上传测试视频', (80, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                _, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            cap = cv2.VideoCapture(vp)
            while alive.is_set():
                vp2 = None
                with test_video_lock:
                    vp2 = test_video_path
                if vp2 != vp:
                    break  # video changed or reset

                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Run detection on frame
                results = model(frame, imgsz=YOLO_IMGSZ, conf=0.5, verbose=False)
                res = results[0]
                dets = all_persons(res)

                # Match tracks
                tracks = _match_or_create_tracks(dets, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

                max_p_fall = 0.0; any_fall = False
                for tid, t in tracks.items():
                    kp, kp_conf = t.get('kp'), t.get('kp_conf')
                    if kp is None or kp_conf is None:
                        continue
                    # Per-person fall check
                    sh = list(hip_history); sa = list(angle_history)
                    hip_history.clear(); hip_history.extend(t['hip_history'])
                    angle_history.clear(); angle_history.extend(t['angle_history'])
                    is_fall, info = check_fall(kp, kp_conf)
                    t['hip_history'] = deque(hip_history, maxlen=5)
                    t['angle_history'] = deque(angle_history, maxlen=5)
                    hip_history.clear(); hip_history.extend(sh)
                    angle_history.clear(); angle_history.extend(sa)
                    pf = info.get('p_fall', 0) if info else 0
                    if pf > max_p_fall: max_p_fall = pf
                    if is_fall:
                        t['fall_counter'] += 1
                    else:
                        t['fall_counter'] = 0
                    if t['fall_counter'] >= FALL_CONSECUTIVE_FRAMES:
                        any_fall = True
                        t['fall_counter'] = 0

                    color = (0, 0, 255) if t['fall_counter'] > 0 else t.get('color', (0, 255, 0))
                    for a, b in SKELETON_EDGES:
                        if kp_conf[a] > 0.5 and kp_conf[b] > 0.5:
                            cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                                     (int(kp[b][0]), int(kp[b][1])), color, 2)
                    for i in range(len(kp)):
                        if kp_conf[i] > 0.5:
                            cv2.circle(frame, (int(kp[i][0]), int(kp[i][1])), 4, color, -1)

                # HUD
                cv2.putText(frame, f'Test Mode | Persons: {len(tracks)} | P_FALL: {max_p_fall:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if max_p_fall >= FALL_PROB_THRESHOLD else (0, 255, 0), 2)
                if any_fall:
                    cv2.putText(frame, 'FALL DETECTED!', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    # Test mode: save to DB + trigger AI if enabled
                    now = time.time()
                    if not hasattr(gen, '_last_fall') or (now - gen._last_fall) > FALL_COOLDOWN_SECONDS:
                        gen._last_fall = now
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        fn = os.path.join('static', 'falls', f'test_fall_{ts}.jpg')
                        cv2.imwrite(fn, frame)
                        conn = get_db()
                        cur = conn.execute('INSERT INTO events (elder_name, confidence, screenshot) VALUES (?, ?, ?)',
                                           ('测试跌倒', max_p_fall, f'/static/falls/test_fall_{ts}.jpg'))
                        eid = cur.lastrowid
                        conn.commit(); conn.close()
                        if cfg.AI_ENABLED and ai_toggle:
                            ai_executor.submit(analyze_fall_image, f'/static/falls/test_fall_{ts}.jpg', eid)
                            print(f'[Test] Fall saved + AI queued event #{eid}')
                _, buf = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

            cap.release()
            with test_video_lock:
                if test_video_path == vp:
                    test_video_path = None
                    for cid, state in test_saved_cam_states.items():
                        camera_enabled[cid] = state
                    test_saved_cam_states.clear()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  跌倒监测微服务 v2.0 — 多摄像头")
    print("  http://localhost:5001")
    for c in CAMERAS:
        name = camera_names.get(str(c['id']), f'摄像头{c["id"]+1}')
        print(f"  /video_feed/{c['id']} → {name} (source={c['source']})")
    print("=" * 55)

    # Init per-camera state
    for c in CAMERAS:
        cid = c['id']
        frame_queues[cid] = queue.Queue(maxsize=60)
        latest_detections[cid] = {'kp_xy': None, 'kp_conf': None, 'is_fall': False, 'tracks': {}}
        detection_locks[cid] = threading.Lock()
        current_fps_list[cid] = 0
        person_count_list[cid] = 0
        last_p_fall_list[cid] = 0

    # Start detection threads per camera
    for c in CAMERAS:
        t = threading.Thread(target=detection_worker, args=(c['id'],), daemon=True,
                             name=f'detection-{c["id"]}')
        t.start()
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name='cleanup')
    cleanup_thread.start()
    print(f"[System] {len(CAMERAS)} camera(s) + Cleanup threads started")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
