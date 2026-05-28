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
# Models & Camera
# ============================================================
model = YOLO('yolov8n-pose.pt')
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 60)

video_source = 'camera'
video_source_lock = threading.Lock()

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
last_p_fall = 0.0
frame_queue = queue.Queue(maxsize=60)
alive = threading.Event()
alive.set()

latest_detection = {'kp_xy': None, 'kp_conf': None, 'is_fall': False}
detection_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

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
def detection_worker():
    global recognized_name, person_count, fall_counter, last_fall_time
    global last_face_match_frame, held_face_name, last_p_fall
    det_frame_count = 0
    warn_hold = 0

    while alive.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        with frame_lock:
            latest_frame = frame.copy()
        det_frame_count += 1

        # Face recognition
        if face_app is not None and det_frame_count % FACE_RECOGNITION_INTERVAL == 0:
            name, _ = recognize_face(frame, det_frame_count)
            if name is not None:
                last_face_match_frame = det_frame_count; held_face_name = name
        if held_face_name is not None:
            if det_frame_count - last_face_match_frame > FACE_NAME_HOLD_FRAMES:
                held_face_name = None
            with state_lock:
                recognized_name = held_face_name

        # YOLO pose detection
        if det_frame_count % DETECTION_INTERVAL == 0:
            results = model(frame, imgsz=YOLO_IMGSZ, conf=0.5, verbose=False)
            result = results[0]
            detections = all_persons(result)

            with tracker_lock:
                tracks = _match_or_create_tracks(detections, det_frame_count)
                with state_lock:
                    person_count = len(tracks)

            # Process each tracked person independently
            max_p_fall = 0.0
            any_is_fall = False
            for tid, t in tracks.items():
                kp = t['kp']; kp_conf = t['kp_conf']
                # Override check_fall's globals with per-person histories
                saved_hip = list(hip_history)
                saved_angle = list(angle_history)
                # Temporarily swap in per-person histories
                hip_history.clear(); hip_history.extend(t['hip_history'])
                angle_history.clear(); angle_history.extend(t['angle_history'])

                is_fall_now, info = check_fall(kp, kp_conf)

                # Save back per-person histories
                t['hip_history'] = deque(hip_history, maxlen=5)
                t['angle_history'] = deque(angle_history, maxlen=5)
                hip_history.clear(); hip_history.extend(saved_hip)
                angle_history.clear(); angle_history.extend(saved_angle)

                p_fall_val = info.get('p_fall', 0.0) if info else 0.0
                if p_fall_val > max_p_fall:
                    max_p_fall = p_fall_val

                # Per-person fall state machine
                if is_fall_now:
                    t['fall_counter'] += 1
                else:
                    t['fall_counter'] = 0

                if t['fall_counter'] >= FALL_CONSECUTIVE_FRAMES:
                    any_is_fall = True
                    t['fall_counter'] = 0

                if info:
                    t['last_p_fall'] = p_fall_val

            if max_p_fall > 0:
                last_p_fall = max_p_fall

            # Level 1 warning (any person with elevated P_FALL)
            if WARN_PROB_THRESHOLD <= max_p_fall < FALL_PROB_THRESHOLD:
                warn_hold = WARN_HOLD_FRAMES
            else:
                warn_hold = max(0, warn_hold - 1)
            if warn_hold > 0 and not any_is_fall:
                broadcast_alert({'type': 'warning', 'level': 1,
                                 'message': '姿态不稳', 'p_fall': max_p_fall})

            # Level 2: find which person fell and trigger
            now = time.time()
            for tid, t in tracks.items():
                if t['fall_counter'] >= FALL_CONSECUTIVE_FRAMES:
                    if (now - last_fall_time) > FALL_COOLDOWN_SECONDS:
                        last_fall_time = now
                        # Use person's name if recognized, else "陌生人"
                        pname = t.get('name') or '陌生人'
                        with state_lock:
                            saved_name = recognized_name
                            recognized_name = pname
                        trigger_fall_event(frame, {'p_fall': t.get('last_p_fall', 0.75),
                                                    'angle': 0, 'velocity': 0,
                                                    'ar': 0, 'angle_accel': 0,
                                                    'p_angle': 0, 'p_vel': 0, 'p_ar': 0,
                                                    'p_accel': 0, 'p_hf': 0})
                        with state_lock:
                            recognized_name = saved_name
                        t['fall_counter'] = 0

            # Publish latest detection for drawing (primary person = highest P_FALL)
            best_t = max(tracks.values(), key=lambda t: t.get('last_p_fall', 0)) if tracks else None
            with detection_lock:
                latest_detection['kp_xy'] = best_t['kp'] if best_t else None
                latest_detection['kp_conf'] = best_t['kp_conf'] if best_t else None
                latest_detection['is_fall'] = any_is_fall
                latest_detection['tracks'] = tracks
        else:
            time.sleep(0.002)


# ============================================================
# MJPEG Stream Generator (reads camera, feeds detection, draws overlays)
# ============================================================
def generate_frames():
    global current_fps, fps_timer, fps_frame_count
    while alive.is_set():
        with video_source_lock:
            src = video_source

        if src == 'camera':
            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue
        else:
            if not hasattr(generate_frames, '_vid_cap') or generate_frames._vid_cap is None:
                try:
                    generate_frames._vid_cap = cv2.VideoCapture(src)
                except:
                    time.sleep(0.5)
                    continue
            success, frame = generate_frames._vid_cap.read()
            if not success:
                generate_frames._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        fps_frame_count += 1
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            current_fps = round(fps_frame_count / elapsed, 1)
            fps_frame_count = 0
            fps_timer = now

        # Feed detection worker
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

        # Draw all tracked persons
        with detection_lock:
            tracks = latest_detection.get('tracks', {})

        for tid, t in tracks.items():
            kp = t.get('kp'); kp_conf = t.get('kp_conf')
            if kp is None or kp_conf is None:
                continue
            # Per-person fall state for coloring
            is_fall_person = t.get('fall_counter', 0) > 0
            color = t.get('color', (0, 255, 0))
            pname = t.get('name') or f'ID:{tid}'
            p_fall_val = t.get('last_p_fall', 0)

            # Draw skeleton with person color
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

            # Name + P_FALL above head
            bx, by = int(t['bbox'][0]), int(t['bbox'][1])
            label = f'{pname} | {p_fall_val:.2f}'
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = max(0, bx + int((t['bbox'][2] - t['bbox'][0] - lw) / 2))
            ly = max(20, by - 8)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # HUD
        cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        p_color = (0, 255, 0) if last_p_fall < WARN_PROB_THRESHOLD else (
            (0, 0, 255) if last_p_fall >= FALL_PROB_THRESHOLD else (0, 165, 255))
        cv2.putText(frame, f"P_FALL: {last_p_fall:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)
        cv2.putText(frame, f"Mode: {src[:20]}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)

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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_frame')
def capture_frame():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        success, frame = camera.read()
        if not success or frame is None:
            return Response(b'', status=503)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype='image/jpeg')


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
        'status': 'ok', 'fps': current_fps, 'persons': person_count,
        'name': recognized_name, 'p_fall': last_p_fall, 'mode': video_source,
        'ai_enabled': cfg.AI_ENABLED and ai_toggle,
        'ai_toggle': ai_toggle, 'uptime': round(time.time() - fps_timer, 0),
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
    if request.method == 'POST':
        vf = request.files.get('video')
        if not vf or vf.filename == '':
            return jsonify({'ok': False, 'error': '请选择视频文件'}), 400
        vp = os.path.join('static', 'test_video.mp4')
        vf.save(vp)
        with video_source_lock:
            global video_source; video_source = vp
        if hasattr(generate_frames, '_vid_cap') and generate_frames._vid_cap:
            generate_frames._vid_cap.release()
            generate_frames._vid_cap = None
        print(f"[Test] Video uploaded: {vp}")
        return jsonify({'ok': True, 'message': '视频已加载，刷新监控页面查看'})
    return render_template('test.html')


@app.route('/test/reset')
def test_reset():
    with video_source_lock:
        global video_source; video_source = 'camera'
    if hasattr(generate_frames, '_vid_cap') and generate_frames._vid_cap:
        generate_frames._vid_cap.release()
        generate_frames._vid_cap = None
    print("[Test] Reset to camera mode")
    return jsonify({'ok': True, 'message': '已切回摄像头模式'})


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  跌倒监测微服务 v2.0")
    print("  http://localhost:5001")
    print("  WebSocket: ws://localhost:5001/ws")
    print("  API docs: 见 README.md")
    print("=" * 55)

    det_thread = threading.Thread(target=detection_worker, daemon=True, name='detection')
    det_thread.start()
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name='cleanup')
    cleanup_thread.start()
    print("[System] Detection + Cleanup threads started")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
