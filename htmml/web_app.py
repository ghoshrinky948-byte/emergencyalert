"""
Emergency camera dashboard (web version).

Detects:
  1. Fire / smoke  - built-in colour+flicker detector (no model file needed)
                     and/or a custom YOLO fire/smoke model (models/fire.pt).
  2. Person down and not moving ("possible unconscious") - YOLO pose model
     finds people, then the app checks they are lying down AND motionless
     for a set number of seconds.

Run:   python web_app.py
Open:  http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SNAPSHOT_DIR = BASE_DIR / "alerts"
SNAPSHOT_DIR.mkdir(exist_ok=True)

DEFAULT_PERSON_MODEL = "yolo11n-pose.pt"   # downloads automatically on first use
FALLBACK_PERSON_MODEL = "yolo11n.pt"       # boxes only, used if the pose model can't load
FIRE_WORDS = ("fire", "smoke", "flame")
MAX_ALERTS = 200

ALERT_LABELS = {
    "FIRE": "Fire or smoke",
    "POSSIBLE_UNCONSCIOUS": "Person down and not moving",
    "TEST": "Test notification",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mask_credentials(text: str) -> str:
    return re.sub(r"//[^/@\s]+@", "//***@", text)


def resolve_path(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    alt = BASE_DIR / p
    return str(alt) if alt.exists() else p


def names_map(result):
    names = result.names
    return names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Settings:
    source: str
    person_model: str
    fire_model: str
    builtin_fire: bool
    fire_sensitivity: str
    confidence: float
    down_seconds: int
    ntfy_server: str
    ntfy_topic: str
    cooldown: int


# --------------------------------------------------------------------------- #
# Person-down analysis
# --------------------------------------------------------------------------- #
def is_lying(box, kpts) -> bool:
    """True if the body is closer to horizontal than vertical.

    Uses shoulder/hip keypoints when the pose model is available,
    otherwise falls back to the bounding-box shape.
    """
    x1, y1, x2, y2 = box
    if kpts is not None:
        sh = [kpts[i] for i in (5, 6) if kpts[i][2] > 0.3]
        hp = [kpts[i] for i in (11, 12) if kpts[i][2] > 0.3]
        if sh and hp:
            sx, sy = np.mean([p[0] for p in sh]), np.mean([p[1] for p in sh])
            hx, hy = np.mean([p[0] for p in hp]), np.mean([p[1] for p in hp])
            return abs(sx - hx) > 1.2 * abs(sy - hy)
    return (x2 - x1) > 1.25 * (y2 - y1)


@dataclass
class Track:
    tid: int
    box: tuple
    last_seen: float
    lying_since: float | None = None
    last_lying: float = 0.0
    still_since: float | None = None
    ref_roi: np.ndarray | None = None
    ref_time: float = 0.0
    ref_center: tuple = (0.0, 0.0)
    lying: bool = False
    still_for: float = 0.0
    lying_for: float = 0.0


class PersonTracker:
    """Very small tracker: matches people between frames and measures how long
    each one has been lying down and how little they have moved."""

    SAMPLE_EVERY = 0.5      # seconds between motion samples
    STILL_DIFF = 5.0        # mean grey-level change under this = "not moving"
    STILL_SHIFT = 0.04      # centre movement (fraction of box size) under this = "not moving"

    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def _match(self, box, used):
        best, best_score = None, 0.0
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        diag = max(1.0, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)
        for t in self.tracks.values():
            if t.tid in used:
                continue
            score = iou(box, t.box)
            if score < 0.15:  # a lying/standing change shrinks IoU, so also allow close centres
                tx, ty = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < 0.6 * diag:
                    score = 0.15
            if score > best_score:
                best, best_score = t, score
        return best

    def update(self, detections, gray, now, down_seconds):
        """detections: list of (box, kpts_or_None). Returns list of (Track, is_alarm)."""
        used, out = set(), []
        h, w = gray.shape[:2]
        for box, kpts in detections:
            t = self._match(box, used)
            if t is None:
                t = Track(self.next_id, box, now)
                self.tracks[t.tid] = t
                self.next_id += 1
            used.add(t.tid)
            t.box, t.last_seen = box, now

            # lying state, with 1 s of tolerance so one bad frame doesn't reset it
            if is_lying(box, kpts):
                t.last_lying = now
                if t.lying_since is None:
                    t.lying_since = now
            elif t.lying_since is not None and now - t.last_lying > 1.0:
                t.lying_since, t.still_since = None, None
            t.lying = t.lying_since is not None
            t.lying_for = now - t.lying_since if t.lying else 0.0

            # motion sample
            x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                              min(w, int(box[2])), min(h, int(box[3])))
            if x2 - x1 > 8 and y2 - y1 > 8:
                roi = cv2.GaussianBlur(cv2.resize(gray[y1:y2, x1:x2], (48, 48)), (5, 5), 0)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                if t.ref_roi is None:
                    t.ref_roi, t.ref_time, t.ref_center = roi, now, center
                elif now - t.ref_time >= self.SAMPLE_EVERY:
                    diff = float(np.mean(cv2.absdiff(roi, t.ref_roi)))
                    shift = ((center[0] - t.ref_center[0]) ** 2 + (center[1] - t.ref_center[1]) ** 2) ** 0.5 / diag
                    if diff < self.STILL_DIFF and shift < self.STILL_SHIFT:
                        if t.still_since is None:
                            t.still_since = now
                    else:
                        t.still_since = None
                    t.ref_roi, t.ref_time, t.ref_center = roi, now, center
            t.still_for = now - t.still_since if (t.lying and t.still_since is not None) else 0.0
            out.append((t, t.lying and t.still_for >= down_seconds))

        for tid in [k for k, t in self.tracks.items() if now - t.last_seen > 2.0]:
            del self.tracks[tid]
        return out


# --------------------------------------------------------------------------- #
# Built-in fire detector (no model needed)
# --------------------------------------------------------------------------- #
FIRE_PRESETS = {
    #            min saturation, min brightness, min area, min flicker, seconds to confirm
    "low":    dict(sat=140, val=210, min_area=0.003,  flicker=0.12, confirm=1.2),
    "medium": dict(sat=110, val=190, min_area=0.001,  flicker=0.07, confirm=0.9),
    "high":   dict(sat=80,  val=170, min_area=0.0005, flicker=0.04, confirm=0.6),
}


class ColorFireDetector:
    """Looks for bright orange/yellow regions whose shape keeps changing (flicker).

    Flicker is measured against a frame about 0.25 s earlier, so it works the
    same at 5 fps or 30 fps. Areas inside detected people are ignored, which
    avoids false alarms from skin and orange clothing. It is a heuristic:
    bright orange lights can still fool it, so treat alerts as 'please check'.
    """

    MAX_AREA = 0.60
    REF_GAP = 0.25          # seconds between the two frames that are compared

    def __init__(self, sensitivity: str = "medium"):
        self.p = FIRE_PRESETS.get(sensitivity, FIRE_PRESETS["medium"])
        self.hist: collections.deque = collections.deque()
        self.since = None
        self.last_candidate = 0.0
        self.last_area = 0.0
        self.last_flicker = 0.0

    def process(self, frame, now, ignore_boxes=()):
        h, w = frame.shape[:2]
        scale = 320 / w if w > 320 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, self.p["sat"], self.p["val"]), (35, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        for (x1, y1, x2, y2) in ignore_boxes:
            mask[max(0, int(y1 * scale)):int(y2 * scale) + 1, max(0, int(x1 * scale)):int(x2 * scale) + 1] = 0

        nz = int(np.count_nonzero(mask))
        area = nz / mask.size

        # reference mask: newest stored frame that is at least ~REF_GAP old
        while len(self.hist) > 1 and now - self.hist[1][0] >= self.REF_GAP:
            self.hist.popleft()
        ref = None
        if self.hist and now - self.hist[0][0] >= self.REF_GAP * 0.8 and self.hist[0][1].shape == mask.shape:
            ref = self.hist[0][1]
        self.hist.append((now, mask))

        flicker = 0.0
        if ref is not None:
            flicker = float(np.count_nonzero(cv2.bitwise_xor(mask, ref))) / max(1, nz, int(np.count_nonzero(ref)))
        self.last_area, self.last_flicker = area, flicker

        candidate = self.p["min_area"] < area < self.MAX_AREA and flicker > self.p["flicker"]
        if candidate:
            self.last_candidate = now
            if self.since is None:
                self.since = now
        elif self.since is not None and now - self.last_candidate > 0.6:
            self.since = None

        if self.since is not None and now - self.since >= self.p["confirm"]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x, y, bw, bh = cv2.boundingRect(np.vstack(cnts))
                return (int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale))
        return None


def fire_check(path: str, sensitivity: str):
    """Diagnostic: run the built-in fire detector over a video file and print what it sees."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = ColorFireDetector(sensitivity)
    i, fire_secs, max_area, max_fl = 0, set(), 0.0, 0.0
    print(f"Checking {path} at {fps:.0f} fps, sensitivity={sensitivity}")
    print("second  fire-colour area  flicker   result")
    sec_stats = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        hit = det.process(frame, t)
        sec = int(t)
        st = sec_stats.setdefault(sec, [0.0, 0.0, False])
        st[0], st[1], st[2] = max(st[0], det.last_area), max(st[1], det.last_flicker), st[2] or hit is not None
        max_area, max_fl = max(max_area, det.last_area), max(max_fl, det.last_flicker)
        i += 1
    for sec, (a, f, hit) in sorted(sec_stats.items()):
        print(f"{sec:>5}   {a * 100:>10.2f}%      {f:>6.3f}   {'FIRE' if hit else '-'}")
        if hit:
            fire_secs.add(sec)
    print(f"\nSeconds with a fire detection: {len(fire_secs)}")
    if not fire_secs:
        pr = FIRE_PRESETS[sensitivity]
        print(f"Highest fire-colour area {max_area * 100:.2f}% (needs > {pr['min_area'] * 100:.2f}%), "
              f"highest flicker {max_fl:.3f} (needs > {pr['flicker']}).")
        print("Try --sensitivity high, or send me these numbers.")


# --------------------------------------------------------------------------- #
# Shared state between detection thread and web routes
# --------------------------------------------------------------------------- #
class Hub:
    def __init__(self):
        self.cond = threading.Condition()
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.running = False
        self.status = "Ready. Enter a camera source and start detection."
        self.level = "info"
        self.fps = 0.0
        self.persons = 0
        self.alerts: collections.deque[dict] = collections.deque()
        self.next_id = 1
        self.last_sent: dict[str, float] = {}
        self.worker = None
        self.stop_event: threading.Event | None = None
        self.models: dict = {}

    def set_status(self, text: str, level: str = "info"):
        print(f"[{level.upper()}] {text}")
        with self.lock:
            self.status, self.level = text, level

    def publish_frame(self, jpeg: bytes):
        with self.cond:
            self.jpeg = jpeg
            self.frame_id += 1
            self.cond.notify_all()

    def add_alert(self, event, message, frame, cooldown, ntfy_server="", ntfy_topic=""):
        now = time.time()
        with self.lock:
            if now - self.last_sent.get(event, 0) < max(1, cooldown):
                return
            self.last_sent[event] = now
            alert_id = self.next_id
            self.next_id += 1

        snapshot, jpeg = "", None
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg = buf.tobytes()
                name = f"{int(now)}_{alert_id}_{event.lower()}.jpg"
                (SNAPSHOT_DIR / name).write_bytes(jpeg)
                snapshot = f"/snapshots/{name}"

        alert = {"id": alert_id, "event": event,
                 "label": ALERT_LABELS.get(event, event.replace("_", " ").title()),
                 "message": message, "time": now, "snapshot": snapshot}
        with self.lock:
            self.alerts.append(alert)
            while len(self.alerts) > MAX_ALERTS:
                old = self.alerts.popleft()
                if old["snapshot"]:
                    (SNAPSHOT_DIR / Path(old["snapshot"]).name).unlink(missing_ok=True)

        print(f"[ALERT] {event}: {message}")
        if ntfy_topic.strip():
            threading.Thread(target=notify_phone, args=(ntfy_server, ntfy_topic, event, message, jpeg),
                             daemon=True).start()

    def get_model(self, path: str):
        with self.lock:
            cached = self.models.get(path)
        if cached is not None:
            return cached
        from ultralytics import YOLO  # imported here so the web server starts fast
        model = YOLO(path)
        with self.lock:
            self.models[path] = model
        return model


hub = Hub()


def send_ntfy(server: str, topic: str, event: str, message: str, jpeg: bytes | None = None):
    """Send a push notification through ntfy. Returns (ok, error_text).
    When a snapshot is given it is attached so the phone shows the photo."""
    server = (server or "https://ntfy.sh").strip().rstrip("/")
    topic = topic.strip().strip("/")
    url = f"{server}/{topic}"
    ascii_msg = message.encode("ascii", "ignore").decode()   # HTTP headers must be plain ASCII
    headers = {
        "Title": f"Emergency alert: {ALERT_LABELS.get(event, event)}",
        "Priority": "urgent" if event in ("FIRE", "POSSIBLE_UNCONSCIOUS") else "default",
        "Tags": "rotating_light,warning" if event != "TEST" else "white_check_mark",
    }
    error = ""
    if jpeg:
        try:
            r = requests.put(url, data=jpeg, timeout=20,
                             headers={**headers, "Filename": f"{event.lower()}.jpg", "Message": ascii_msg})
            r.raise_for_status()
            return True, ""
        except requests.RequestException as exc:
            error = str(exc)          # fall through and try a plain text notification
    try:
        r = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=8)
        r.raise_for_status()
        return True, ""
    except requests.RequestException as exc:
        return False, str(exc) or error


def notify_phone(server, topic, event, message, jpeg=None):
    ok, err = send_ntfy(server, topic, event, message, jpeg)
    print(f"[PHONE] {event}: sent" if ok else f"[PHONE] {event}: FAILED - {err}")


# --------------------------------------------------------------------------- #
# Detection thread
# --------------------------------------------------------------------------- #
def open_capture(source: str):
    """Open a webcam, IP camera URL or video file, and make sure frames can really be read."""
    if source.isdigit():
        # Windows webcams need DirectShow/MSMF; the FFMPEG backend often opens but never delivers frames.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        targets = [(int(source), b) for b in backends]
    else:
        targets = [(source, cv2.CAP_FFMPEG), (source, cv2.CAP_ANY)]

    for src, backend in targets:
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            ok = False
            for _ in range(15):               # webcams can need a moment to deliver the first frame
                ok, _frame = cap.read()
                if ok:
                    break
                time.sleep(0.1)
            if ok:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low for RTSP
                return cap
        cap.release()
    return None


class DetectionWorker(threading.Thread):
    def __init__(self, settings: Settings, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.s = settings
        self.stop_event = stop_event
        self.tracker = PersonTracker()
        self.color_fire = ColorFireDetector(settings.fire_sensitivity)
        self.person_boxes: list = []
        self.pose_mode = False

    def load(self, path, label, download_ok=False):
        if not path:
            return None
        path = resolve_path(path)
        if not Path(path).exists() and not download_ok:
            hub.set_status(f"{label}: file not found: {path}", "error")
            return None
        try:
            model = hub.get_model(path)
            return model
        except Exception as exc:  # noqa: BLE001
            hub.set_status(f"{label}: could not load {path}: {exc}", "error")
            return None

    def run(self):
        hub.running = True
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            hub.set_status(f"Detection stopped unexpectedly: {exc}", "error")
        finally:
            hub.running = False
            if hub.level != "error":
                hub.set_status("Camera stopped.")

    def _run(self):
        s = self.s
        hub.set_status("Loading models… (the first run may download the pose model)")

        person_model = self.load(s.person_model or DEFAULT_PERSON_MODEL, "Person model", download_ok=True)
        if person_model is None and (s.person_model or DEFAULT_PERSON_MODEL) != FALLBACK_PERSON_MODEL:
            hub.set_status(f"Pose model unavailable, trying {FALLBACK_PERSON_MODEL} (less accurate)…")
            person_model = self.load(FALLBACK_PERSON_MODEL, "Person model", download_ok=True)
        if person_model is None:
            hub.set_status("No person model could be loaded, so 'person down' detection is unavailable. "
                           "Connect to the internet once so yolo11n-pose.pt can download.", "error")
            return
        self.pose_mode = getattr(person_model, "task", "") == "pose"

        fire_model, fire_note = None, ""
        if s.fire_model:
            fire_model = self.load(s.fire_model, "Fire model")
            if fire_model is not None:
                raw = fire_model.names
                names = [str(n) for n in (raw.values() if isinstance(raw, dict) else raw)]
                matching = [n for n in names if any(k in n.lower() for k in FIRE_WORDS)]
                if matching or len(names) == 1:
                    fire_note = f"Fire model classes: {', '.join(names)}."
                else:
                    fire_note = (f"WARNING: fire model classes are {', '.join(names)}. None contain "
                                 "fire/smoke/flame, so it will never alert.")
        if not fire_model and not s.builtin_fire:
            fire_note = "Fire detection is OFF (no model and built-in detector disabled)."

        fire_note += " Phone alerts ON." if s.ntfy_topic else " Phone alerts OFF (no ntfy topic)."
        mode = "pose model" if self.pose_mode else "box-only mode"
        hub.set_status(f"Connecting to {mask_credentials(s.source)} …")
        cap = open_capture(s.source)
        if cap is None:
            hub.set_status("Cannot read from the source. For a webcam: close other apps using it and allow camera access in Windows settings. For an IP camera: check the stream URL, credentials and network.", "error")
            return
        hub.set_status(f"Running ({mode}). {fire_note}".strip(), "ok")

        last_t = time.time()
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                hub.set_status("Stream interrupted. Reconnecting… (close other apps using the camera and check the URL/cable)")
                cap.release()
                cap = None
                while cap is None and not self.stop_event.wait(2):
                    cap = open_capture(s.source)
                if cap is None:
                    break
                hub.set_status(f"Reconnected. Running ({mode}). {fire_note}".strip(), "ok")
                continue

            now = time.time()
            events = {}
            self.detect_people(frame, person_model, now, events)
            self.detect_fire(frame, fire_model, now, events)
            for event, message in events.items():
                hub.add_alert(event, message, frame, s.cooldown, s.ntfy_server, s.ntfy_topic)

            inst = 1.0 / max(now - last_t, 1e-3)
            hub.fps = inst if hub.fps == 0 else hub.fps * 0.9 + inst * 0.1
            last_t = now

            out = frame
            h, w = out.shape[:2]
            if w > 1280:
                out = cv2.resize(out, (1280, int(h * 1280 / w)))
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                hub.publish_frame(buf.tobytes())

        if cap is not None:
            cap.release()

    # ---- people -------------------------------------------------------- #
    def detect_people(self, frame, model, now, events):
        s = self.s
        dets = []
        try:
            for result in model.predict(frame, conf=s.confidence, imgsz=640, verbose=False):
                labels = names_map(result)
                if result.boxes is None:
                    continue
                kp_all = result.keypoints.data.cpu().numpy() if getattr(result, "keypoints", None) is not None \
                    and result.keypoints.data is not None else None
                for i, box in enumerate(result.boxes):
                    if str(labels.get(int(box.cls.item()), "person")).lower() != "person":
                        continue
                    xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                    kp = kp_all[i] if kp_all is not None and i < len(kp_all) else None
                    dets.append((xyxy, kp))
        except Exception as exc:  # noqa: BLE001
            hub.set_status(f"Person inference error: {exc}", "error")
            return

        hub.persons = len(dets)
        self.person_boxes = [d[0] for d in dets]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for track, alarm in self.tracker.update(dets, gray, now, s.down_seconds):
            x1, y1, x2, y2 = map(int, track.box)
            if alarm:
                color, text = (255, 0, 180), f"POSSIBLE UNCONSCIOUS - still {int(track.still_for)}s"
                events["POSSIBLE_UNCONSCIOUS"] = (
                    f"A person has been lying down and not moving for about {int(track.still_for)} seconds. "
                    "Check on them now. The camera cannot confirm unconsciousness.")
            elif track.lying:
                color = (0, 165, 255)
                text = f"Lying {int(track.lying_for)}s" + (f", still {int(track.still_for)}s" if track.still_for else "")
            else:
                color, text = (0, 200, 0), "Person"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

    # ---- fire ---------------------------------------------------------- #
    def detect_fire(self, frame, model, now, events):
        s = self.s
        if model is not None:
            try:
                for result in model.predict(frame, conf=s.confidence, imgsz=640, verbose=False):
                    labels = names_map(result)
                    if result.boxes is None:
                        continue
                    single = len(labels) == 1
                    for box in result.boxes:
                        label = str(labels.get(int(box.cls.item()), "fire"))
                        if not single and not any(k in label.lower() for k in FIRE_WORDS):
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 70, 255), 2)
                        cv2.putText(frame, f"{label} {float(box.conf.item()):.2f}", (x1, max(22, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 70, 255), 2)
                        events["FIRE"] = f"Possible {label.lower()} detected by the fire model. Verify immediately."
            except Exception as exc:  # noqa: BLE001
                hub.set_status(f"Fire inference error: {exc}", "error")

        if s.builtin_fire:
            box = self.color_fire.process(frame, now, self.person_boxes)
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 70, 255), 2)
                cv2.putText(frame, "Flickering flame colour", (x1, max(22, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 70, 255), 2)
                events.setdefault("FIRE", "Flickering flame-coloured area detected. Verify immediately.")


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
LOGIN_USER = os.environ.get("EMS_USER", "admin")      # change before the demo
LOGIN_PASS = os.environ.get("EMS_PASS", "admin123")


@app.before_request
def require_login():
    if request.endpoint in ("login", "logout", "static") or session.get("ok"):
        return None
    if request.path.startswith(("/api/", "/snapshots/")) or request.path == "/video_feed":
        return jsonify(error="login required"), 401
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("u") == LOGIN_USER and request.form.get("p") == LOGIN_PASS:
            session["ok"] = True
            return redirect("/")
        return redirect("/login?err=1")
    for folder in (STATIC_DIR, BASE_DIR):
        if (folder / "login.html").exists():
            return send_from_directory(folder, "login.html")
    return "login.html not found. Put it next to web_app.py.", 404


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")



@app.get("/")
def index():
    for folder in (STATIC_DIR, BASE_DIR):
        if (folder / "index.html").exists():
            return send_from_directory(folder, "index.html")
    return ("index.html not found. Put it next to web_app.py or inside a folder named 'static'. "
            f"Looked in: {STATIC_DIR} and {BASE_DIR}"), 404


@app.get("/snapshots/<path:name>")
def snapshot(name: str):
    return send_from_directory(SNAPSHOT_DIR, name)


@app.get("/video_feed")
def video_feed():
    def generate():
        last = -1
        while True:
            with hub.cond:
                hub.cond.wait_for(lambda: hub.frame_id != last, timeout=3)
                jpeg, last = hub.jpeg, hub.frame_id
            if not hub.running and not jpeg:
                break
            if jpeg is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            if not hub.running:
                break

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    since = request.args.get("since", default=0, type=int)
    with hub.lock:
        return jsonify({
            "running": hub.running, "status": hub.status, "level": hub.level,
            "fps": round(hub.fps, 1), "persons": hub.persons, "total": len(hub.alerts),
            "alerts": [a for a in hub.alerts if a["id"] > since],
        })


@app.get("/api/models")
def models():
    found = set()
    for folder in (Path.cwd(), BASE_DIR, BASE_DIR / "models", Path.cwd() / "models"):
        if folder.is_dir():
            for f in folder.glob("*.pt"):
                try:
                    found.add(str(f.relative_to(BASE_DIR)))
                except ValueError:
                    found.add(str(f))
    return jsonify(sorted(found))


@app.post("/api/start")
def start():
    if hub.worker and hub.worker.is_alive():
        return jsonify(error="Detection is already running."), 409
    data = request.get_json(silent=True) or {}
    source = str(data.get("source", "")).strip()
    if not source:
        return jsonify(error="Enter a camera source or IP camera URL."), 400
    try:
        confidence = float(data.get("confidence", 0.4))
        cooldown = int(data.get("cooldown", 20))
        down_seconds = int(data.get("down_seconds", 15))
    except (TypeError, ValueError):
        return jsonify(error="Confidence must be a number; cooldown and seconds must be whole numbers."), 400
    if not 0 < confidence <= 1:
        return jsonify(error="Confidence must be greater than 0 and at most 1."), 400
    if cooldown < 1 or down_seconds < 3:
        return jsonify(error="Cooldown must be at least 1 second and 'still for' at least 3 seconds."), 400

    settings = Settings(
        source=source,
        person_model=str(data.get("person_model", "")).strip() or DEFAULT_PERSON_MODEL,
        fire_model=str(data.get("fire_model", "")).strip(),
        builtin_fire=bool(data.get("builtin_fire", True)),
        fire_sensitivity=(str(data.get("fire_sensitivity", "medium")).lower()
                          if str(data.get("fire_sensitivity", "medium")).lower() in FIRE_PRESETS else "medium"),
        confidence=confidence, down_seconds=down_seconds,
        ntfy_server=str(data.get("ntfy_server", "")).strip() or "https://ntfy.sh",
        ntfy_topic=str(data.get("ntfy_topic", "")).strip(), cooldown=cooldown)

    with hub.cond:
        hub.jpeg = None
    hub.fps, hub.persons, hub.level = 0.0, 0, "info"
    hub.stop_event = threading.Event()
    hub.worker = DetectionWorker(settings, hub.stop_event)
    hub.running = True
    hub.worker.start()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    if hub.stop_event:
        hub.stop_event.set()
    with hub.cond:
        hub.cond.notify_all()
    hub.set_status("Stopping…")
    return jsonify(ok=True)


@app.post("/api/test")
def test_notification():
    data = request.get_json(silent=True) or {}
    topic = str(data.get("ntfy_topic", "")).strip()
    if not topic:
        return jsonify(error="Enter or generate an ntfy topic first."), 400
    ok, err = send_ntfy(str(data.get("ntfy_server", "")), topic, "TEST",
                        "Test notification from your emergency camera system.")
    if not ok:
        return jsonify(error=f"Could not reach ntfy: {err}"), 502
    hub.add_alert("TEST", "Test notification sent to your phone via ntfy.", None, 1)
    return jsonify(ok=True)


@app.post("/api/clear")
def clear():
    with hub.lock:
        for a in hub.alerts:
            if a["snapshot"]:
                (SNAPSHOT_DIR / Path(a["snapshot"]).name).unlink(missing_ok=True)
        hub.alerts.clear()
    return jsonify(ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency camera alert dashboard")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Use 0.0.0.0 to open the dashboard from other devices on your network "
                             "(there is no login, so only do this on a trusted network).")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--fire-check", metavar="VIDEO", help="Test the built-in fire detector on a video file and exit.")
    parser.add_argument("--sensitivity", choices=list(FIRE_PRESETS), default="medium")
    args = parser.parse_args()
    if args.fire_check:
        fire_check(args.fire_check, args.sensitivity)
        raise SystemExit
    print(f"Dashboard: http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
