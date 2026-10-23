"""
Download a public fire/smoke YOLO model into models/fire.pt, or test any model.

  Download:   python get_fire_model.py
  Test:       python get_fire_model.py --test 0        (0 = your webcam)
              python get_fire_model.py --test "C:\\path\\to\\fire_video.mp4"
  Test other: python get_fire_model.py --test video.mp4 --model path\\to\\other.pt

Model: rabahdev/fire-smoke-yolov8n on Hugging Face (YOLOv8n trained on the D-Fire
dataset, classes: smoke, fire, license AGPL-3.0). Check the license suits your use.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "models" / "fire.pt"
REPO, FILE = "rabahdev/fire-smoke-yolov8n", "best.pt"


def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing package. Run this first, then run this script again:")
        print(f'  "{sys.executable}" -m pip install huggingface_hub')
        sys.exit(1)
    print(f"Downloading {FILE} from {REPO} ...")
    path = hf_hub_download(repo_id=REPO, filename=FILE)
    DEST.parent.mkdir(exist_ok=True)
    shutil.copy(path, DEST)
    print(f"Saved to {DEST}")


def test(model_path: str, media: str, conf: float):
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print("Model classes:", names)
    if not any(k in str(n).lower() for n in names.values() for k in ("fire", "smoke", "flame")) and len(names) != 1:
        print("WARNING: no class name contains fire/smoke/flame, so the dashboard would ignore this model.")

    is_cam = media.isdigit()
    cap = (cv2.VideoCapture(int(media), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
           if is_cam else cv2.VideoCapture(media))
    if not cap.isOpened():
        print(f"Cannot open {'webcam ' if is_cam else ''}{media}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if is_cam:
        print("Webcam test running for 20 seconds. Hold a fire or smoke picture/video up to the camera "
              "(for example on a phone screen).")
    total, hits, i = {}, 0, 0
    best = 0.0
    start = time.time()
    last_print = 0.0
    while True:
        if is_cam and time.time() - start > 20:
            break
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if not is_cam and i % max(1, int(fps // 5)) != 0:      # about 5 checks per second for files
            continue
        for r in model.predict(frame, conf=conf, imgsz=640, verbose=False):
            if r.boxes is not None and len(r.boxes):
                hits += 1
                found = []
                for b in r.boxes:
                    label = str(names.get(int(b.cls.item()), "?"))
                    total[label] = total.get(label, 0) + 1
                    best = max(best, float(b.conf.item()))
                    found.append(f"{label} {float(b.conf.item()):.2f}")
                if is_cam and time.time() - last_print > 1:
                    print("  detected:", ", ".join(found))
                    last_print = time.time()
    cap.release()
    print(f"Checked {i} frames. Frames with a detection: {hits}. By class: {total or 'nothing found'}")
    if best:
        print(f"Highest confidence seen: {best:.2f}  (the dashboard confidence setting must be below this)")
    else:
        print(f"Nothing detected at confidence {conf}. Try --conf 0.15, or show clearer fire/smoke.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", metavar="MEDIA", help="0 for your webcam, or an image/video file path")
    ap.add_argument("--model", default=str(DEST), help="model to test (default: models/fire.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if a.test:
        test(a.model, a.test, a.conf)
    else:
        download()
        print("\nNow test it with your webcam (show a fire picture on a phone screen):")
        print("  python get_fire_model.py --test 0")
