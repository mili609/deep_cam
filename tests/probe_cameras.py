"""Probe each Windows camera backend in an isolated, bounded process."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--probe", nargs=2, type=int)
parser.add_argument("--count", type=int, default=6)
args = parser.parse_args()
if args.probe:
    import cv2
    import time
    index, backend = args.probe
    cap = cv2.VideoCapture(index, backend)
    result = {"index": index, "backend": backend, "opened": cap.isOpened(), "frame": False}
    try:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            result["frame"] = bool(ok and frame is not None and frame.size)
            if result["frame"]:
                result["shape"] = list(frame.shape)
                start = time.perf_counter()
                frames = 0
                for _ in range(10):
                    ok, frame = cap.read()
                    frames += bool(ok)
                result["capture_fps"] = frames/(time.perf_counter()-start)
    finally:
        cap.release()
    print(json.dumps(result), flush=True)
else:
    results = []
    for backend in (700, 1400):
        for index in range(args.count):
            try:
                proc = subprocess.run([sys.executable, __file__, "--probe", str(index), str(backend)], cwd=ROOT,
                                      capture_output=True, text=True, timeout=15)
                result = json.loads(proc.stdout.strip())
                if proc.stderr:
                    result["backend_log"] = proc.stderr.strip()
            except subprocess.TimeoutExpired:
                result = {"index": index, "backend": backend, "error": "15-second timeout", "frame": False}
            except Exception as exc:
                result = {"index": index, "backend": backend, "error": str(exc), "frame": False}
            results.append(result)
            print(json.dumps(result), flush=True)
    (ROOT/"diagnostics/camera-probes.json").write_text(json.dumps(results, indent=2))
    if not any(r.get("frame") for r in results):
        raise SystemExit("No physical webcam frame could be captured")
