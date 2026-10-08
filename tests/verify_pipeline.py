r"""Opt-in integration check using bundled media; never opens a real camera.
Run: venv\Scripts\python.exe tests\verify_pipeline.py
"""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
from PySide6.QtCore import QTimer
import modules.globals as g
from modules import ui, core, imread_unicode
from modules.face_analyser import get_face_analyser
from modules.processors.frame.face_swapper import get_face_swapper

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
ARTIFACTS = ROOT / "diagnostics"
ARTIFACTS.mkdir(exist_ok=True)
capture = cv2.VideoCapture(str(ROOT / "media/demo.gif"))
ok, target = capture.read()
capture.release()
assert ok
source = cv2.resize(target[:, 300:], None, fx=2, fy=2)
cv2.imwrite(str(ARTIFACTS / "source.png"), source)
cv2.imwrite(str(ARTIFACTS / "target.png"), target)
g.source_path = str(ARTIFACTS / "source.png")
g.target_path = str(ARTIFACTS / "target.png")
g.output_path = str(ARTIFACTS / "output.png")
g.execution_providers = ["CPUExecutionProvider"]
g.execution_threads = 2
g.frame_processors = ["face_swapper"]
g.map_faces = False
g.many_faces = False
g.nsfw_filter = False
g.headless = False
g.opacity = 1.0
g.show_fps = False
ui.init(core.start, lambda: None, "en")
# Preferences loaded at init must not change this deterministic check.
g.map_faces = g.many_faces = g.nsfw_filter = False
g.frame_processors = ["face_swapper"]
g.fp_ui = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}
g.opacity = 1.0
g.show_fps = False
heartbeats = [0]
timer = QTimer()
timer.timeout.connect(lambda: heartbeats.__setitem__(0, heartbeats[0] + 1))
timer.start(10)

def wait_until(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while not predicate():
        ui._APP.processEvents()
        time.sleep(0.005)
        assert time.monotonic() < deadline, "Timed out waiting for workers"
    ui._APP.processEvents()

ui._MAIN._run_task(core.start)
wait_until(lambda: ui._MAIN._task is None)
result = imread_unicode(g.output_path)
assert result is not None and np.count_nonzero(result != target) > 0
assert ui._PREVIEW._image_label.pixmap() is not None
assert heartbeats[0] > 100
print("IMAGE PASS: saved pixels differ, output rendered, GUI timer responsive", heartbeats[0])
analyser, swapper = get_face_analyser(), get_face_swapper()

class FakeCamera:
    opened = 0
    released = 0
    def __init__(self, index):
        assert index == 0
    def start(self, *args):
        FakeCamera.opened += 1
        return True
    def read(self):
        time.sleep(0.03)
        return True, target.copy()
    def release(self):
        FakeCamera.released += 1

ui.VideoCapturer = FakeCamera
ui._open_webcam_preview(0)
window = ui._WEBCAM_PREVIEW
ui._open_webcam_preview(0)
wait_until(lambda: getattr(window, "_received_frame", False))
# Read another actual processed frame without the display timer draining it.
window._timer.stop()
wait_until(lambda: not window._processed_queue.empty())
frame = window._processed_queue.get_nowait()
assert np.count_nonzero(frame != target) > 0
window._timer.start(16)
window.close()
wait_until(lambda: ui._WEBCAM_PREVIEW is None)
assert FakeCamera.opened == FakeCamera.released == 1
assert get_face_analyser() is analyser and get_face_swapper() is swapper
print("LIVE PASS: continuous swapped frames, single capture, released on stop, models reused")

class FailedCamera(FakeCamera):
    def start(self, *args):
        return False
ui.VideoCapturer = FailedCamera
ui._open_webcam_preview(0)
wait_until(lambda: ui._WEBCAM_PREVIEW is None)
print("CAMERA FAILURE PASS: safe cleanup without uninitialized worker errors")
# Exercise the existing many-face and explicit mapper paths with real models.
from modules.face_analyser import get_one_face, get_many_faces
g.many_faces = True
ui._MAIN._run_task(core.start)
wait_until(lambda: ui._MAIN._task is None)
many_result = imread_unicode(g.output_path)
faces = get_many_faces(target)
for face in faces:
    x1, y1, x2, y2 = face.bbox.astype(int)
    assert np.count_nonzero(many_result[y1:y2,x1:x2] != target[y1:y2,x1:x2]) > 0
g.many_faces = False
g.map_faces = True
g.source_target_map = [{"source": {"face": get_one_face(source)},
                        "target": {"face": faces[1]}}]
ui._MAIN._run_task(core.start)
wait_until(lambda: ui._MAIN._task is None)
assert np.count_nonzero(imread_unicode(g.output_path) != target) > 0
g.map_faces = False
print("MULTI-FACE AND MAPPING PASS: existing selection logic exercised")
# No-face failures cannot replace an earlier output with an unchanged target.
blank = ARTIFACTS / "blank.png"
cv2.imwrite(str(blank), np.zeros_like(target))
g.target_path = str(blank)
before = Path(g.output_path).read_bytes()
ui._MAIN._run_task(core.start)
wait_until(lambda: ui._MAIN._task is None)
assert Path(g.output_path).read_bytes() == before
print("NO FACE PASS: earlier output preserved")
timer.stop()
ui._PREVIEW.close()
ui._MAIN.close()
