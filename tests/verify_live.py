"""Integration check: real models, controlled capture, live GUI callbacks."""
import os
import sys
import time
import hashlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
from PySide6.QtCore import QTimer
from modules import ui, core
import modules.globals as g
from modules.runtime import initialize_execution
from modules.face_analyser import get_face_analyser
from modules.processors.frame.face_swapper import get_face_swapper
root = Path(__file__).resolve().parents[1]
os.chdir(root)
g.execution_providers = core.decode_execution_providers([core.suggest_default_execution_provider()])
initialize_execution()
g.frame_processors = ["face_swapper"]
g.execution_threads = 2
g.headless = False
cap = cv2.VideoCapture(str(root / "media/demo.gif"))
ok, target = cap.read()
cap.release()
assert ok
source = root / "diagnostics/source.png"
assert source.exists()
output = root / "output.mp4"
def output_digest():
    return hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
before = output_digest()
class Camera:
    opened = released = 0
    def __init__(self, index):
        assert index == 3, index
    def start(self, *args):
        Camera.opened += 1
        return True
    def read(self):
        time.sleep(0.02)
        return True, target.copy()
    def release(self):
        Camera.released += 1
with patch.object(ui, "get_available_cameras", return_value=([3], ["Test camera"])), patch.object(ui, "VideoCapturer", Camera):
    ui.init(lambda: (_ for _ in ()).throw(AssertionError("Offline pipeline called")), lambda: None, "en")
    g.map_faces = g.many_faces = g.nsfw_filter = False
    g.opacity = 1.0
    g.show_fps = False
    g.frame_processors = ["face_swapper"]
    g.fp_ui = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}
    g.target_path = str(root / "media/demo.gif")
    with patch.object(ui.QFileDialog, "getOpenFileName", return_value=(str(source), "")):
        ui._MAIN._on_select_source()
    assert g.source_path == str(source)
    assert ui._MAIN.source_label.pixmap() is not None
    ui._MAIN.show()
    ui._MAIN.btn_start.click()
    window = ui._WEBCAM_PREVIEW
    assert window is not None
    ui._MAIN.btn_preview.click()
    assert ui._WEBCAM_PREVIEW is window
    beats = [0]
    timer = QTimer()
    timer.timeout.connect(lambda: beats.__setitem__(0, beats[0]+1))
    timer.start(10)
    def wait(predicate, timeout=60):
        deadline = time.monotonic()+timeout
        while not predicate():
            ui._APP.processEvents()
            time.sleep(0.005)
            assert time.monotonic() < deadline, "Live worker timeout"
    wait(lambda: getattr(window, "_received_frame", False))
    analyser, swapper = get_face_analyser(), get_face_swapper()
    worker = window._processing_worker
    initial_count = worker.processed_frames
    start = time.perf_counter()
    wait(lambda: worker.processed_frames >= initial_count+10)
    fps = (worker.processed_frames-initial_count)/(time.perf_counter()-start)
    window._timer.stop()
    wait(lambda: not window._processed_queue.empty())
    swapped = window._processed_queue.get_nowait()
    changed = np.count_nonzero(swapped != target)
    assert changed > 0
    ui._MAIN.grab().save(str(root / "diagnostics/live-gui.png"))
    window.grab().save(str(root / "diagnostics/live-preview.png"))
    window._timer.start(16)
    stop_start = time.perf_counter()
    ui._MAIN.btn_destroy.click()
    wait(lambda: ui._WEBCAM_PREVIEW is None)
    assert Camera.opened == Camera.released == 1
    assert get_face_analyser() is analyser and get_face_swapper() is swapper
    assert output_digest() == before, "Live mode modified output.mp4"
    assert beats[0] > 10
    print(f"LIVE PASS | Provider: {worker.actual_provider} | FPS: {fps:.2f} | changed pixels: {changed} | GUI ticks: {beats[0]} | stop: {time.perf_counter()-stop_start:.2f}s", flush=True)
    print("Source selection, selected camera index, Start with loaded target, Preview reuse, model reuse, no offline callback, output.mp4 unchanged: PASS", flush=True)
    timer.stop()
    ui._MAIN.close()
