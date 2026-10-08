"""Opt-in physical webcam check. Opens camera 0 and shows the existing GUI."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DLC_DEBUG"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from PySide6.QtCore import QTimer
import modules.globals as g
from modules import ui, core
root = Path(__file__).resolve().parents[1]
os.chdir(root)
g.execution_providers = core.decode_execution_providers([core.suggest_default_execution_provider()])
from modules.runtime import initialize_execution
initialize_execution()
g.frame_processors = ["face_swapper"]
g.execution_threads = 2
g.source_path = str(root / "diagnostics/source.png")
g.target_path = str(root / "media/demo.gif")  # Start must ignore a loaded file.
ui.init(core.start, lambda: None, "en")
g.map_faces = False
g.many_faces = False
g.frame_processors = ["face_swapper"]
g.fp_ui = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}
g.det_size = 320
g.capture_resolution = (640, 480)
g.opacity = 1.0
g.show_fps = False
ui._MAIN.show()
beats = [0]
timer = QTimer()
timer.timeout.connect(lambda: beats.__setitem__(0, beats[0]+1))
timer.start(20)
ui._MAIN._on_start()
window = ui._WEBCAM_PREVIEW
assert window is not None
output = root / "output.mp4"
output_before = (output.stat().st_size, output.stat().st_mtime_ns) if output.exists() else None
deadline = time.monotonic()+120
while not getattr(window, "_received_frame", False) and ui._WEBCAM_PREVIEW is not None:
    ui._APP.processEvents()
    time.sleep(0.005)
    if time.monotonic() >= deadline:
        break
worker = window._processing_worker
if getattr(window, "_received_frame", False):
    count = worker.processed_frames
    benchmark_start = time.perf_counter()
    while worker.processed_frames < count+10 and ui._WEBCAM_PREVIEW is not None and time.perf_counter()-benchmark_start < 60:
        ui._APP.processEvents()
        time.sleep(0.005)
    actual_fps = (worker.processed_frames-count)/(time.perf_counter()-benchmark_start)
    print(f"PHYSICAL BENCHMARK: {actual_fps:.2f} FPS; provider={worker.actual_provider}; processed={worker.processed_frames-count}", flush=True)
    ui._MAIN.grab().save(str(root/"diagnostics/physical-gui.png"))
    window.grab().save(str(root/"diagnostics/physical-live.png"))
print("PHYSICAL CAMERA: displayed_processed_frame=", getattr(window, "_received_frame", False),
      "diagnostics=", getattr(window._processing_worker, "last_swap_diagnostics", None),
      "gui_heartbeats=", beats[0],
      "FPS=", getattr(window._processing_worker, "measured_fps", None), flush=True)
window.close()
deadline = time.monotonic()+30
while ui._WEBCAM_PREVIEW is not None:
    ui._APP.processEvents()
    time.sleep(0.01)
    assert time.monotonic() < deadline
assert ((output.stat().st_size, output.stat().st_mtime_ns) if output.exists() else None) == output_before
print("STOP: workers finished=", not window._capture_worker.isRunning() and not window._processing_worker.isRunning())
from modules.video_capture import VideoCapturer
camera = VideoCapturer(0)
print("REOPEN AFTER STOP:", camera.start(640,480,30))
camera.release()
if getattr(window, "_received_frame", False):
    diagnostics = getattr(worker, "last_swap_diagnostics", {})
    success = getattr(worker, "last_successful_swap_diagnostics", {})
    assert success.get("target_faces", 0) > 0, "No face detected in physical webcam"
    assert success.get("changed_values", 0) > 0, "No swapped pixels in physical webcam"
    print("SWAP VERIFIED:", success, flush=True)
    print(f"FACE PROCESSING: {worker.face_frames/worker.face_processing_seconds:.2f} FPS over {worker.face_frames} face-bearing webcam frames", flush=True)
timer.stop()
ui._MAIN.close()

if not getattr(window, "_received_frame", False):
    raise SystemExit("PHYSICAL CAMERA VERIFICATION FAILED: camera did not deliver a processed frame")
