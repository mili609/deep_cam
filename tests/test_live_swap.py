"""Regression checks for camera/source roles and discarded swap output."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from modules import ui
import modules.globals as g

APP = QApplication.instance() or QApplication([])

class Output(queue.Queue):
    def __init__(self, stop, count):
        super().__init__()
        self.stop, self.count = stop, count
    def put_nowait(self, frame):
        super().put_nowait(frame)
        if self.qsize() == self.count:
            self.stop.set()

class LiveTests(unittest.TestCase):
    def run_worker(self, swap, face=True, count=2):
        stop = threading.Event()
        capture, output = queue.Queue(), Output(stop, count)
        frames = [np.full((24, 24, 3), i + 10, np.uint8) for i in range(count)]
        for frame in frames:
            capture.put(frame)
        source = SimpleNamespace(normed_embedding=np.ones(512))
        target = SimpleNamespace(bbox=np.array([2, 2, 20, 20]))
        processor = SimpleNamespace(NAME="DLC.FACE-SWAPPER", swap_face=swap,
                                    apply_post_processing=lambda f, b: f)
        worker = ui._ProcessingWorker(capture, output, stop, 30)
        with patch.multiple(g, source_path="selected.png", target_path="never-read.png",
                            frame_processors=["face_swapper"], map_faces=True,
                            many_faces=False, live_mirror=False, mouth_mask=False, show_fps=False), \
             patch.object(ui, "imread_unicode", return_value=np.zeros((8, 8, 3), np.uint8)) as read, \
             patch.object(ui, "get_one_face", return_value=source) as extract, \
             patch.object(ui, "detect_one_face_fast", return_value=target if face else None) as detect, \
             patch.object(ui, "get_frame_processors_modules", return_value=[processor]), \
             patch("modules.processors.frame.face_swapper.get_face_swapper", return_value=SimpleNamespace(
                 session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))), \
             patch.object(ui, "update_status"):
            if swap is None:
                processor.swap_face = lambda *args: (_ for _ in ()).throw(RuntimeError("inference failed"))
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    worker._process()
                self.assertTrue(output.empty())
                return
            worker._process()
            self.assertEqual(extract.call_count, 1)
            read.assert_called_once_with("selected.png")
            self.assertEqual(detect.call_count, count)
        return frames, list(output.queue), worker

    def test_selected_source_and_returned_camera_frame_with_mapping_enabled(self):
        calls = []
        def swap(source, target, frame):
            calls.append((source, target, frame.copy()))
            return np.full_like(frame, 99)
        raw, shown, worker = self.run_worker(swap)
        self.assertEqual(len(calls), 2)
        for i in range(2):
            np.testing.assert_array_equal(calls[i][2], raw[i])
            self.assertTrue(np.all(shown[i] == 99))
        self.assertTrue(worker.last_swap_diagnostics["swap_applied"])

    def test_no_target_can_display_camera(self):
        raw, shown, worker = self.run_worker(lambda *a: self.fail("Unexpected swap"), face=False)
        for original, displayed in zip(raw, shown):
            np.testing.assert_array_equal(original, displayed)
        self.assertFalse(worker.last_swap_diagnostics["swap_applied"])

    def test_swap_exception_never_enqueues_original(self):
        self.run_worker(None)

    def test_unchanged_swap_is_not_displayed(self):
        with self.assertRaisesRegex(RuntimeError, "unchanged camera pixels"):
            self.run_worker(lambda s, t, f: f)

if __name__ == "__main__":
    unittest.main()
