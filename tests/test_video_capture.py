"""Physical identity mapping and first-frame backend fallback regressions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from modules import video_capture as v

class Capture:
    instances = []
    def __init__(self, index, backend):
        self.index, self.backend = index, backend
        self.released = False
        self.instances.append(self)
    def isOpened(self):
        return True
    def set(self, *args):
        return True
    def get(self, key):
        return 30 if key == v.cv2.CAP_PROP_FPS else 640
    def read(self):
        return (False, None) if self.backend == v.cv2.CAP_DSHOW else (True, np.zeros((480, 640, 3), dtype=np.uint8))
    def release(self):
        self.released = True

class CameraTests(unittest.TestCase):
    def setUp(self):
        v._CAMERA_ROUTES.clear()
        v._CAMERA_NAMES.clear()
        v._WORKING_ROUTES.clear()
        Capture.instances.clear()
    def tearDown(self):
        v._CAMERA_ROUTES.clear()
        v._CAMERA_NAMES.clear()
        v._WORKING_ROUTES.clear()
    def test_same_physical_device_merges_different_backend_indices(self):
        def enumerate_backend(backend):
            return [SimpleNamespace(index=0 if backend == 700 else 4, name="Integrated Camera", path="usb#vid_1234#serial#{"+str(backend)+"}")]
        with patch("cv2_enumerate_cameras.enumerate_cameras", side_effect=enumerate_backend), patch.object(v.platform, "system", return_value="Windows"):
            indices, names = v.get_camera_choices()
        self.assertEqual(indices, [0])
        self.assertEqual(names, ["Integrated Camera"])
        self.assertEqual(v.camera_routes(0), [(0, 700), (4, 1400)])
    def test_media_foundation_only_device_is_selectable(self):
        def enumerate_backend(backend):
            return [] if backend == 700 else [SimpleNamespace(index=5, name="MSMF camera", path="usb#device#{guid}")]
        with patch("cv2_enumerate_cameras.enumerate_cameras", side_effect=enumerate_backend), patch.object(v.platform, "system", return_value="Windows"):
            self.assertEqual(v.get_camera_choices(), ([0], ["MSMF camera"]))
        self.assertEqual(v.camera_routes(0), [(5, 1400)])
    def test_open_without_frames_falls_back_and_retains_first_frame(self):
        v._CAMERA_ROUTES[0] = [(0, 700), (4, 1400)]
        with patch.object(v.platform, "system", return_value="Windows"), patch.object(v.cv2, "VideoCapture", Capture):
            camera = v.VideoCapturer(0)
            self.assertTrue(camera.start(640, 480, 30))
            self.assertTrue(Capture.instances[0].released)
            self.assertEqual((camera.active_index, camera.active_backend), (4, 1400))
            self.assertTrue(camera.read()[0])
            camera.release()
            self.assertTrue(Capture.instances[1].released)
            self.assertEqual(v.camera_routes(0)[0], (4, 1400))

if __name__ == "__main__":
    unittest.main()
