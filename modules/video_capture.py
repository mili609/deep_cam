import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading


# Stable dropdown IDs map to physical identities, not backend-specific indices.
_CAMERA_ROUTES = {}
_CAMERA_NAMES = {}
_WORKING_ROUTES = {}

def get_camera_choices():
    """Enumerate both Windows APIs and merge only matching physical device paths."""
    if platform.system() != "Windows":
        return [], []
    from cv2_enumerate_cameras import enumerate_cameras
    devices = {}
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        try:
            for camera in enumerate_cameras(backend):
                path = (camera.path or "").lower().split("#{")[0]
                key = path or (backend, camera.index)
                entry = devices.setdefault(key, {"name": camera.name, "routes": []})
                route = (camera.index, backend)
                if route not in entry["routes"]:
                    entry["routes"].append(route)
        except Exception as exc:
            print(f"[camera] Enumeration failed for backend {backend}: {exc}", flush=True)
    _CAMERA_ROUTES.clear()
    _CAMERA_NAMES.clear()
    for token, entry in enumerate(devices.values()):
        _CAMERA_ROUTES[token] = entry["routes"]
        _CAMERA_NAMES[token] = entry["name"]
    return list(_CAMERA_NAMES), list(_CAMERA_NAMES.values())


def camera_routes(device_index):
    if not _CAMERA_ROUTES:
        get_camera_choices()
    routes = list(_CAMERA_ROUTES.get(device_index, []))
    if not routes:
        # Manual indices still work when Windows enumeration is unavailable.
        routes = [(device_index, cv2.CAP_DSHOW), (device_index, cv2.CAP_MSMF)]
    preferred = _WORKING_ROUTES.get(device_index)
    if preferred in routes:
        routes.remove(preferred)
        routes.insert(0, preferred)
    return routes

class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        if self.is_running:
            return True
        try:
            if platform.system() == "Windows":
                methods = camera_routes(self.device_index)
            elif platform.system() == "Linux":
                methods = [(f"/dev/video{self.device_index}", cv2.CAP_ANY)]
            else:
                methods = [(self.device_index, cv2.CAP_ANY)]
            failures = []
            for index, backend in methods:
                self.release()
                try:
                    self.cap = cv2.VideoCapture(index, backend)
                    if not self.cap.isOpened():
                        raise RuntimeError("open returned false")
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.cap.set(cv2.CAP_PROP_FPS, fps)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, first = self.cap.read()
                    if not ret or first is None or first.size == 0:
                        raise RuntimeError("opened but could not capture a frame")
                    self._pending_frame = first
                    self.active_index, self.active_backend = index, backend
                    _WORKING_ROUTES[self.device_index] = (index, backend)
                    print(f"[VideoCapturer] Frame verified: index={index} backend={backend} shape={first.shape}", flush=True)
                    break
                except Exception as exc:
                    failures.append(f"index={index} backend={backend}: {exc}")
                    print(f"[VideoCapturer] {failures[-1]}", flush=True)
                    self.release()
            if self.cap is None:
                raise RuntimeError("All camera routes failed: " + "; ".join(failures))

            # Read back resolution (usually reliable)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # CAP_PROP_FPS is unreliable on DirectShow — often reports 30
            # even when the camera delivers 60.  Measure empirically by
            # timing a burst of frames.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = reported_fps if 0 < reported_fps <= 240 else float(fps)

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            self.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        if getattr(self, "_pending_frame", None) is not None:
            ret, frame = True, self._pending_frame
            self._pending_frame = None
        else:
            ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        self.is_running = False
        self._pending_frame = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS.

        This is more reliable than CAP_PROP_FPS which often lies on
        DirectShow.  Takes ~0.5-1s at startup but gives a ground-truth
        number for adaptive polling/detection intervals.
        """
        try:
            for _ in range(warmup):
                self.cap.read()
            t0 = time.perf_counter()
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret:
                    return fallback
            elapsed = time.perf_counter() - t0
            if elapsed <= 0:
                return fallback
            return sample / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
