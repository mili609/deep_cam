"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import logging
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    pass

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 820
ROOT_WIDTH = 640

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QMainWindow, QDialog { background-color: #1e1e1e; color: #e6e6e6; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; }

QGroupBox {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #3a3a3a;
}
QPushButton#secondary:hover { background-color: #4a4a4a; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 36px; height: 18px;
    border-radius: 9px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #2a2a2a;
    border: 2px dashed #444;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #262626;
    border-radius: 10px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Display BGR directly, with an owned image buffer for Qt."""
    h, w = bgr.shape[:2]
    bgr = np.ascontiguousarray(bgr)
    qimg = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)
    outputReady = Signal(object)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))


def show_processed_image(frame):
    if _BRIDGE is not None:
        _BRIDGE.outputReady.emit(frame.copy())


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        from modules.video_capture import get_camera_choices
        indices, names = get_camera_choices()
        if names:
            return indices, names
        return list(range(5)), [f"Camera {i} (probe DirectShow / Media Foundation)" for i in range(5)]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class _TaskWorker(QThread):
    resultReady = Signal(object)

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        try:
            self.resultReady.emit(self.task())
        except Exception as exc:
            update_status(f"Processing failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())

        # Action buttons
        layout.addLayout(self._build_action_row())

        # Camera selection
        layout.addWidget(self._build_camera_card())

        # Status & footer
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        footer = QLabel("Deep Live Cam")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        layout.addWidget(footer)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown
        enhancer_label = QLabel(_("Face Enhancer:"))
        grid.addWidget(enhancer_label, len(items) // 2, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, len(items) // 2, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Start live face swapping with the selected webcam"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(self._on_stop)

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show the live processed webcam feed"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        self.btn_process_file = QPushButton(_("Process file"))
        self.btn_process_file.setObjectName("secondary")
        self.btn_process_file.setToolTip(_("Explicitly save a processed target image/video"))
        self.btn_process_file.clicked.connect(self._on_process_file)
        row.addWidget(self.btn_process_file)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Row 0: camera selector + Live button
        grid.addWidget(QLabel(_("Select Camera:")), 0, 0)
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.currentIndexChanged.connect(self._on_camera_change)
        self.cb_camera.activated.connect(self._on_camera_change)
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        grid.addWidget(self.cb_camera, 0, 1)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        grid.addWidget(self.btn_live, 0, 2)

        # Row 1: capture resolution
        grid.addWidget(QLabel(_("Resolution:")), 1, 0)
        self.cb_resolution = QComboBox()
        # 640x480 is native on virtually every webcam and the safest 30fps
        # mode on USB 2.0. Higher tiers are 16:9 and useful when the camera
        # supports them natively (DSLR + capture card, USB 3.0 webcams).
        self._resolution_options = [
            ("640 x 480", (640, 480)),
            ("960 x 540 (qHD)", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 x 1080 (FHD)", (1920, 1080)),
        ]
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == cur), 0)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.cb_resolution.setToolTip(_(
            "Requested webcam resolution. Camera may negotiate to its "
            "nearest supported size — actual size printed in console as "
            "'[VideoCapturer] WxH @ FPS'. Applies on next Live start.\n\n"
            "640x480 is native on virtually every webcam and the safest "
            "30fps choice. qHD often works too. HD/FHD typically drop to "
            "~10fps on USB 2.0 webcams due to isochronous bandwidth limits."
        ))
        grid.addWidget(self.cb_resolution, 1, 1, 1, 2)

        # Row 2: face detection size
        grid.addWidget(QLabel(_("Det size:")), 2, 0)
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v} x {v}")
        cur_det = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
        # Normalize to a valid option first so .index() can never raise: fall back
        # to the default size, then to the last (highest) option if even that is
        # somehow missing.
        if cur_det not in self._det_size_options:
            cur_det = (modules.globals.DEFAULT_DET_SIZE
                       if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                       else self._det_size_options[-1])
        idx = self._det_size_options.index(cur_det)
        self.cb_det_size.setCurrentIndex(idx)
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(_(
            "Face detection input resolution. Lower = faster, less accurate at "
            "distance. Changes take effect on next face analyser init."
        ))
        grid.addWidget(self.cb_det_size, 2, 1, 1, 2)

        grid.setColumnStretch(1, 1)
        return card

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            # Pre-translate the static template; append the dynamic size after
            # so the message still participates in localization (update_status
            # only translates whole-string keys).
            update_status(
                _("Capture resolution set (applies on next Live start):")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("Face detection size set (analyser re-inits on next call):")
                + f" {v}x{v}"
            )

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(path, (200, 200)))
            self.source_label.setText("")
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_select_target(self) -> None:
        self._camera_mode = False
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(render_image_preview(temp_path, (200, 200)))
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (200, 200)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_camera_change(self, _index):
        self._camera_mode = True
        self.btn_start.setToolTip(_("Start live processing with the selected camera"))

    def _on_start(self) -> None:
        """Start always means the selected live webcam, even with a target loaded."""
        self._on_live()

    def _on_process_file(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            def analyse():
                modules.globals.source_target_map = []
                update_status("Getting unique faces")
                if is_image(modules.globals.target_path):
                    get_unique_faces_from_target_image()
                elif is_video(modules.globals.target_path):
                    get_unique_faces_from_target_video()
                return modules.globals.source_target_map
            self._run_task(analyse, self._mapping_ready)
        else:
            self._select_output_and_start()

    def _mapping_ready(self, mapping):
        if mapping:
            _open_mapper_dialog(lambda: self._run_task(self._start_cb), mapping)
        else:
            update_status("No faces found in target")

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._run_task(self._start_cb)

    def _on_toggle_preview(self) -> None:
        self._on_live()

    def _on_file_preview(self) -> None:
        if _WEBCAM_PREVIEW is not None:
            _WEBCAM_PREVIEW.show()
            _WEBCAM_PREVIEW.raise_()
            return
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            if (getattr(_PREVIEW, "_saved_signature", None) ==
                    (modules.globals.source_path, modules.globals.target_path)):
                _PREVIEW.display_frame(_PREVIEW._saved_frame)
            else:
                _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        if getattr(self, "_task", None) is not None:
            update_status("Wait for file processing to finish.")
            return
        self._camera_mode = True
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if _PREVIEW is not None:
            _PREVIEW.hide()
        if not modules.globals.source_path:
            update_status("Please select a source image first")
            return
        _open_webcam_preview(camera_index)

    def _run_task(self, task, result_cb=None):
        if getattr(self, "_task", None) is not None:
            update_status("Processing is already running.")
            return
        if _WEBCAM_PREVIEW is not None:
            update_status("Stop the live camera before processing files.")
            return
        self._set_busy(True)
        self._task = _TaskWorker(task)
        if result_cb is not None:
            self._task.resultReady.connect(result_cb)
        self._task.finished.connect(self._task_finished)
        self._task.start()

    def _set_busy(self, busy):
        if busy:
            self._busy_widgets = [(widget, widget.isEnabled())
                for widget in self.centralWidget().findChildren(QWidget)
                if isinstance(widget, (QPushButton, QComboBox, QCheckBox, QSlider))]
            for widget, _enabled in self._busy_widgets:
                widget.setEnabled(False)
            self.btn_destroy.setEnabled(True)
        else:
            for widget, enabled in getattr(self, "_busy_widgets", []):
                widget.setEnabled(enabled)
            self._busy_widgets = []

    def _task_finished(self):
        self._task.deleteLater()
        self._task = None
        self._set_busy(False)

    def _on_stop(self):
        if _WEBCAM_PREVIEW is not None:
            _WEBCAM_PREVIEW.close()
        else:
            self.close()

    def closeEvent(self, event):
        if _WEBCAM_PREVIEW is not None:
            _WEBCAM_PREVIEW.close()
        if _WEBCAM_PREVIEW is not None or getattr(self, "_task", None) is not None:
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if _PREVIEW is not None:
            _PREVIEW.close()
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    # The processing worker refreshes the processor list between frames.


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def display_saved_image(self, frame):
        self._saved_signature = (modules.globals.source_path, modules.globals.target_path)
        self._saved_frame = frame.copy()
        self._slider.hide()
        self.display_frame(frame)

    def display_frame(self, frame):
        if frame is None:
            return
        self._image_label.setPixmap(_bgr_to_qpixmap(
            fit_image_to_size(frame, max(self.width(), 1), max(self.height(), 1))))
        self.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        source_path = modules.globals.source_path
        target_path = modules.globals.target_path

        def process():
            update_status("Processing preview...")
            frame = (imread_unicode(target_path) if is_image(target_path)
                     else get_video_frame(target_path, frame_number))
            if frame is None:
                raise ValueError("Could not read target frame")
            if modules.globals.nsfw_filter and check_and_ignore_nsfw(frame):
                return None
            source = get_one_face(imread_unicode(source_path))
            if source is None:
                raise ValueError("No face found in source image")
            for fp in get_frame_processors_modules(modules.globals.frame_processors):
                if modules.globals.map_faces:
                    frame = (fp.process_frame_v2(frame, target_path)
                             if fp.NAME == "DLC.FACE-SWAPPER" else fp.process_frame_v2(frame))
                else:
                    frame = fp.process_frame(source, frame)
            update_status("Preview processing completed.")
            return frame

        _MAIN._run_task(process, self.display_frame)


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Own the camera entirely on the capture thread, including release."""
    startedCapture = Signal()

    def __init__(self, camera_index, capture_queue, stop_event, resolution):
        super().__init__()
        self._index = camera_index
        self._queue = capture_queue
        self._stop = stop_event
        self._resolution = resolution

    def run(self):
        cap = VideoCapturer(self._index)
        try:
            if not cap.start(*self._resolution, 30):
                raise RuntimeError(f"Cannot open selected camera {self._index} (Windows capture backends). Check camera permissions or another application using it.")
            self.startedCapture.emit()
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Camera stopped delivering frames")
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        pass
        except Exception as exc:
            update_status(str(exc))
        finally:
            self._stop.set()
            cap.release()


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        try:
            self._process()
        except Exception as exc:
            logging.exception("Live processing failed")
            update_status(f"Live processing failed: {exc}")
        finally:
            self._stop.set()

    def _process(self):
        from modules.processors.frame.face_swapper import get_face_swapper
        processors = list(get_frame_processors_modules(modules.globals.frame_processors))
        if not any(fp.NAME == "DLC.FACE-SWAPPER" for fp in processors):
            raise RuntimeError("Live mode requires the face_swapper processor")
        swapper = get_face_swapper()
        if swapper is None:
            raise RuntimeError("Face swapper initialization failed")
        self.actual_provider = swapper.session.get_providers()[0]
        self.processed_frames = self.face_frames = 0
        self.face_processing_seconds = 0.0
        self.measured_fps = 0.0
        source_face = None
        last_source_path = None
        previous_time = time.perf_counter()
        interval_frames = 0
        no_face_logged = False
        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue
            source_path = modules.globals.source_path
            if source_face is None or source_path != last_source_path:
                source_pixels = imread_unicode(source_path) if source_path else None
                if source_pixels is None:
                    print("[webcam] source face detected = NO", flush=True)
                    raise ValueError("Selected source image could not be read")
                source_face = get_one_face(source_pixels)
                if source_face is None:
                    print("[webcam] source face detected = NO", flush=True)
                    raise ValueError("No face detected in selected source image")
                last_source_path = source_path
                print(f"[webcam] Cached selected source face: {source_path!r}", flush=True)
            started = time.perf_counter()
            temp_frame = frame
            h, w = temp_frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                temp_frame = cv2.resize(temp_frame, (round(w * scale), round(h * scale)))
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)
            # The camera is always the target. Saved file mappings do not apply here.
            if modules.globals.many_faces:
                faces = detect_many_faces_fast(temp_frame) or []
            else:
                face = detect_one_face_fast(temp_frame)
                faces = [face] if face is not None else []
            baseline = temp_frame.copy() if faces else None
            swap_applied = False
            if faces:
                if modules.globals.mouth_mask:
                    ensure_landmarks(temp_frame, faces)
                processors = list(get_frame_processors_modules(modules.globals.frame_processors))
                fp = next((p for p in processors if p.NAME == "DLC.FACE-SWAPPER"), None)
                if fp is None:
                    raise RuntimeError("Live face_swapper processor was removed")
                for target_face in faces:
                    temp_frame = fp.swap_face(source_face, target_face, temp_frame)
                    if not isinstance(temp_frame, np.ndarray) or temp_frame.shape != baseline.shape:
                        raise RuntimeError("Face swap returned an invalid frame")
                swap_applied = bool(np.any(temp_frame != baseline))
                if not swap_applied:
                    print("[webcam] source face detected = YES | target face detected = YES | swap applied = NO", flush=True)
                    raise RuntimeError("Swap returned unchanged camera pixels; check source identity and opacity")
                temp_frame = fp.apply_post_processing(temp_frame, [f.bbox.astype(int) for f in faces])
                for enhancer in processors:
                    keys = {"DLC.FACE-ENHANCER": "face_enhancer",
                            "DLC.FACE-ENHANCER-GPEN256": "face_enhancer_gpen256",
                            "DLC.FACE-ENHANCER-GPEN512": "face_enhancer_gpen512"}
                    key = keys.get(enhancer.NAME)
                    if key and modules.globals.fp_ui.get(key, False):
                        temp_frame = enhancer.process_frame(None, temp_frame, detected_faces=faces)
                no_face_logged = False
                self.face_frames += 1
                self.face_processing_seconds += time.perf_counter() - started
            elif not no_face_logged:
                print("[webcam] No target face detected", flush=True)
                no_face_logged = True
            self.last_swap_diagnostics = {
                "source_face_detected": source_face is not None,
                "target_faces": len(faces), "swap_applied": swap_applied,
                "changed_values": int(np.count_nonzero(temp_frame != baseline)) if faces else 0,
                "shape": temp_frame.shape,
            }
            if self.processed_frames == 0:
                print(f"[webcam] source face detected = YES | target face detected = {'YES' if faces else 'NO'} | swap applied = {'YES' if swap_applied else 'NO'}", flush=True)
            if swap_applied:
                self.last_successful_swap_diagnostics = self.last_swap_diagnostics.copy()
                if os.environ.get("DLC_DEBUG") == "1" and not getattr(self, "_saved_physical_frame", False):
                    os.makedirs("diagnostics", exist_ok=True)
                    cv2.imwrite(os.path.join("diagnostics", "physical-original-frame.png"), baseline)
                    cv2.imwrite(os.path.join("diagnostics", "physical-swapped-frame.png"), temp_frame)
                    self._saved_physical_frame = True
            now = time.perf_counter()
            self.processed_frames += 1
            interval_frames += 1
            if now - previous_time >= 0.5:
                self.measured_fps = interval_frames / (now - previous_time)
                previous_time, interval_frames = now, 0
            provider = self.actual_provider.replace("ExecutionProvider", "")
            status = f"LIVE | FPS: {self.measured_fps:.1f} | Provider: {provider}"
            update_status(status + (" | No target face detected" if not faces else ""))
            if modules.globals.show_fps:
                cv2.putText(temp_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                self._pq.put_nowait(temp_frame)



class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)
        self._capture_queue = queue.Queue(maxsize=1)
        self._processed_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._closing = False
        self._capture_worker = _CaptureWorker(camera_index, self._capture_queue, self._stop_event,
                                              tuple(modules.globals.capture_resolution))
        self._processing_worker = _ProcessingWorker(self._capture_queue, self._processed_queue, self._stop_event, 30)
        self._capture_worker.startedCapture.connect(self._camera_started)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self._capture_worker.start()
        self._processing_worker.start()
        update_status("Opening camera and loading models...")

    def _camera_started(self):
        if not self._closing:
            update_status("Camera opened; waiting for processed frames...")

    def _tick(self):
        if self._stop_event.is_set():
            self.close()
            return
        try:
            frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        frame = fit_image_to_size(frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(frame))
        self._received_frame = True

    def closeEvent(self, event):
        self._closing = True
        self._stop_event.set()
        self.hide()
        if self._capture_worker.isRunning() or self._processing_worker.isRunning():
            event.ignore()
            return
        self._timer.stop()
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        if _MAIN is not None:
            _MAIN.btn_start.setEnabled(True)
            _MAIN.btn_live.setEnabled(bool(_MAIN._camera_indices))
            _MAIN.cb_camera.setEnabled(bool(_MAIN._camera_indices))
            _MAIN.btn_destroy.setText(_("Destroy"))
            _MAIN.cb_det_size.setEnabled(True)
            _MAIN.sw_map_faces.setEnabled(True)
        event.accept()


def _open_webcam_preview(camera_index):
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        if _WEBCAM_PREVIEW._closing:
            update_status("Camera is stopping; please wait for resources to be released.")
        else:
            _WEBCAM_PREVIEW.show()
            _WEBCAM_PREVIEW.raise_()
        return
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()
    if _MAIN is not None:
        _MAIN.btn_start.setEnabled(False)
        _MAIN.btn_live.setEnabled(False)
        _MAIN.cb_camera.setEnabled(False)
        _MAIN.btn_destroy.setText(_("Stop"))
        _MAIN.cb_det_size.setEnabled(False)
        _MAIN.sw_map_faces.setEnabled(False)


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
