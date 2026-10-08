from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_utilities_module():
    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_threads = 0
    fake_globals.log_level = "error"
    fake_globals.execution_providers = []
    fake_globals.video_encoder = "libx264"
    fake_globals.video_quality = 23

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None))

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.uint8 = int
    fake_numpy.fromfile = lambda *args, **kwargs: b""

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    with mock.patch.dict(
        sys.modules,
        {
            "modules.globals": fake_globals,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "tqdm": fake_tqdm,
        },
        clear=False,
    ):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.utilities", None)
        utilities = importlib.import_module("modules.utilities")
        utilities.modules.globals = fake_globals
        return utilities


class DetectFpsTests(unittest.TestCase):
    def test_detect_fps_returns_fraction_result(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="30000/1001\n") as check:
            self.assertAlmostEqual(utilities.detect_fps("demo.mp4"), 30000 / 1001)

        check.assert_called_once_with(mock.ANY, encoding="utf-8")

    def test_detect_fps_falls_back_on_command_failure(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=utilities.subprocess.CalledProcessError(1, ["ffprobe"]),
        ):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_malformed_output(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="not-a-fraction"):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_zero_denominator(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="30/0"):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)


class FfmpegTests(unittest.TestCase):
    def test_no_invalid_auto_pixel_format_is_injected(self):
        utilities = _load_utilities_module()
        with mock.patch.object(utilities.subprocess, "check_output", return_value=b"") as check:
            self.assertTrue(utilities.run_ffmpeg(["-i", "input.mp4", "-c:v", "copy", "output.mp4"]))
        command = check.call_args.args[0]
        self.assertNotIn("-hwaccel_output_format", command)
        self.assertNotIn("-hwaccel", command)

    def test_audio_map_is_optional_and_mp4_compatible(self):
        utilities = _load_utilities_module()
        with mock.patch.object(utilities, "run_ffmpeg", return_value=True) as run:
            utilities.restore_audio("silent.mp4", "output.mp4")
        command = run.call_args.args[0]
        self.assertIn("1:a:0?", command)
        self.assertEqual(command[command.index("-c:a")+1], "aac")

if __name__ == "__main__":
    unittest.main()
