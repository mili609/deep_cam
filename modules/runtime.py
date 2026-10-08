"""Shared, bounded ONNX sessions and a one-time CUDA availability check."""
import os
import threading
import onnxruntime as ort
import modules.globals

_LOCK = threading.Lock()
_CUDA_READY = None
CUDA_FAILURE = None
GPU_NAME = None


def session_options():
    options = ort.SessionOptions()
    # Multiple default ORT pools oversubscribe small CPUs badly.
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def validated_providers(providers=None):
    global _CUDA_READY, CUDA_FAILURE, GPU_NAME
    selected = list(providers if providers is not None else modules.globals.execution_providers)
    names = [p[0] if isinstance(p, tuple) else p for p in selected]
    if "CUDAExecutionProvider" in names:
        with _LOCK:
            if _CUDA_READY is None:
                try:
                    if os.name == "nt":
                        import ctypes
                        driver_path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll")
                        if not os.path.isfile(driver_path):
                            raise RuntimeError(f"NVIDIA CUDA driver is unavailable: {driver_path} is missing. CUDA requires an NVIDIA GPU and its installed display driver.")
                        driver = ctypes.WinDLL(driver_path)
                        result = driver.cuInit(0)
                        if result:
                            raise RuntimeError(f"NVIDIA driver cuInit failed with CUDA error {result}")
                        device = ctypes.c_int()
                        result = driver.cuDeviceGet(ctypes.byref(device), 0)
                        if result:
                            raise RuntimeError(f"NVIDIA driver cuDeviceGet failed with CUDA error {result}")
                        name = ctypes.create_string_buffer(256)
                        driver.cuDeviceGetName(name, len(name), device)
                        GPU_NAME = name.value.decode(errors="replace")
                    if hasattr(ort, "preload_dlls"):
                        ort.preload_dlls()
                    from onnx import helper, TensorProto
                    graph = helper.make_graph(
                        [helper.make_node("Identity", ["x"], ["y"])], "cuda_probe",
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
                    model.ir_version = 8
                    probe = ort.InferenceSession(model.SerializeToString(),
                        sess_options=session_options(),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    _CUDA_READY = "CUDAExecutionProvider" in probe.get_providers()
                    if not _CUDA_READY:
                        raise RuntimeError("ONNX Runtime rejected CUDAExecutionProvider. See the native ORT error above for missing CUDA/cuDNN DLLs or driver incompatibility.")
                    if _CUDA_READY:
                        import numpy as np
                        probe.run(None, {"x": np.ones(1, dtype=np.float32)})
                except Exception as exc:
                    CUDA_FAILURE = str(exc)
                    print(f"[DLC.RUNTIME] CUDA initialization failed: {CUDA_FAILURE}", flush=True)
                    _CUDA_READY = False
                if not _CUDA_READY:
                    print("[DLC.RUNTIME] CUDA initialization failed; using CPU fallback.")
        if not _CUDA_READY:
            selected = [p for p in selected if (p[0] if isinstance(p, tuple) else p) != "CUDAExecutionProvider"]
    selected = [p for p in selected if (p[0] if isinstance(p, tuple) else p) in ort.get_available_providers()]
    if "CPUExecutionProvider" not in [p[0] if isinstance(p, tuple) else p for p in selected]:
        selected.append("CPUExecutionProvider")
    if providers is None:
        modules.globals.execution_providers = selected
    return selected


def verify_session(session, label):
    """Never label a CPU-only model session as CUDA."""
    active = session.get_providers()
    requested = [p[0] if isinstance(p, tuple) else p
                 for p in modules.globals.execution_providers]
    print(f"[DLC.RUNTIME] {label}: {active}", flush=True)
    if "CUDAExecutionProvider" in requested and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"{label} failed to initialize CUDA; actual providers: {active}. "
                           "See ONNX Runtime's native CUDA error above.")
    return active


def initialize_execution():
    """Resolve the provider once and report actual initialization, not build support."""
    from importlib.metadata import version, PackageNotFoundError
    installed = []
    for package in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"):
        try:
            installed.append(f"{package}={version(package)}")
        except PackageNotFoundError:
            pass
    print(f"[DLC.RUNTIME] Python: {os.sys.executable}", flush=True)
    print(f"[DLC.RUNTIME] ORT: {ort.__version__}, packages: {', '.join(installed)}", flush=True)
    print(f"[DLC.RUNTIME] Available providers: {ort.get_available_providers()}", flush=True)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[DLC.RUNTIME] CUDA is absent from this ONNX Runtime build. "
              "Use the project venv with onnxruntime-gpu; a CPU-only package may shadow it.", flush=True)
    providers = validated_providers()
    print(f"GPU: {GPU_NAME or 'NVIDIA CUDA unavailable (see diagnostic above)'}", flush=True)
    print(f"Execution provider: {providers[0]}", flush=True)
    return providers
