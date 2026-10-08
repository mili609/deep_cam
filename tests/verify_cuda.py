"""Direct CUDA allocation and ONNX inference; fail if ORT falls back to CPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto
print("ORT", ort.__version__, ort.__file__, flush=True)
print("AVAILABLE", ort.get_available_providers(), flush=True)
ort.preload_dlls()
graph = helper.make_graph([helper.make_node("MatMul", ["x", "x"], ["y"])], "cuda_verify",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ACTUAL SESSION", session.get_providers(), flush=True)
if "CUDAExecutionProvider" not in session.get_providers():
    raise SystemExit("CUDA VERIFICATION FAILED: session initialized on CPU; see native driver/dependency error above")
value = ort.OrtValue.ortvalue_from_numpy(np.ones((2, 2), dtype=np.float32), "cuda", 0)
binding = session.io_binding()
binding.bind_ortvalue_input("x", value)
binding.bind_output("y", "cuda", 0)
session.run_with_iobinding(binding)
np.testing.assert_allclose(binding.copy_outputs_to_cpu()[0], np.full((2, 2), 2, dtype=np.float32))
print("CUDA VERIFIED: GPU allocation and MatMul inference passed", flush=True)
