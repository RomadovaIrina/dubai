"""Exit 0 only if onnxruntime can actually run a session on CUDAExecutionProvider (sm_120 has no official SASS)."""
import sys
import numpy as np
import torch  # noqa: F401  (maps libcublas/cudnn from the torch wheels before ORT loads)
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception:
    pass
print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
if "CUDAExecutionProvider" not in ort.get_available_providers():
    sys.exit(1)
import onnx
from onnx import helper, TensorProto
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 30, 30])
W = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3], np.random.rand(4 * 3 * 9).astype(np.float32).tolist())
g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"])], "g", [X], [Y], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 9
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider"])
print("session providers:", s.get_providers())
out = s.run(None, {"x": np.random.rand(1, 3, 32, 32).astype(np.float32)})[0]
ok = s.get_providers()[0] == "CUDAExecutionProvider" and np.isfinite(out).all()
print("CUDA conv OK" if ok else "CUDA provider not active")
sys.exit(0 if ok else 1)
