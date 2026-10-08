"""Provider validation must detect per-model silent CUDA fallback."""
import unittest
from unittest.mock import patch
import modules.globals as g
from modules import runtime

class Session:
    def __init__(self, providers):
        self.providers = providers
    def get_providers(self):
        return self.providers

class ProviderTests(unittest.TestCase):
    def test_cpu_only_session_rejected_when_cuda_requested(self):
        with patch.object(g, "execution_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "failed to initialize CUDA"):
                runtime.verify_session(Session(["CPUExecutionProvider"]), "Swapper")

    def test_cuda_session_accepted_with_cpu_node_fallback(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with patch.object(g, "execution_providers", providers):
            self.assertEqual(runtime.verify_session(Session(providers), "Detection"), providers)

    def test_explicit_cpu_remains_supported(self):
        with patch.object(g, "execution_providers", ["CPUExecutionProvider"]):
            self.assertEqual(runtime.validated_providers(), ["CPUExecutionProvider"])

if __name__ == "__main__":
    unittest.main()
