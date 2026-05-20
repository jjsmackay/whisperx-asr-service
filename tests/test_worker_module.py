"""Verify app.worker does not pull in GPU/CUDA modules at import time."""
import sys


def test_worker_module_imports_without_torch():
    # Drop any previously cached heavy modules first
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
        if mod == "whisperx" or mod.startswith("whisperx."):
            del sys.modules[mod]
        if mod == "app.worker":
            del sys.modules[mod]

    import app.worker  # noqa: F401

    assert "torch" not in sys.modules, "torch must not be imported by app.worker"
    assert "whisperx" not in sys.modules, "whisperx must not be imported by app.worker"


def test_worker_module_exposes_worker_main():
    import app.worker
    assert callable(app.worker.worker_main)
