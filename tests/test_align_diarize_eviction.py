"""
Tests that idle align models and the diarize pipeline are evicted by the
MODEL_KEEP_ALIVE_SECONDS sweep, mirroring the existing whisper-model behavior.

These tests stub out the heavy ML dependencies (torch / numpy / whisperx /
prometheus_client) via sys.modules injection so the test suite can run in a
minimal environment without GPU or model downloads.
"""
import importlib
import sys
import time
import types
from unittest.mock import MagicMock, patch


def _install_heavy_module_stubs():
    """Insert lightweight stand-ins for the heavy ML deps into sys.modules.

    Must run before `import app.pipeline` (or its reload) so the module's
    top-level `import whisperx` etc. resolves to our stubs.
    """
    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = type("ndarray", (), {})
        numpy_stub.floating = type("floating", (), {})
        numpy_stub.integer = type("integer", (), {})
        sys.modules["numpy"] = numpy_stub

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.is_available = lambda: False
        cuda_stub.empty_cache = lambda: None
        cuda_stub.memory_allocated = lambda: 0
        torch_stub.cuda = cuda_stub
        torch_stub.device = lambda *a, **kw: ("device", a, kw)
        sys.modules["torch"] = torch_stub
        sys.modules["torch.cuda"] = cuda_stub

    if "whisperx" not in sys.modules:
        whisperx_stub = types.ModuleType("whisperx")
        whisperx_stub.load_model = MagicMock(return_value=MagicMock())
        whisperx_stub.load_align_model = MagicMock(
            return_value=(MagicMock(), {"language": "en"})
        )
        whisperx_stub.align = MagicMock(return_value={"segments": []})
        whisperx_stub.assign_word_speakers = MagicMock(return_value={})
        diarize_stub = types.ModuleType("whisperx.diarize")
        diarize_stub.DiarizationPipeline = MagicMock(return_value=MagicMock())
        whisperx_stub.diarize = diarize_stub
        sys.modules["whisperx"] = whisperx_stub
        sys.modules["whisperx.diarize"] = diarize_stub

    if "prometheus_client" not in sys.modules:
        prom_stub = types.ModuleType("prometheus_client")

        class _Metric:
            def __init__(self, *a, **kw):
                pass

            def labels(self, *a, **kw):
                return self

            def inc(self, *a, **kw):
                return None

            def set(self, *a, **kw):
                return None

            def observe(self, *a, **kw):
                return None

            def info(self, *a, **kw):
                return None

        prom_stub.Counter = _Metric
        prom_stub.Gauge = _Metric
        prom_stub.Histogram = _Metric
        prom_stub.Info = _Metric
        prom_stub.CONTENT_TYPE_LATEST = "text/plain"
        prom_stub.generate_latest = lambda *a, **kw: b""
        sys.modules["prometheus_client"] = prom_stub

    if "faster_whisper" not in sys.modules:
        fw_stub = types.ModuleType("faster_whisper")
        fw_stub.available_models = lambda: ["large-v3"]
        sys.modules["faster_whisper"] = fw_stub


_install_heavy_module_stubs()


def _fresh_pipeline(monkeypatch, sweep_interval=1, **env):
    """Reload app.pipeline with the given env vars set.

    The module enforces a 30s floor on MODEL_EVICTION_INTERVAL_SECONDS, so the
    sweep cadence is overridden directly on the reloaded module to keep the
    tests fast rather than weakening the production floor.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _install_heavy_module_stubs()
    if "app.pipeline" in sys.modules:
        p = importlib.reload(sys.modules["app.pipeline"])
    else:
        import app.pipeline as p  # noqa: F401
        p = sys.modules["app.pipeline"]
    p.MODEL_EVICTION_INTERVAL_SECONDS = sweep_interval
    return p


def test_align_model_evicted_after_keepalive(monkeypatch):
    p = _fresh_pipeline(
        monkeypatch,
        MODEL_KEEP_ALIVE_SECONDS="1",
    )
    with patch.object(
        p.whisperx,
        "load_align_model",
        return_value=(MagicMock(), {"language": "en"}),
    ):
        p.load_align_model("en")
        assert "en" in p._align_models
        time.sleep(3.5)
        assert "en" not in p._align_models


def test_diarize_pipeline_evicted_after_keepalive(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake")
    p = _fresh_pipeline(
        monkeypatch,
        MODEL_KEEP_ALIVE_SECONDS="1",
    )
    # Patch the DiarizationPipeline symbol that pipeline.py imported at module load.
    with patch.object(p, "DiarizationPipeline", return_value=MagicMock()):
        p.load_diarize_pipeline()
        assert p._diarize_pipeline is not None
        time.sleep(3.5)
        assert p._diarize_pipeline is None
