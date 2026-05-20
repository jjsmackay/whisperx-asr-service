"""/asr must delegate transcription to app.state.worker_pool, not run_in_queue."""
import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_module_stubs() -> None:
    """Install lightweight stubs so app.main can import in the test env."""
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _CudaStub:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def memory_allocated() -> int:
                return 0

            @staticmethod
            def empty_cache() -> None:
                return None

        torch_stub.cuda = _CudaStub()
        torch_stub.device = lambda *args, **kwargs: None
        sys.modules["torch"] = torch_stub

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = type("ndarray", (), {})
        numpy_stub.floating = type("floating", (), {})
        numpy_stub.integer = type("integer", (), {})
        sys.modules["numpy"] = numpy_stub

    if "whisperx" not in sys.modules:
        whisperx_stub = types.ModuleType("whisperx")
        whisperx_stub.load_model = lambda *a, **kw: None
        whisperx_stub.load_align_model = lambda *a, **kw: (None, None)
        whisperx_stub.align = lambda *a, **kw: None
        whisperx_stub.assign_word_speakers = lambda *a, **kw: None
        sys.modules["whisperx"] = whisperx_stub

        diarize_stub = types.ModuleType("whisperx.diarize")
        diarize_stub.DiarizationPipeline = type("DiarizationPipeline", (), {})
        sys.modules["whisperx.diarize"] = diarize_stub
        whisperx_stub.diarize = diarize_stub

    # Always (re)install load_audio so it returns a deterministic length
    class _FakeArr:
        def __len__(self):
            return 16000

    sys.modules["whisperx"].load_audio = lambda path: _FakeArr()


@pytest.fixture
def stubbed_app(monkeypatch):
    """Build a TestClient against app.main with WorkerPool replaced by a stub
    whose asubmit is an AsyncMock.
    """
    # Another test (test_align_diarize_eviction.py) may have left an
    # incomplete prometheus_client stub in sys.modules — drop it so the real
    # package (installed in the test venv) is used, and reload our metrics
    # module so app.main sees real Counter/Gauge/Histogram with full APIs.
    prom_stub = sys.modules.get("prometheus_client")
    if prom_stub is not None and not hasattr(prom_stub, "REGISTRY"):
        del sys.modules["prometheus_client"]
        if "app.metrics" in sys.modules:
            importlib.reload(sys.modules["app.metrics"])

    _install_heavy_module_stubs()

    # Reload app.main so stubs and a fresh app.state are used
    if "app.main" in sys.modules:
        importlib.reload(sys.modules["app.main"])
    import app.main as main_mod

    stub_pool = MagicMock()
    stub_pool.asubmit = AsyncMock(return_value=(
        {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "word_segments": [],
        },
        None,
    ))
    stub_pool.shutdown = MagicMock()

    # Patch the WorkerPool symbol in app.main so the lifespan constructs our stub
    monkeypatch.setattr(main_mod, "WorkerPool", lambda *a, **kw: stub_pool)

    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as client:
        yield client, stub_pool


def test_asr_calls_worker_pool_asubmit(stubbed_app):
    client, pool = stubbed_app
    files = {"audio_file": ("sample.wav", b"\x00\x00\x00\x00", "audio/wav")}
    r = client.post("/asr?model=tiny&language=en&diarize=false", files=files)
    assert r.status_code == 200, r.text
    assert pool.asubmit.await_count == 1
    args, kwargs = pool.asubmit.await_args
    # First positional arg is the temp audio path (a string), NOT a numpy array
    assert isinstance(args[0], str)
    assert kwargs.get("model_name") == "tiny"
    assert kwargs.get("language") == "en"
    assert kwargs.get("should_diarize") is False
