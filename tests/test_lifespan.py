"""Behavior test for app startup: lifespan/on_event must run and /health must respond.

The real `app.main` transitively imports heavyweight ML libraries (torch,
numpy, whisperx). These are not present in the test environment because the
PR under test only touches FastAPI startup wiring. We therefore install
lightweight `sys.modules` shims for those modules *before* importing
`app.main`, so the import succeeds and we can drive the app with
`TestClient`. The shims are import-only; the `/health` endpoint does not
exercise any of them.
"""

import sys
import types


def _install_heavy_module_stubs() -> None:
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
        whisperx_stub.load_audio = lambda *a, **kw: None
        whisperx_stub.load_model = lambda *a, **kw: None
        whisperx_stub.load_align_model = lambda *a, **kw: (None, None)
        whisperx_stub.align = lambda *a, **kw: None
        whisperx_stub.assign_word_speakers = lambda *a, **kw: None
        sys.modules["whisperx"] = whisperx_stub

        diarize_stub = types.ModuleType("whisperx.diarize")
        diarize_stub.DiarizationPipeline = type("DiarizationPipeline", (), {})
        sys.modules["whisperx.diarize"] = diarize_stub
        whisperx_stub.diarize = diarize_stub


_install_heavy_module_stubs()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_lifespan_runs_and_health_responds():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
