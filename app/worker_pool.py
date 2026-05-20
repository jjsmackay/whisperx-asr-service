"""
WorkerPool — single-subprocess manager for VRAM-isolated transcription.

Spawns one child process (via the "spawn" multiprocessing context — no fork
inheritance issues with CUDA) on first request. After
WORKER_KEEP_ALIVE_SECONDS of idleness, sends a shutdown sentinel and joins
the child so the OS reclaims the CUDA context.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import threading
import uuid
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger(__name__)

_SHUTDOWN_SENTINEL = None


def _default_worker_entry(request_q, response_q):
    """Indirection so we can lazily import app.worker without forcing
    GPU/heavy imports at WorkerPool module load."""
    from app.worker import worker_main
    worker_main(request_q, response_q)


class WorkerPool:
    """Manages a single subprocess worker with idle teardown.

    Parameters
    ----------
    worker_entry:
        Callable run in the child process. Defaults to app.worker.worker_main.
        Tests can inject a stub.
    keep_alive_seconds:
        After the last successful submit, idle window before tearing the
        worker down. 0 = tear down immediately after each request.
        -1 = never auto-teardown (useful for tests). Defaults to env var
        WORKER_KEEP_ALIVE_SECONDS, or 0 if unset.
    response_timeout:
        Per-request hard cap on waiting for a response (seconds).
    """

    def __init__(
        self,
        worker_entry: Optional[Callable] = None,
        keep_alive_seconds: Optional[float] = None,
        response_timeout: float = 600.0,
    ) -> None:
        self._worker_entry = worker_entry or _default_worker_entry
        if keep_alive_seconds is None:
            keep_alive_seconds = float(os.getenv("WORKER_KEEP_ALIVE_SECONDS", "0"))
        self._keep_alive = float(keep_alive_seconds)
        self._response_timeout = response_timeout

        self._ctx = mp.get_context("spawn")
        self._lifecycle_lock = threading.Lock()
        self._timer_lock = threading.Lock()
        self._submit_lock = threading.Lock()  # serialize submits (single worker)

        self._process: Optional[mp.process.BaseProcess] = None
        self._request_q = None
        self._response_q = None
        self._timer: Optional[threading.Timer] = None

    # ----- public API ---------------------------------------------------

    def submit(self, audio_path: str, **kwargs: Any) -> Tuple[Any, Any]:
        """Blocking submit. Returns (result, embeddings) or raises."""
        with self._submit_lock:
            self._cancel_timer()
            self._ensure_worker()

            request_id = str(uuid.uuid4())
            self._request_q.put((request_id, audio_path, kwargs))

            response = self._await_response(request_id)
            self._reset_keepalive_timer()
            return response

    async def asubmit(self, audio_path: str, **kwargs: Any) -> Tuple[Any, Any]:
        """Async wrapper — runs the blocking submit on a thread."""
        return await asyncio.to_thread(self.submit, audio_path, **kwargs)

    def shutdown(self) -> None:
        """Tear down the worker and cancel any pending timer."""
        with self._lifecycle_lock:
            self._cancel_timer()
            self._shutdown_worker_locked()

    # ----- internals ----------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lifecycle_lock:
            if self._process is not None and self._process.is_alive():
                return
            # Stale dead process — clean up before respawning
            if self._process is not None:
                self._cleanup_process_locked()

            self._request_q = self._ctx.Queue()
            self._response_q = self._ctx.Queue()
            self._process = self._ctx.Process(
                target=self._worker_entry,
                args=(self._request_q, self._response_q),
                name="whisperx-worker",
                daemon=False,
            )
            self._process.start()
            log.info("worker spawned pid=%s", self._process.pid)

    def _await_response(self, request_id: str) -> Tuple[Any, Any]:
        import queue as queue_mod

        deadline = None
        if self._response_timeout > 0:
            import time
            deadline = time.monotonic() + self._response_timeout

        poll = 0.5
        while True:
            try:
                rid, err_tb, result, embeddings = self._response_q.get(timeout=poll)
            except queue_mod.Empty:
                if not self._process.is_alive():
                    exitcode = self._process.exitcode
                    self._cleanup_process_locked()
                    raise RuntimeError(
                        f"worker subprocess exited unexpectedly (exit code {exitcode})"
                    )
                if deadline is not None:
                    import time
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"worker did not respond within {self._response_timeout}s"
                        )
                continue

            if rid != request_id:
                # Out-of-order response from a stale request — skip
                continue
            if err_tb is not None:
                raise RuntimeError(f"worker error:\n{err_tb}")
            return result, embeddings

    def _reset_keepalive_timer(self) -> None:
        if self._keep_alive < 0:
            return
        if self._keep_alive == 0:
            # Fire-and-forget teardown
            threading.Thread(target=self.shutdown, daemon=True).start()
            return
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            t = threading.Timer(self._keep_alive, self.shutdown)
            t.daemon = True
            t.start()
            self._timer = t

    def _cancel_timer(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _shutdown_worker_locked(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._request_q.put(_SHUTDOWN_SENTINEL)
            except Exception:
                pass
            self._process.join(timeout=10)
            if self._process.is_alive():
                log.warning("worker did not exit gracefully; terminating")
                self._process.terminate()
                self._process.join(timeout=10)
            if self._process.is_alive():
                log.warning("worker still alive after terminate; killing")
                self._process.kill()
                self._process.join(timeout=3)
        log.info("worker exited pid=%s exitcode=%s",
                 self._process.pid, self._process.exitcode)
        self._cleanup_process_locked()

    def _cleanup_process_locked(self) -> None:
        self._process = None
        self._request_q = None
        self._response_q = None
