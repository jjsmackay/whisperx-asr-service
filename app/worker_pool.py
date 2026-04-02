"""
Manages a single GPU worker subprocess.

WORKER_KEEP_ALIVE controls how long the subprocess stays alive after the last
request.  When it exits, the OS reclaims the full CUDA context in addition to
all model VRAM — something that cannot be achieved by in-process model unloading.

MODEL_KEEP_ALIVE (in pipeline.py) independently controls how long individual
models stay loaded within a running subprocess.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.pipeline import parse_keep_alive
from app.queue import GPU_CONCURRENCY

logger = logging.getLogger(__name__)

WORKER_KEEP_ALIVE: float = parse_keep_alive(os.getenv("WORKER_KEEP_ALIVE", "-1"))


class WorkerPool:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._request_q = None
        self._response_q = None
        self._lock = threading.Lock()          # guards process/queue state
        self._timer_lock = threading.Lock()    # guards _timer only
        self._timer = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._semaphore = None                 # created lazily (needs running event loop)
        self._keep_alive = WORKER_KEEP_ALIVE
        self._concurrency = GPU_CONCURRENCY

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)
        return self._semaphore

    def _ensure_worker(self) -> None:
        """Start the worker subprocess if it is not currently running. Must be called under self._lock."""
        if self._process and self._process.is_alive():
            return
        from app.worker import worker_main
        self._request_q = self._ctx.Queue()
        self._response_q = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=worker_main,
            args=(self._request_q, self._response_q),
            daemon=True,
        )
        self._process.start()
        logger.info(f"Worker subprocess started (pid={self._process.pid})")

    def _reset_timer(self) -> None:
        """(Re)start the WORKER_KEEP_ALIVE countdown after a request completes."""
        if self._keep_alive < 0:
            return
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        if self._keep_alive == 0:
            self._shutdown_worker()
        else:
            with self._timer_lock:
                t = threading.Timer(self._keep_alive, self._shutdown_worker)
                t.daemon = True
                t.start()
                self._timer = t
            logger.debug(f"Worker keep-alive timer set: subprocess exits in {self._keep_alive}s")

    def _shutdown_worker(self) -> None:
        """Gracefully stop the worker subprocess, then force-kill if needed."""
        with self._lock:
            if not (self._process and self._process.is_alive()):
                self._process = None
                return
            logger.info(f"Shutting down worker subprocess (pid={self._process.pid})...")
            try:
                self._request_q.put(None)       # sentinel → worker breaks out of loop
                self._process.join(timeout=10)
            except Exception:
                pass
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=3)
            logger.info(
                f"Worker subprocess exited (code={self._process.exitcode}) — VRAM fully freed"
            )
            self._process = None

    def submit(self, audio_path: str, **kwargs):
        """Blocking: send a request to the worker and wait for the response."""
        import queue as _q

        request_id = str(uuid.uuid4())
        with self._lock:
            self._ensure_worker()
            self._request_q.put((request_id, audio_path, kwargs))

        # Poll with a timeout so a worker crash doesn't hang forever
        while True:
            try:
                req_id, error, result, embeddings = self._response_q.get(timeout=5.0)
                break
            except _q.Empty:
                with self._lock:
                    if not (self._process and self._process.is_alive()):
                        raise RuntimeError(
                            f"Worker subprocess died unexpectedly "
                            f"(exit={getattr(self._process, 'exitcode', '?')})"
                        )

        assert req_id == request_id, f"Response ID mismatch: {req_id} != {request_id}"
        self._reset_timer()
        if error:
            raise RuntimeError(f"Worker error:\n{error}")
        return result, embeddings

    async def asubmit(self, audio_path: str, **kwargs):
        """Async wrapper: runs submit() in the thread pool executor."""
        sem = self._get_semaphore()
        loop = asyncio.get_running_loop()
        async with sem:
            return await loop.run_in_executor(
                self._executor, lambda: self.submit(audio_path, **kwargs)
            )

    def shutdown(self) -> None:
        """Cleanly shut down the pool (called from lifespan teardown)."""
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        self._shutdown_worker()
        self._executor.shutdown(wait=False)
