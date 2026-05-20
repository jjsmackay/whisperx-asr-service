"""
Subprocess entry point for WhisperX inference.

All GPU/CUDA-touching imports (torch, whisperx, faster_whisper) are
deferred inside worker_main() so that the parent uvicorn process — which
imports this module to dispatch the child — never initialises CUDA.
"""
from __future__ import annotations

import logging
import traceback
from multiprocessing import Queue

log = logging.getLogger(__name__)


def worker_main(request_q: "Queue", response_q: "Queue") -> None:
    """Run inside the child process. Loops on the request queue until a
    `None` sentinel arrives, then exits so the OS reclaims the CUDA context.
    """
    # Deferred GPU imports — child process only.
    import whisperx
    from app.pipeline import run_pipeline

    log.info("worker started; pipeline imports deferred until first request")

    while True:
        msg = request_q.get()
        if msg is None:
            log.info("worker received shutdown sentinel; exiting")
            return

        request_id, audio_path, kwargs = msg
        try:
            audio = whisperx.load_audio(audio_path)
            result, embeddings = run_pipeline(audio, **kwargs)
            response_q.put((request_id, None, result, embeddings))
        except Exception:
            response_q.put((request_id, traceback.format_exc(), None, None))
