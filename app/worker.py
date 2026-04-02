"""
Subprocess entry point for GPU inference.

All GPU imports are deferred inside worker_main() so CUDA never initialises
in the parent uvicorn process.  When this process exits (on WORKER_KEEP_ALIVE
expiry or shutdown), the OS reclaims the full CUDA context (~346 MiB residual
VRAM) in addition to all model allocations.
"""


def worker_main(request_q, response_q):
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = _logging.getLogger(__name__)
    _logger.info("Worker process started, importing pipeline...")

    import whisperx
    from app.pipeline import run_pipeline
    _logger.info("Worker process ready")

    while True:
        msg = request_q.get()           # blocks until a request arrives
        if msg is None:                 # shutdown sentinel
            break
        request_id, audio_path, kwargs = msg
        try:
            audio = whisperx.load_audio(audio_path)
            result, embeddings = run_pipeline(audio, **kwargs)
            result["_duration"] = len(audio) / 16000   # for openai_compat verbose_json
            response_q.put((request_id, None, result, embeddings))
        except Exception:
            import traceback
            response_q.put((request_id, traceback.format_exc(), None, None))
