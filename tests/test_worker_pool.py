"""Lifecycle tests for WorkerPool using an injected stub entry function."""
import multiprocessing as mp
import time

import pytest

from app.worker_pool import WorkerPool


def echo_entry(request_q, response_q):
    """Stub worker — no GPU, just echoes back."""
    while True:
        msg = request_q.get()
        if msg is None:
            return
        request_id, audio_path, kwargs = msg
        response_q.put((request_id, None, {"echo": audio_path, "kwargs": kwargs}, None))


def test_pool_round_trip_returns_result():
    pool = WorkerPool(worker_entry=echo_entry, keep_alive_seconds=-1)
    try:
        result, embeddings = pool.submit("hello.wav", language="en")
        assert result == {"echo": "hello.wav", "kwargs": {"language": "en"}}
        assert embeddings is None
    finally:
        pool.shutdown()


def test_pool_keep_alive_zero_tears_down_after_request():
    pool = WorkerPool(worker_entry=echo_entry, keep_alive_seconds=0)
    try:
        pool.submit("a.wav")
        # Give the timer a moment to fire and the join to complete
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pool._process is None or not pool._process.is_alive():
                break
            time.sleep(0.1)
        assert pool._process is None or not pool._process.is_alive(), \
            "worker should have shut down after keep_alive=0"
    finally:
        pool.shutdown()


def test_pool_respawns_worker_on_next_submit_after_teardown():
    pool = WorkerPool(worker_entry=echo_entry, keep_alive_seconds=0)
    try:
        pool.submit("first.wav")
        # Wait for teardown
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pool._process is None or not pool._process.is_alive():
                break
            time.sleep(0.1)
        # Next submit should respawn cleanly
        result, _ = pool.submit("second.wav")
        assert result == {"echo": "second.wav", "kwargs": {}}
    finally:
        pool.shutdown()


@pytest.mark.asyncio
async def test_pool_asubmit_works():
    pool = WorkerPool(worker_entry=echo_entry, keep_alive_seconds=-1)
    try:
        result, _ = await pool.asubmit("async.wav", model_name="tiny")
        assert result == {"echo": "async.wav", "kwargs": {"model_name": "tiny"}}
    finally:
        pool.shutdown()


def crashing_entry(request_q, response_q):
    """Stub worker that crashes hard on the first request."""
    msg = request_q.get()
    if msg is None:
        return
    import os
    os._exit(2)


def test_pool_raises_when_worker_crashes():
    pool = WorkerPool(worker_entry=crashing_entry, keep_alive_seconds=-1)
    try:
        with pytest.raises(RuntimeError, match=r"exit code"):
            pool.submit("doomed.wav")
    finally:
        pool.shutdown()


def test_pool_recovers_after_crash():
    """After a crash, the next submit should respawn cleanly."""
    pool = WorkerPool(worker_entry=crashing_entry, keep_alive_seconds=-1)
    try:
        # First submit crashes
        with pytest.raises(RuntimeError):
            pool.submit("doom1.wav")
        # Swap in the echo entry so the next spawn works
        pool._worker_entry = echo_entry
        result, _ = pool.submit("recovery.wav")
        assert result == {"echo": "recovery.wav", "kwargs": {}}
    finally:
        pool.shutdown()
