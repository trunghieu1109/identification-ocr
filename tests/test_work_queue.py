from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from time import monotonic, sleep

import pytest

from identification_demo.engine import OCRBusy
from identification_demo.work_queue import ModelQueue


def pending(queue, count):
    deadline = monotonic() + 2
    while queue.describe()["pending"] != count and monotonic() < deadline:
        sleep(.005)
    assert queue.describe()["pending"] == count


def test_fifo_serializes_jobs_on_one_thread_and_reports_wait_time():
    queue = ModelQueue("test-fifo", capacity=1)
    started, release = Event(), Event()
    threads, order = [], []
    def work(value):
        threads.append(get_ident())
        order.append(value)
        if value == 1:
            started.set()
            assert release.wait(2)
        return value
    with ThreadPoolExecutor(2) as clients:
        first = clients.submit(queue.run, work, 1)
        assert started.wait(2)
        second = clients.submit(queue.run, work, 2)
        pending(queue, 1)
        with pytest.raises(OCRBusy, match="đã đầy"):
            queue.run(work, 3)
        release.set()
        assert first.result()[0] == 1
        value, timings = second.result()
        assert value == 2 and timings["queue_wait"] > 0
    assert order == [1, 2] and len(set(threads)) == 1
    assert queue.describe()["completed"] == 2
    queue.close()


def test_expired_pending_job_never_uses_released_request_data():
    queue = ModelQueue("test-timeout", capacity=2, wait_timeout=.03)
    started, release = Event(), Event()
    touched = []
    def blocker():
        started.set()
        assert release.wait(2)
    with ThreadPoolExecutor(1) as clients:
        first = clients.submit(queue.run, blocker)
        assert started.wait(2)
        with pytest.raises(OCRBusy, match="thời gian chờ"):
            queue.run(lambda: touched.append(True))
        release.set()
        first.result()
    assert queue.run(lambda: "alive")[0] == "alive"
    assert not touched
    queue.close()


def test_worker_survives_model_error_and_close_rejects_new_work():
    queue = ModelQueue("test-errors")
    def broken():
        raise RuntimeError("model error")
    with pytest.raises(RuntimeError):
        queue.run(broken)
    assert queue.run(lambda: 42)[0] == 42
    queue.close()
    with pytest.raises(OCRBusy):
        queue.run(lambda: 0)
