"""Bounded FIFO execution on one persistent model-owning thread."""
import os
from concurrent.futures import Future, TimeoutError
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import perf_counter

from .engine import OCRBusy


class ModelQueue:
    def __init__(self, name, capacity=None, wait_timeout=None):
        self.capacity = max(1, int(capacity if capacity is not None else os.getenv("OCR_QUEUE_SIZE", "8")))
        self.wait_timeout = float(wait_timeout if wait_timeout is not None else os.getenv("OCR_QUEUE_TIMEOUT", "120"))
        self._jobs = Queue(maxsize=self.capacity)
        self._guard = Lock()
        self._thread = None
        self._name = name
        self._closed = False
        self._active = False
        self._completed = 0

    def describe(self):
        with self._guard:
            return {"capacity": self.capacity, "pending": self._jobs.qsize(),
                    "active": self._active, "completed": self._completed}

    def run(self, function, *args, **kwargs):
        future = Future()
        submitted = perf_counter()
        with self._guard:
            if self._closed:
                raise OCRBusy("Bộ xử lý OCR đang dừng. Hãy thử lại sau ít giây.")
            if self._thread is None:
                self._thread = Thread(target=self._work, name=self._name, daemon=True)
                self._thread.start()
            try:
                self._jobs.put_nowait((future, submitted, function, args, kwargs))
            except Full:
                raise OCRBusy("Hàng đợi OCR đã đầy. Hãy thử lại sau ít giây.") from None
        try:
            return future.result(timeout=self.wait_timeout)
        except TimeoutError:
            # A cancelled pending job will never touch the request's pixels.
            # If inference has started, retain ownership until it finishes.
            if future.cancel():
                raise OCRBusy("Đã hết thời gian chờ trong hàng đợi OCR. Hãy thử lại sau.") from None
            return future.result()

    def _work(self):
        while True:
            try:
                future, submitted, function, args, kwargs = self._jobs.get(timeout=0.2)
            except Empty:
                if self._closed:
                    return
                continue
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                with self._guard:
                    self._active = True
                started = perf_counter()
                error = None
                value = None
                try:
                    value = function(*args, **kwargs)
                    finished = perf_counter()
                    timings = {
                        "queue_wait": round((started - submitted) * 1000, 2),
                        "inference": round((finished - started) * 1000, 2),
                    }
                except BaseException as exc:
                    error = exc
                finally:
                    with self._guard:
                        self._active = False
                        self._completed += 1
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result((value, timings))
            finally:
                self._jobs.task_done()
                # Do not retain the last request's pixels, OCR text or exception
                # traceback in this long-lived thread while the queue is idle.
                future = function = args = kwargs = value = error = None

    def close(self):
        with self._guard:
            self._closed = True
