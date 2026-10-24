import sys
import types

import pytest

from identification_demo.engine import NamLhnEngine, OCRBusy, OCRUnavailable


def test_upstream_extractor_is_initialized_once_and_reused(monkeypatch):
    engine = NamLhnEngine()
    initialized = []
    calls = []

    class Extractor:
        def __init__(self, **kwargs):
            initialized.append(kwargs)

        def extract(self, path, verbose):
            calls.append((path, verbose))
            return {"Name": "TEST"}

    monkeypatch.setitem(sys.modules, "ocr", types.SimpleNamespace(IDCardExtractor=Extractor))
    assert engine.extract("first.png") == {"Name": "TEST"}
    assert engine.extract("second.png") == {"Name": "TEST"}
    assert len(initialized) == 1
    assert initialized[0]["max_workers"] == 1
    assert calls == [("first.png", False), ("second.png", False)]
    assert engine.describe()["model_loaded"] is True


def test_parallel_inference_is_rejected_without_loading_another_model():
    engine = NamLhnEngine()
    engine._lock.acquire()
    try:
        with pytest.raises(OCRBusy):
            engine.extract("card.png")
        assert engine.describe()["model_loaded"] is False
    finally:
        engine._lock.release()


def test_failed_initialization_can_be_retried(monkeypatch):
    engine = NamLhnEngine()

    class BrokenExtractor:
        def __init__(self, **kwargs):
            raise RuntimeError("Model download failed")

    monkeypatch.setitem(sys.modules, "ocr", types.SimpleNamespace(IDCardExtractor=BrokenExtractor))
    for _ in range(2):
        with pytest.raises(OCRUnavailable):
            engine.extract("card.png")
        assert not engine._lock.locked()
        assert engine.describe()["model_loaded"] is False
