from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from identification_demo.engine import OCRBusy, OCRUnavailable, OCRMultipleDocuments
from identification_demo.gpu_worker import DocULocalEngine, create_gpu_app
from test_pipeline import image_data


@pytest.mark.parametrize("model", ["vgg_transformer", "vgg_seq2seq"])
def test_configured_recognizer_uses_the_matching_checkpoint(monkeypatch, model):
    monkeypatch.setenv("GPU_OCR_RECOGNITION_MODEL", model)
    engine = DocULocalEngine()
    assert engine.models["recognition"].name == model + ".pth"
    assert engine.describe()["recognition_model"] == model
    assert engine.device == "cuda:0"
    engine.close()


def test_default_gpu_uses_seq2seq_for_both_general_text_and_authority(monkeypatch):
    monkeypatch.delenv("GPU_OCR_RECOGNITION_MODEL", raising=False)
    monkeypatch.delenv("GPU_OCR_AUTHORITY_MODEL", raising=False)
    engine = DocULocalEngine()
    assert engine.recognition_model == "vgg_seq2seq"
    assert engine.models["recognition"].name == "vgg_seq2seq.pth"
    assert engine.authority_recognition_model == engine.recognition_model
    assert engine.models["authority_recognition"] == engine.models["recognition"]
    assert engine.describe()["authority_recognition_model"] == "vgg_seq2seq"
    engine.close()


@pytest.mark.parametrize("setting", ["GPU_OCR_RECOGNITION_MODEL", "GPU_OCR_AUTHORITY_MODEL"])
def test_unsupported_recognizer_is_rejected_before_model_loading(monkeypatch, setting):
    monkeypatch.setenv(setting, "other")
    with pytest.raises(ValueError, match=setting):
        DocULocalEngine()


class Worker:
    def __init__(self, error=None):
        self.error = error
        self.paths = []
        self.sides = []
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def describe(self):
        return {"pipeline_ready": self.initialized, "device": "cuda:0", "cuda_available": True}

    def extract_document(self, path, side):
        self.paths.append(Path(path))
        self.sides.append(side)
        assert path.is_file()
        if self.error:
            raise self.error
        return {"success": True, "doc_type": f"cccd_{side}", "extracted_fields": {"issue_date": "2021-03-15"}}


def submit(client, data=None, hint="cccd_back", headers=None):
    return client.post('/api/v1/extract', params={"doc_type_hint": hint}, headers=headers or {},
                       files={"file": ('../../private.png', image_data() if data is None else data, 'image/png')})


def test_worker_initializes_gpu_and_cleans_temporary_uploads():
    worker = Worker()
    with TestClient(create_gpu_app(worker)) as client:
        assert client.get('/api/v1/health/ready').json()["pipeline_ready"]
        response = submit(client)
        assert response.status_code == 200
        assert response.json()["extracted_fields"]["issue_date"] == "2021-03-15"
        assert worker.sides == ["back"]
        assert not worker.paths[0].parent.exists()


def test_worker_without_cuda_is_not_ready_but_remains_observable():
    class NotReady(Worker):
        def initialize(self):
            raise OCRUnavailable("CUDA unavailable")
    with TestClient(create_gpu_app(NotReady())) as client:
        assert client.get('/api/v1/health/ready').status_code == 503


def test_multiple_regions_have_a_machine_readable_reason_and_cleanup():
    worker = Worker(OCRMultipleDocuments(2))
    with TestClient(create_gpu_app(worker)) as client:
        response = submit(client)
    assert response.status_code == 422
    assert response.json()["error_code"] == "multiple_documents"
    assert response.json()["detected_regions"] == 2
    assert "nghi là thẻ" in response.json()["error"]
    assert not worker.paths[0].parent.exists()


@pytest.mark.parametrize("error,status", [(OCRBusy('busy'),429), (OCRUnavailable('unavailable'),503), (RuntimeError('private'),503)])
def test_worker_failure_always_cleans_images(error,status):
    worker = Worker(error)
    with TestClient(create_gpu_app(worker)) as client:
        response = submit(client)
        assert response.status_code == status
        assert "private" not in response.text
        assert not worker.paths[0].parent.exists()


def test_worker_auth_and_invalid_uploads_never_reach_models(monkeypatch):
    monkeypatch.setenv('GPU_OCR_API_KEY','test-secret')
    worker = Worker()
    with TestClient(create_gpu_app(worker)) as client:
        assert submit(client).status_code == 401
        assert submit(client, headers={'Authorization':'Bearer wrong'}).status_code == 401
        headers={'Authorization':'Bearer test-secret'}
        assert submit(client, b'not image', headers=headers).status_code == 422
        assert submit(client, hint='gplx', headers=headers).status_code == 422
        assert not worker.paths
        assert submit(client,headers=headers).status_code == 200
