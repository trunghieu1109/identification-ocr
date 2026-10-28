from fastapi.testclient import TestClient

from identification_demo.web import create_app
from test_pipeline import Engine, Reader


def test_only_supported_apis_are_exposed_and_health_does_not_probe_retired_workers():
    class GPU(Engine):
        def describe(self):
            return {"provider": "namlhn/vietnam-cccd-ocr", "available": True}
    with TestClient(create_app(Engine(), Reader(), GPU())) as client:
        paths = set(client.get('/openapi.json').json()['paths'])
        assert paths == {'/api/health', '/api/cccd/extract', '/api/qr/cccd', '/api/ocr/cccd',
                         '/api/cccd/experiment/health', '/api/cccd/experiment/extract'}
        health = client.get('/api/health').json()
        assert 'qr_experiments' not in health
        assert set(health['ocr_profiles']) == {'medium', 'high'}
        for path in ('/api/cccd/expiry/experiment',
                     '/api/qr/experiment', '/api/qr/experiment/pair'):
            assert client.post(path).status_code == 404
        assert client.get('/assets/qr-lab.js').status_code == 404
        assert client.get('/assets/ocr-lab.js').status_code == 404
