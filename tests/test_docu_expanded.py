from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identification_demo.docu_region_engine import DocUExpandedGPUEngine
from identification_demo.docu_region_worker import DocURegionLocalEngine,ExpandedView
from identification_demo.gpu_worker import create_gpu_app
from identification_demo.expiry_context import STRATEGY
from identification_demo.web import create_app
from test_gpu_engine import READY
from test_ocr_experiments import full_engine,submit
from test_pipeline import Reader,image_data


def test_v2_and_v3_share_the_fifo_without_leaking_request_mode():
    engine=DocURegionLocalEngine()
    entered,release=Event(),Event()
    modes=[]
    def read(image,side,needed):
        modes.append(engine._expanded_active)
        if engine._expanded_active:
            entered.set()
            assert release.wait(5)
        return {'text_lines':[]}
    engine._extract_pixels=read
    try:
        with Image.new('RGB',(10,10)) as image,ThreadPoolExecutor(2) as pool:
            v3=pool.submit(engine.extract_expanded_image,image,'front')
            assert entered.wait(5)
            v2=pool.submit(engine.extract_image,image,'front')
            release.set()
            v3.result(5);v2.result(5)
        assert modes==[True,False] and not engine._expanded_active
    finally:
        engine.close()


def test_v3_mode_is_cleared_after_inference_failure():
    engine=DocURegionLocalEngine()
    def fail(*args): raise RuntimeError('test failure')
    engine._extract_pixels=fail
    try:
        with pytest.raises(RuntimeError):engine.extract_expanded_image(None,'front')
        assert not engine._expanded_active
    finally:engine.close()


def test_mounted_v3_route_uses_shared_engine_and_enforces_auth(monkeypatch):
    monkeypatch.setenv('GPU_OCR_API_KEY','test-token')
    class Worker:
        _queue=None
        initialized=False
        calls=[]
        def initialize(self):self.initialized=True
        def close(self):pass
        def describe(self):return {'pipeline_ready':self.initialized,'strategy':'docu_regions_v2'}
        def extract_expanded_image(self,image,side,needed_fields):
            self.calls.append(side)
            return {'success':True}
    worker=Worker()
    app=create_gpu_app(worker)
    app.mount('/expanded',create_gpu_app(ExpandedView(worker),extra_front_fields=('expiry_date',)))
    with TestClient(app) as client:
        assert client.get('/expanded/api/v1/health/ready').json()['strategy']==STRATEGY
        files={'file':('card.png',image_data(),'image/png')}
        assert client.post('/expanded/api/v1/extract',files=files).status_code==401
        response=client.post('/expanded/api/v1/extract',files=files,
            params={'fields':'expiry_date'},headers={'Authorization':'Bearer test-token'})
        assert response.status_code==200 and worker.calls==['front']


def test_v3_client_uses_mount_and_debug_is_returned_only_when_requested():
    seen=[]
    def handler(request):
        seen.append(request.url.path)
        if request.method=='GET':return httpx.Response(200,json={**READY,'strategy':STRATEGY})
        return httpx.Response(200,json={'success':True,'doc_type':request.url.params['doc_type_hint'],
            'extracted_fields':{},'text_lines':['Có giá trị đến: 25/10/2036'],
            'field_recovery':{'strategy':STRATEGY,'target':'expiry_date','status':'recovered','reread_regions':2},
            'region_debug':[{'processing':'v2_expanded_box_v1','search_crop':{'image':'data:image/jpeg;base64,ZmFrZQ=='}}]})
    engine=DocUExpandedGPUEngine('http://gpu.test',httpx.MockTransport(handler))
    client=TestClient(create_app(full_engine(),Reader(),docu_expanded_engine=engine))
    result=submit(client,'docu_gpu_v3',include_ocr_text='false').json()
    assert result['ocr_engine']['strategy']==STRATEGY
    assert result['ocr']['front']['field_recovery']['reread_regions']==2
    assert 'data:image/' not in str(result) and 'debug' not in result['ocr']['front']
    result=submit(client,'docu_gpu_v3',include_ocr_text='true').json()
    assert result['ocr']['front']['debug']['region_reads'][0]['search_crop']['image'].startswith('data:image/')
    assert all(path.startswith('/expanded/api/v1/') for path in seen)
