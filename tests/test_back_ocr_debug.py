import httpx
from fastapi.testclient import TestClient

from identification_demo.gpu_engine import DocUGPUEngine
from identification_demo.web import create_app
from identification_demo.ocr_debug import authority_crop_preview
from test_gpu_engine import FRONT, READY
from test_pipeline import Engine, Reader, image_data


def submit(client, include=None):
    data = {"quality": "high"}
    if include is not None:
        data["include_back_ocr"] = include
    return client.post('/api/cccd/extract', data=data, files={
        'front_image': ('front.png', image_data(), 'image/png'),
        'back_image': ('back.png', image_data('back'), 'image/png'),
    })


def test_requested_back_text_reaches_ui_without_extra_ocr_or_changing_qr_priority():
    posts = []
    initial = ['Ngày cấp: 20/04/2021', 'CỤC TRƯỞNG CỤC CẢNH SÁT', 'NGUYỄN VĂN MẪU']
    final = initial[:2] + ['QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI'] + initial[2:]

    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json=READY)
        posts.append(request.url.params['doc_type_hint'])
        return httpx.Response(200, json=FRONT if posts[-1] == 'cccd_front' else {
            'success': True, 'doc_type': 'cccd_back', 'extracted_fields': {},
            'initial_text_lines': initial, 'text_lines': final,
        })

    cpu = Engine()
    gpu = DocUGPUEngine('http://gpu.test', transport=httpx.MockTransport(handler))
    response = submit(TestClient(create_app(cpu, Reader(), gpu)), 'true')
    assert response.status_code == 200
    result = response.json()
    debug = result['ocr']['back']['debug']
    assert [line['text'] for line in debug['initial_lines']] == initial
    assert [line['text'] for line in debug['final_lines']] == final
    assert [line['number'] for line in debug['final_lines'] if line['in_authority']] == [2, 3]
    assert debug['authority_candidate'] == result['fields']['issuing_authority']
    assert not debug['ambiguous']
    assert result['fields']['issue_date'] == '15/03/2021' and result['field_sources']['issue_date'] == 'qr_front'
    assert posts == ['cccd_front', 'cccd_back'] and not cpu.calls
    assert len(result['fields']) == 12


def test_common_api_keeps_back_text_out_unless_requested():
    for include in (None, 'false'):
        cpu, gpu = Engine(), Engine()
        response = submit(TestClient(create_app(cpu, Reader(), gpu)), include)
        result = response.json()
        assert response.status_code == 200
        assert 'debug' not in result['ocr']['back']
        assert 'raw_text' not in response.text and 'NGUYỄN VĂN MẪU' not in response.text


def test_debug_remains_available_when_only_partial_institution_was_read():
    gpu = Engine(back={'text_lines': ['Ngày cấp: 20/04/2021', 'CỤC TRƯỞNG CỤC CẢNH SÁT',
                                     '<script>window.injected = true</script>']})
    result = submit(TestClient(create_app(Engine(), Reader(), gpu)), 'true').json()
    assert result['fields']['issuing_authority'] == 'CỤC CẢNH SÁT'
    debug = result['ocr']['back']['debug']
    assert debug['authority_candidate'] == 'CỤC CẢNH SÁT'
    assert debug['final_lines'][1]['in_authority']
    assert not debug['final_lines'][2]['in_authority']
    assert debug['final_lines'][2]['text'].startswith('<script>')


def test_crop_preview_shows_the_exact_searched_pixels_and_only_on_debug_request():
    import base64
    import io
    import numpy as np
    from PIL import Image
    from identification_demo.gpu_engine import normalize_result

    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    image[50:150, 100:300] = [0, 0, 255]
    crop = authority_crop_preview(image, [100, 50, 300, 150])
    with Image.open(io.BytesIO(base64.b64decode(crop['image'].split(',', 1)[1]))) as preview:
        assert preview.size == (200, 100)
        assert preview.getpixel((100, 50))[0] > 240
        assert preview.getpixel((100, 50))[2] < 10
    payload = {'success': True, 'doc_type': 'cccd_back', 'extracted_fields': {},
               'text_lines': ['CỤC TRƯỞNG CỤC CẢNH SÁT'], 'authority_search_crop': crop}
    normalized = normalize_result(payload)
    assert normalized['back_ocr_debug']['search_crop'] == crop

    class GPU(Engine):
        def extract(self, path):
            super().extract(path)
            return normalized if self.calls[-1] == 'back' else super_front

    super_front = Engine().results['front']
    for include in (None, 'true'):
        result = submit(TestClient(create_app(Engine(), Reader(), GPU())), include).json()
        assert ('search_crop' in result['ocr']['back'].get('debug', {})) == (include == 'true')
