import re

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from identification_demo.web import create_app
from test_pipeline import Engine, Reader, image_data


def test_http_timings_include_slow_upload_and_dispatch_wait(monkeypatch):
    import identification_demo.web as web
    original = web.run_in_threadpool

    async def delayed_dispatch(function, *args):
        await anyio.sleep(.035)
        return await original(function, *args)

    monkeypatch.setattr(web, "run_in_threadpool", delayed_dispatch)
    app = create_app(Engine(), Reader())
    body = httpx.Request("POST", "http://test/api/cccd/extract", files={
        "front_image": ("front.png", image_data(), "image/png"),
        "back_image": ("back.png", image_data("back"), "image/png"),
    })
    content = body.read()

    async def request():
        async def slow_upload():
            yield content[:len(content) // 2]
            await anyio.sleep(.045)
            yield content[len(content) // 2:]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            return await client.post("/api/cccd/extract", content=slow_upload(),
                                     headers={"Content-Type": body.headers["Content-Type"]})

    response = anyio.run(request)
    assert response.status_code == 200
    result = response.json()
    stages = result["request_timings_ms"]
    assert result["status"] == "complete"
    assert stages["receive_upload"] >= 40
    assert stages["dispatch_wait"] >= 30
    assert stages["pipeline"] == pytest.approx(result["timings_ms"]["total"], abs=2)
    assert stages["server_until_result"] >= result["processing_time_ms"] + 70
    assert sum(value for key, value in stages.items() if key != "server_until_result") == pytest.approx(stages["server_until_result"], abs=.05)
    header = response.headers["server-timing"]
    assert set(re.findall(r"(?:^|, )([a-z]+);dur=", header)) == {"server", "receive", "prepare", "dispatch", "pipeline"}
    assert float(re.search(r"server;dur=([\d.]+)", header).group(1)) >= stages["server_until_result"] - .1
    assert response.headers["cache-control"] == "no-store"


def test_validation_error_still_has_http_timing_without_running_ocr():
    engine = Engine()
    response = TestClient(create_app(engine, Reader())).post("/api/cccd/extract")
    assert response.status_code == 422
    assert "server;dur=" in response.headers["server-timing"]
    assert not engine.calls


def test_unrelated_routes_keep_their_existing_contract():
    client = TestClient(create_app(Engine(), Reader()))
    response = client.get("/")
    assert response.status_code == 200
    assert "server-timing" not in response.headers
