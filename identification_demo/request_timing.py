"""Measure HTTP receive and dispatch time outside the QR/OCR pipeline."""
from time import perf_counter


class RequestTimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/api/cccd/extract":
            return await self.app(scope, receive, send)
        timing = {"started": perf_counter()}
        scope.setdefault("state", {})["request_timing"] = timing

        async def timed_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                timing.setdefault("received", perf_counter())
            return message

        async def timed_send(message):
            if message["type"] == "http.response.start":
                elapsed = (perf_counter() - timing["started"]) * 1000
                metrics = [f'server;dur={elapsed:.2f}']
                for key, name in (("receive_upload", "receive"), ("prepare_request", "prepare"),
                                  ("dispatch_wait", "dispatch"), ("pipeline", "pipeline")):
                    if key in timing.get("stages", {}):
                        metrics.append(f'{name};dur={timing["stages"][key]:.2f}')
                message = {**message, "headers": [*message.get("headers", []),
                                                  (b"server-timing", ", ".join(metrics).encode("ascii"))]}
            await send(message)

        await self.app(scope, timed_receive, timed_send)


def request_timings(timing, dispatched, worker_started, worker_finished):
    finished = perf_counter()
    received = timing.get("received", dispatched)
    started = timing.get("started", received)
    values = {
        "receive_upload": received - started,
        "prepare_request": dispatched - received,
        "dispatch_wait": worker_started - dispatched,
        "pipeline": worker_finished - worker_started,
        "finish_result": finished - worker_finished,
        "server_until_result": finished - started,
    }
    stages = {key: round(max(0, value) * 1000, 2) for key, value in values.items()}
    timing["stages"] = stages
    return stages
