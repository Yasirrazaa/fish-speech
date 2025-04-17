          │     │           (layers...

          │     └ ()

          └ <function generate_agent at 0x7e5acb88d300>

TypeError: generate_agent() missing 1 required keyword-only argument: 'prompt'

ERROR:    Exception in ASGI application

Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi

    result = await app(  # type: ignore[func-returns-value]

             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__

    return await self.app(scope, receive, send)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/kui/asgi/applications.py", line 163, in __call__

    await self.app(scope, receive, send)

  File "/usr/local/lib/python3.11/dist-packages/kui/asgi/applications.py", line 118, in app

    return await getattr(self, scope_type)(scope, receive, send)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/kui/asgi/applications.py", line 142, in http

    return await response(scope, receive, send)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/baize/asgi/responses.py", line 167, in __call__

    chunk = await generator.asend(None)

            ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/baize/asgi/responses.py", line 191, in render_stream

    async for chunk in self.iterable:

  File "/app/fish-speech/tools/api.py", line 719, in inference_async

    for chunk in inference(req):

  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 57, in generator_context

    response = gen.send(request)

               ^^^^^^^^^^^^^^^^^

  File "/app/fish-speech/tools/api.py", line 682, in inference

    if result.status == "error":

       ^^^^^^^^^^^^^

AttributeError: 'str' object has no attribute 'status'

_client.py          :1786 2025-04-17 16:22:41,164 HTTP Request: POST http://localhost:8080/v1/tts "HTTP/1.1 200 OK"

2025-04-17 16:22:41.164 | INFO     | __main__:process_direct_tts_request:315 - TTS request started successfully, beginning to stream response...

2025-04-17 16:22:41.165 | WARNING  | __main__:process_direct_tts_request:356 - Streaming error: peer closed connection without sending complete message body (incomplete chunked read)

2025-04-17 16:22:41.165 | ERROR    | __main__:process_direct_tts_request:359 - Failed to stream after 3 attempts: peer closed connection without sending complete message body (incomplete chunked read)

2025-04-17 16:22:41.165 | INFO     | __main__:process_direct_tts_request:361 - Falling back to non-streaming approach...

{"requestId": "65188188-39b0-4484-9cb6-6bc7381756fc-e1", "message": "Finished.", "level": "INFO"}