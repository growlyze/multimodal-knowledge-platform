"""Offline tests for litellm-hooks/gemini_malformed_retry.py (P169).

The shapes below are what LiteLLM v1.100.0 actually hands to proxy hooks when
Google ends a turn with finishReason MALFORMED_FUNCTION_CALL — captured by
running the pinned litellm against a fake Gemini endpoint replaying the raw
chunk logged on the VPS (/root/litellm-debug-20261021-1734.txt line 6260):

  stream:     {delta: {role: assistant, provider_specific_fields:
                        {thought_signatures: [...]}}}
              {finish_reason: "stop", delta: {}}
  non-stream: choices[0] = {finish_reason: "stop",
                            message: {content: None, tool_calls: None}}

The MALFORMED_FUNCTION_CALL marker itself is NOT visible at this layer (the
streaming path maps it to "stop" before building the chunk, and the
non-stream Choices keeps no native_finish_reason for it), which is why the
hook keys on "Gemini returned nothing and said stop", not on the marker.

Pure tests use SimpleNamespace stand-ins and run without litellm installed;
test_proxy_end_to_end drives a real `litellm` proxy subprocess against a fake
Gemini server and is skipped when litellm is not importable.
Run: python -m pytest litellm-hooks/tests -q
"""
import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

HOOK_DIR = Path(__file__).resolve().parents[1]
HOOK_FILE = HOOK_DIR / "gemini_malformed_retry.py"


def _load():
    spec = importlib.util.spec_from_file_location("gemini_malformed_retry", HOOK_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# ---------------------------------------------------------------- fakes ---
def chunk(content=None, tool_calls=None, finish=None, reasoning=None, role=None):
    delta = NS(content=content, tool_calls=tool_calls, function_call=None,
               reasoning_content=reasoning, role=role)
    return NS(choices=[NS(delta=delta, finish_reason=finish, index=0)])


def malformed_stream():
    # Exactly what LiteLLM yields for the MALFORMED_FUNCTION_CALL turn.
    return [chunk(role="assistant"), chunk(finish="stop")]


def text_stream(text="The answer"):
    return [chunk(role="assistant"), chunk(content=text), chunk(content="."),
            chunk(finish="stop")]


def tool_stream():
    tc = [NS(index=0, id="call_1", type="function",
             function=NS(name="run_analysis", arguments='{"x": 1}'))]
    return [chunk(role="assistant"), chunk(tool_calls=tc), chunk(finish="tool_calls")]


async def agen(items, log=None, tag=""):
    for i, it in enumerate(items):
        if log is not None:
            log.append(f"{tag}produce{i}")
        yield it


class Recorder(mod.GeminiMalformedRetry):
    """Hook with the network re-call replaced by a queue of canned replies."""

    def __init__(self, replies, stream=True):
        super().__init__()
        self.replies = list(replies)
        self.calls = []
        self.stream = stream

    async def _recall(self, request_data):
        self.calls.append(dict(request_data))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return agen(reply) if self.stream else reply


TOOLS = [{"type": "function", "function": {
    "name": "run_command", "parameters": {"type": "object",
                                          "properties": {"command": {"type": "string"}}}}}]
MESSAGES = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Which trade hit both the stop and the target?"}]


def base_request(model="gemini-3.8-flash", **extra):
    data = {"model": model, "stream": True, "messages": [dict(m) for m in MESSAGES],
            "tools": TOOLS, "litellm_logging_obj": object(), "litellm_call_id": "c0"}
    data.update(extra)
    return data


def run_stream(hook, first, model="gemini-3.8-flash", request_data=None):
    request_data = request_data if request_data is not None else base_request(model)

    async def go():
        out = []
        async for c in hook.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=agen(first),
                request_data=request_data):
            out.append(c)
        return out
    return asyncio.run(go())


def text_of(chunks):
    return "".join(c.choices[0].delta.content or "" for c in chunks)


def finishes(chunks):
    return [c.choices[0].finish_reason for c in chunks if c.choices[0].finish_reason]


# ------------------------------------------------------------ streaming ---
def test_empty_stop_stream_is_retried_and_answer_delivered(caplog):
    hook = Recorder([text_stream("Net profit -$337.50")])
    with caplog.at_level("WARNING"):
        out = run_stream(hook, malformed_stream())
    assert len(hook.calls) == 1
    assert text_of(out) == "Net profit -$337.50."
    # The failed attempt's "stop" must not reach Open WebUI — it would end the
    # message before the retried text arrives.
    assert finishes(out) == ["stop"]
    assert "P169" in caplog.text and "retry 1/2" in caplog.text


def test_retry_call_gets_fresh_logging_identity():
    hook = Recorder([text_stream()])
    run_stream(hook, malformed_stream())
    sent = hook.calls[0]
    # Reusing the first attempt's logging object would merge two provider
    # calls into one spend/trace record.
    assert "litellm_logging_obj" not in sent
    assert sent["litellm_call_id"] != "c0"
    assert sent["model"] == "gemini-3.8-flash" and sent["stream"] is True


def test_two_empties_then_answer_uses_both_retries():
    hook = Recorder([malformed_stream(), text_stream("ok")])
    out = run_stream(hook, malformed_stream())
    assert len(hook.calls) == 2
    assert text_of(out) == "ok."
    assert finishes(out) == ["stop"]


def test_gives_up_after_two_retries_and_returns_the_empty_turn(caplog):
    hook = Recorder([malformed_stream(), malformed_stream(), text_stream()])
    with caplog.at_level("WARNING"):
        out = run_stream(hook, malformed_stream())
    assert len(hook.calls) == 2                     # max 2 retries, never a 3rd
    assert text_of(out) == ""                       # rag_filter's notice takes over
    assert finishes(out) == ["stop"]                # stream still terminates normally
    assert "giving up" in caplog.text


def test_text_stream_passes_through_unchanged_without_retry():
    first = text_stream("hello")
    hook = Recorder([])
    out = run_stream(hook, first)
    assert hook.calls == []
    assert out == first


def test_tool_call_stream_passes_through_without_retry():
    first = tool_stream()
    hook = Recorder([])
    out = run_stream(hook, first)
    assert hook.calls == []
    assert out == first


def test_content_is_streamed_live_not_buffered():
    # The hook must not hold the whole answer back until the stream ends —
    # only the finish chunk is held. Interleaving proves chunk 1 was yielded
    # before the upstream produced chunk 2.
    log = []
    hook = Recorder([])

    async def go():
        async for c in hook.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=agen(text_stream("a"), log),
                request_data={"model": "gemini-3.8-flash"}):
            log.append("yield:" + (c.choices[0].delta.content or "-"))
    asyncio.run(go())
    assert log.index("yield:a") < log.index("produce2")


def test_non_gemini_model_never_retried():
    hook = Recorder([text_stream()])
    out = run_stream(hook, malformed_stream(), model="claude-sonnet-5")
    assert hook.calls == []
    assert out == malformed_stream()


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls"])
def test_empty_with_other_finish_reason_not_retried(reason):
    # SAFETY -> content_filter, MAX_TOKENS -> length: retrying would just burn
    # money on the same refusal/limit.
    hook = Recorder([text_stream()])
    out = run_stream(hook, [chunk(role="assistant"), chunk(finish=reason)])
    assert hook.calls == []
    assert finishes(out) == [reason]


def test_retry_failure_degrades_to_original_empty_turn(caplog):
    hook = Recorder([RuntimeError("429 quota")])
    with caplog.at_level("WARNING"):
        out = run_stream(hook, malformed_stream())
    assert len(hook.calls) == 1
    assert finishes(out) == ["stop"]
    assert "retry failed" in caplog.text


def test_reasoning_only_then_empty_is_retried():
    hook = Recorder([text_stream("x")])
    out = run_stream(hook, [chunk(role="assistant"), chunk(reasoning="thinking"),
                            chunk(finish="stop")])
    assert len(hook.calls) == 1
    assert text_of(out) == "x."


# ------------------------------------------------- retry variants (v2) ---
# Live 2026-10-25 (chat 7fb11f85, turns 3 and 5): identical re-sends came back
# blank 2/2 twice, so the retries must CHANGE the request, not repeat it.
def _nudge_of(data):
    last = data["messages"][-1]
    return last if last["role"] == "system" else None


def test_retry1_appends_nudge_system_message_only():
    hook = Recorder([text_stream()])
    original = base_request()
    run_stream(hook, malformed_stream(), request_data=original)
    sent = hook.calls[0]
    nudge = _nudge_of(sent)
    assert nudge and nudge["content"] == mod.NUDGE_TEXT
    assert sent["messages"][:-1] == MESSAGES               # history untouched
    assert len(original["messages"]) == len(MESSAGES)      # caller's dict not mutated
    assert "temperature" not in sent and "extra_body" not in sent


def test_nudge_names_both_allowed_shapes_and_is_short():
    t = mod.NUDGE_TEXT.lower()
    assert "plain text" in t and "function call" in t and "schema" in t
    assert len(mod.NUDGE_TEXT) < 400


def test_retry2_adds_sampling_change_and_validated_mode(caplog):
    hook = Recorder([malformed_stream(), text_stream("ok")])
    with caplog.at_level("WARNING"):
        out = run_stream(hook, malformed_stream())
    assert text_of(out) == "ok."
    second = hook.calls[1]
    assert _nudge_of(second)["content"] == mod.NUDGE_TEXT
    assert sum(1 for m in second["messages"] if m["role"] == "system") == 2  # nudge once
    assert second["temperature"] == 0.7                    # unset -> 0.7
    assert second["extra_body"]["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"
    assert "variant=nudge)" in caplog.text
    assert "variant=nudge+temperature=0.7+validated" in caplog.text


@pytest.mark.parametrize("given,expected", [(1.0, 0.8), (0.2, 0.4), (0.0, 0.2)])
def test_retry2_moves_a_set_temperature_slightly(given, expected):
    hook = Recorder([malformed_stream(), text_stream()])
    run_stream(hook, malformed_stream(), request_data=base_request(temperature=given))
    assert hook.calls[0]["temperature"] == given           # retry 1 keeps it
    assert hook.calls[1]["temperature"] == pytest.approx(expected)


def test_retry2_without_tools_gets_no_validated_mode():
    hook = Recorder([malformed_stream(), text_stream()])
    data = base_request()
    data.pop("tools")
    run_stream(hook, malformed_stream(), request_data=data)
    assert "extra_body" not in hook.calls[1]


@pytest.mark.parametrize("choice", ["none", "required",
                                    {"type": "function", "function": {"name": "run_command"}}])
def test_retry2_respects_an_explicit_tool_choice(choice):
    hook = Recorder([malformed_stream(), text_stream()])
    run_stream(hook, malformed_stream(), request_data=base_request(tool_choice=choice))
    assert "extra_body" not in hook.calls[1]
    assert hook.calls[1]["tool_choice"] == choice


def test_retry2_tool_choice_auto_is_replaced_by_validated():
    # "auto" and VALIDATED mean the same thing apart from the validation; two
    # toolConfig sources would otherwise be merged in an unspecified order.
    hook = Recorder([malformed_stream(), text_stream()])
    run_stream(hook, malformed_stream(), request_data=base_request(tool_choice="auto"))
    assert "tool_choice" not in hook.calls[1]
    assert hook.calls[1]["extra_body"]["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"


def test_retry2_existing_extra_body_is_merged_not_replaced():
    hook = Recorder([malformed_stream(), text_stream()])
    run_stream(hook, malformed_stream(), request_data=base_request(extra_body={"keep": 1}))
    eb = hook.calls[1]["extra_body"]
    assert eb["keep"] == 1 and eb["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"


class Rejected(Exception):
    status_code = 400


def test_validated_rejected_falls_back_to_same_retry_without_it(caplog):
    # VALIDATED is in Google's v1beta discovery doc but was never exercised on
    # gemini-3.8-flash; if the API 400s it, the last retry must still run.
    hook = Recorder([malformed_stream(), Rejected("mode VALIDATED not supported"),
                     text_stream("rescued")])
    with caplog.at_level("WARNING"):
        out = run_stream(hook, malformed_stream())
    assert text_of(out) == "rescued."
    assert len(hook.calls) == 3
    third = hook.calls[2]
    assert "extra_body" not in third and third["temperature"] == 0.7
    assert _nudge_of(third)["content"] == mod.NUDGE_TEXT
    assert "rejected" in caplog.text and "without VALIDATED" in caplog.text


def test_validated_fallback_not_used_for_non_400_errors():
    hook = Recorder([malformed_stream(), RuntimeError("timeout")])
    out = run_stream(hook, malformed_stream())
    assert len(hook.calls) == 2 and finishes(out) == ["stop"]


# ---------------------------------------------- opt-in source-side mode ---
def run_pre(hook, data):
    return asyncio.run(hook.async_pre_call_hook(user_api_key_dict=None, cache=None,
                                                data=data, call_type="acompletion"))


def _has_validated(data):
    return data is not None and "VALIDATED" in json.dumps((data.get("extra_body") or {}))


def test_pre_call_validated_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("GEMINI_FC_MODE_VALIDATED", raising=False)
    hook = mod.GeminiMalformedRetry()
    assert not _has_validated(run_pre(hook, base_request()))


def test_pre_call_validated_mode_when_enabled(monkeypatch):
    monkeypatch.setenv("GEMINI_FC_MODE_VALIDATED", "1")
    hook = mod.GeminiMalformedRetry()
    assert _has_validated(run_pre(hook, base_request()))
    # never for other providers, tool-less calls or an explicit tool_choice
    assert not _has_validated(run_pre(hook, base_request(model="claude-sonnet-5")))
    d = base_request()
    d.pop("tools")
    assert not _has_validated(run_pre(hook, d))
    assert not _has_validated(run_pre(hook, base_request(tool_choice="none")))


# ----------------------------------------------------------- non-stream ---
def resp(content=None, tool_calls=None, finish="stop"):
    return NS(choices=[NS(finish_reason=finish,
                          message=NS(content=content, tool_calls=tool_calls,
                                     function_call=None))])


def run_success(hook, response, model="gemini-3.8-flash"):
    return asyncio.run(hook.async_post_call_success_hook(
        data={"model": model, "litellm_call_id": "c0",
              "messages": [dict(m) for m in MESSAGES]},
        user_api_key_dict=None, response=response))


def test_non_stream_empty_is_replaced_by_retry():
    good = resp(content="answer")
    hook = Recorder([good], stream=False)
    assert run_success(hook, resp()) is good
    assert len(hook.calls) == 1


def test_non_stream_retries_use_the_same_variants():
    hook = Recorder([resp(), resp(content="ok")], stream=False)
    run_success(hook, resp())
    assert _nudge_of(hook.calls[0])["content"] == mod.NUDGE_TEXT
    assert "temperature" not in hook.calls[0]
    assert hook.calls[1]["temperature"] == 0.7


def test_non_stream_gives_up_after_two():
    hook = Recorder([resp(), resp(), resp(content="never")], stream=False)
    assert run_success(hook, resp()) is None          # None = keep original
    assert len(hook.calls) == 2


@pytest.mark.parametrize("r", [resp(content="hi"),
                               resp(tool_calls=[NS(id="c")], finish="tool_calls"),
                               resp(finish="length")])
def test_non_stream_normal_responses_untouched(r):
    hook = Recorder([], stream=False)
    assert run_success(hook, r) is None
    assert hook.calls == []


def test_non_stream_non_gemini_untouched():
    hook = Recorder([resp(content="x")], stream=False)
    assert run_success(hook, resp(), model="claude-opus-5-5") is None
    assert hook.calls == []


# ------------------------------------------------- real proxy, end to end ---
MAL_RAW = {"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "abc"}],
                                       "role": "model"},
                           "finishReason": "MALFORMED_FUNCTION_CALL", "index": 0,
                           "finishMessage": "Malformed function call: Failed to parse function "
                                            "call: Function call is empty - no input to parse."}],
           "usageMetadata": {"promptTokenCount": 18058, "totalTokenCount": 18058},
           "modelVersion": "gemini-3.8-flash", "responseId": "r1"}


def raw_text(text):
    return {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3,
                              "totalTokenCount": 13},
            "modelVersion": "gemini-3.8-flash", "responseId": "r2"}


def raw_tool():
    return {"candidates": [{"content": {"parts": [{"functionCall": {
        "name": "run_analysis", "args": {"x": 1}}}], "role": "model"},
        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3,
                          "totalTokenCount": 13},
        "modelVersion": "gemini-3.8-flash", "responseId": "r3"}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def proxy(tmp_path_factory):
    pytest.importorskip("litellm")
    import httpx  # noqa: F401  (litellm dependency)

    script = []          # queue of raw Gemini replies, consumed per request
    hits = []
    bodies = []          # request bodies exactly as Google would receive them

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            hits.append(self.path)
            body = script.pop(0) if script else raw_text("default")
            if body == "REJECT_VALIDATED":
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {
                    "code": 400, "status": "INVALID_ARGUMENT",
                    "message": "function calling mode VALIDATED not supported"}}).encode())
                return
            self.send_response(200)
            if "stream" in self.path:
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(("data: " + json.dumps(body) + "\r\n\r\n").encode())
            else:
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    # Same layout as the container: hook file next to config.yaml, loaded by
    # LiteLLM's get_instance_fn relative to the config's directory.
    d = tmp_path_factory.mktemp("proxy")
    (d / "gemini_malformed_retry.py").write_text(HOOK_FILE.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
    (d / "config.yaml").write_text(f"""
model_list:
  - model_name: gemini-3.8-flash
    litellm_params:
      model: gemini/gemini-3.8-flash
      api_key: fake
      api_base: http://127.0.0.1:{fake.server_port}
litellm_settings:
  callbacks: gemini_malformed_retry.proxy_handler_instance
""", encoding="utf-8")
    port = _free_port()
    exe = Path(sys.executable).with_name("litellm" + (".exe" if os.name == "nt" else ""))
    logf = open(d / "proxy.log", "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.Popen([str(exe), "--config", str(d / "config.yaml"), "--port", str(port)],
                         stdout=logf, stderr=subprocess.STDOUT, env=env)
    import httpx
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(base + "/health/liveliness", timeout=1).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        p.kill()
        pytest.fail("proxy did not start: " + (d / "proxy.log").read_text(encoding="utf-8")[-3000:])
    yield NS(base=base, script=script, hits=hits, bodies=bodies, log=d / "proxy.log")
    p.kill()
    fake.shutdown()


def _post(proxy, stream, tools=True):
    import httpx
    body = {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hi"}],
            "stream": stream}
    if tools:
        body["tools"] = [{"type": "function", "function": {
            "name": "run_analysis", "parameters": {"type": "object",
                                                   "properties": {"x": {"type": "integer"}}}}}]
    r = httpx.post(proxy.base + "/v1/chat/completions", json=body,
                   headers={"Authorization": "Bearer sk-litellm"}, timeout=30)
    assert r.status_code == 200, r.text
    if not stream:
        return r.json()
    chunks = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    return chunks


def _stream_text(chunks):
    return "".join((c["choices"][0]["delta"].get("content") or "")
                   for c in chunks if c.get("choices"))


def test_proxy_end_to_end(proxy):
    # 1. Streamed malformed turn -> retried -> client sees the answer, one stop.
    proxy.hits.clear()
    proxy.script[:] = [MAL_RAW, raw_text("Net profit -$337.50")]
    chunks = _post(proxy, stream=True)
    assert _stream_text(chunks) == "Net profit -$337.50"
    assert [c["choices"][0].get("finish_reason") for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason")] == ["stop"]
    assert len(proxy.hits) == 2

    # 2. Three malformed in a row -> exactly 1 + 2 upstream calls, empty reply.
    #    What Google receives on each attempt is the point of the v2 strategy.
    proxy.hits.clear()
    proxy.bodies.clear()
    proxy.script[:] = [MAL_RAW, MAL_RAW, MAL_RAW, raw_text("never")]
    chunks = _post(proxy, stream=True)
    assert _stream_text(chunks) == ""
    assert len(proxy.hits) == 3
    b0, b1, b2 = proxy.bodies

    def sys_text(b):
        parts = (b.get("system_instruction") or {}).get("parts", [])
        return " ".join(p.get("text", "") for p in parts)
    assert mod.NUDGE_TEXT not in sys_text(b0)
    assert mod.NUDGE_TEXT in sys_text(b1) and mod.NUDGE_TEXT in sys_text(b2)
    assert b1["generationConfig"].get("temperature") == b0["generationConfig"].get("temperature")
    assert b2["generationConfig"]["temperature"] == 0.7
    assert b2["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"
    assert "VALIDATED" not in json.dumps(b0.get("toolConfig", {}))
    assert "VALIDATED" not in json.dumps(b1.get("toolConfig", {}))

    # 2b. Google rejects VALIDATED -> the last retry re-runs without it.
    proxy.hits.clear()
    proxy.bodies.clear()
    proxy.script[:] = [MAL_RAW, MAL_RAW, "REJECT_VALIDATED", raw_text("rescued")]
    assert _stream_text(_post(proxy, stream=True)) == "rescued"
    assert len(proxy.hits) == 4
    assert "VALIDATED" not in json.dumps(proxy.bodies[3].get("toolConfig", {}))
    assert proxy.bodies[3]["generationConfig"]["temperature"] == 0.7

    # 3. Normal streamed tool call -> one upstream call, tool call delivered.
    proxy.hits.clear()
    proxy.script[:] = [raw_tool()]
    chunks = _post(proxy, stream=True)
    calls = [tc for c in chunks if c.get("choices")
             for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
    assert calls and calls[0]["function"]["name"] == "run_analysis"
    assert len(proxy.hits) == 1

    # 4. Normal streamed text -> one call.
    proxy.hits.clear()
    proxy.script[:] = [raw_text("plain")]
    assert _stream_text(_post(proxy, stream=True)) == "plain"
    assert len(proxy.hits) == 1

    # 5. Non-stream malformed (Open WebUI's title/follow-up tasks) -> retried.
    proxy.hits.clear()
    proxy.script[:] = [MAL_RAW, raw_text("Title")]
    j = _post(proxy, stream=False, tools=False)
    assert j["choices"][0]["message"]["content"] == "Title"
    assert len(proxy.hits) == 2

    log = proxy.log.read_text(encoding="utf-8")
    assert log.count("P169 empty Gemini reply") >= 4      # 1 + 2 + 1 retries logged
    assert "giving up" in log
