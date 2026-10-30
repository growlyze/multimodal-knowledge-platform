"""LiteLLM proxy callback: retry a Gemini turn that came back empty (P169).

Why this exists
---------------
Gemini (gemini-3.8-flash; the gemini-3.7-flash picker entry is served by the
same model since Google retired 3.7 on 2026-10-08) sometimes ends a turn with
finishReason MALFORMED_FUNCTION_CALL ("Function call is empty - no input to
parse") and zero output tokens. LiteLLM v1.100.0 maps that to an ordinary
finish_reason "stop", Open WebUI saves a blank assistant message, and the
user only sees rag_filter's EMPTY_REPLY_NOTICE ("...Please send your message
again."). Resending works ~5 times in 6, so the fix is to do that resend here,
before anything reaches Open WebUI.

What it detects — and why not the MALFORMED marker itself
---------------------------------------------------------
At the hook layer the marker is gone (verified against the pinned 1.100.0 by
replaying the raw chunk from the VPS debug log through a fake Gemini server):
the stream is just {role: assistant} then {finish_reason: "stop", delta: {}},
and the non-stream Choices carries no native_finish_reason for it. So a retry
fires when a *Gemini* turn produced no text, no tool call and finished with
"stop" (or no reason). An intentionally empty answer is not a thing users of
this stack ever want, so the broader rule costs nothing real. Empty turns that
finish with "length" or "content_filter" (MAX_TOKENS / SAFETY) are NOT retried
— the same request would hit the same limit or refusal and just burn money.

How (streaming is the path that matters — Open WebUI streams chat turns)
--------------------------------------------------------------------------
async_post_call_streaming_iterator_hook wraps the stream. Every chunk is
passed through live EXCEPT the one carrying finish_reason (and anything after
it, e.g. the usage chunk): holding back only the ending means answers still
stream token by token, while a failed attempt's "stop" never reaches Open
WebUI (it would end the message before the retried text arrived). If the
stream ends without text/tool call, the request is re-sent through the proxy's
own router (same model group, messages and tools) up to MAX_RETRIES times,
each logged as "P169 empty Gemini reply ... retry n/2 (variant=...)".

Retries are NOT identical re-sends (v2, 2026-10-26): live, identical re-sends
came back blank 2/2 on two turns of chat 7fb11f85, so each retry changes the
request:
  retry 1  "nudge": + NUDGE_TEXT as a trailing system message;
  retry 2  "nudge+temperature=<t>+validated": the nudge, a sampling change and
           Gemini's function-calling mode VALIDATED (constrained decoding of
           calls; only when tools are declared and tool_choice is unset/auto).
           If Google rejects VALIDATED with a 400 the same retry is re-sent
           without it ("VALIDATED mode rejected ... re-running") so the last
           chance is never lost to an untested setting.
Opt-in source fix: GEMINI_FC_MODE_VALIDATED=1 in the container env makes
EVERY Gemini request with tools use VALIDATED (async_pre_call_hook). Off by
default until one live probe shows gemini-3.8-flash accepts it.

If every attempt is empty, or a retry raises, the held-back ending of
the last attempt is released unchanged — the turn degrades to exactly what it
did before this hook (blank -> rag_filter's notice), never to a broken stream.

async_post_call_success_hook covers non-stream calls (Open WebUI's title /
follow-up / tag tasks): returning a response object replaces the original
(utils.py ProxyLogging.post_call_success_hook), returning None keeps it.

Limits (residual risk)
----------------------
* A turn that streams some text and THEN ends malformed is not retried — the
  text is already on the user's screen. Not observed so far (all P169 cases
  had zero output tokens).
* Each retry is a full extra prompt (~18k input tokens on the observed turn)
  billed by Google; max 2 per turn.
* The retry runs on a fresh LiteLLM logging object, so spend/trace shows the
  retry as its own call (the empty attempt is logged as before).

Deployed by mounting this file next to /app/config.yaml in the litellm container and
listing it in litellm-config.yaml under litellm_settings.callbacks.
"""
from __future__ import annotations

import copy
import logging
import os
import uuid

try:  # Real base class inside the litellm container.
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # Offline tests without litellm installed.
    CustomLogger = object  # type: ignore[assignment,misc]

try:
    # LiteLLM configures this logger's handler/level; a plain getLogger()
    # would print nothing in the container at WARNING without a handler.
    from litellm._logging import verbose_proxy_logger as log
except ImportError:
    log = logging.getLogger("litellm-hooks.gemini-malformed-retry")

MAX_RETRIES = 2
# Finish reasons a blank P169 turn arrives with (None: stream ended without
# a finish chunk). Anything else is a deliberate stop we must not repeat.
_RETRYABLE_FINISH = {None, "stop"}
# Fields that tie a request to its first attempt; the retry must get its own,
# or LiteLLM folds two provider calls into one logging/spend record.
_PER_ATTEMPT_KEYS = ("litellm_logging_obj",)
_END = object()

# Appended as a trailing system message on BOTH retries. Live 2026-10-25
# (chat 7fb11f85, turns 3 and 5) identical re-sends came back blank 2/2 twice:
# the failure is a property of the request/context, not a coin flip, so a
# retry has to change what the model sees. The stored reasoning of both blank
# turns shows the model was about to write a long multi-line run_command
# script, i.e. it failed while emitting the call, so the nudge names the two
# well-formed shapes it may answer with. Kept short: it is sent with ~18k
# tokens of history and must not compete with the system prompt.
NUDGE_TEXT = ("Your previous attempt at this reply returned nothing because its function "
              "call could not be parsed. Reply now with either plain text, or exactly one "
              "valid function call to a declared tool whose arguments are a JSON object "
              "matching that tool's parameter schema exactly.")

# Retry 2 also changes sampling (lead's call, 2026-10-26). Note LiteLLM already
# sends temperature 1.0 to Gemini 3 when the request sets none, and Google
# recommends 1.0 for Gemini 3 (lower can loop on long reasoning) — so this is a
# deliberate one-turn deviation on a turn that has already failed twice, not a
# general setting: unset -> 0.7, a set value moves by 0.2 (down from >0.5,
# else up).
_RETRY2_DEFAULT_TEMPERATURE = 0.7
_TEMPERATURE_STEP = 0.2


def _nudged_temperature(current):
    if current is None:
        return _RETRY2_DEFAULT_TEMPERATURE
    current = float(current)
    moved = current - _TEMPERATURE_STEP if current > 0.5 else current + _TEMPERATURE_STEP
    return round(moved, 2)


def _validated_allowed(data: dict) -> bool:
    """VALIDATED only where it means "AUTO, but constrained-decode the call":
    tools declared and no explicit tool_choice other than "auto". An explicit
    "none"/"required"/named tool is the caller's decision and is kept."""
    if not data.get("tools"):
        return False
    return data.get("tool_choice") in (None, "auto")


def _add_validated(data: dict) -> None:
    """Ask Gemini for functionCallingConfig.mode = VALIDATED.

    Google's v1beta discovery doc (FunctionCallingConfig.mode enum, fetched
    2026-10-26): "Model decides to predict either a function call or a natural
    language response, but will validate function calls with constrained
    decoding" — the documented lever against malformed calls. LiteLLM 1.100.0
    cannot express it via tool_choice (auto/required/none only), but merges
    extra_body into the Gemini request body verbatim (verified offline: body
    gets toolConfig.functionCallingConfig.mode = VALIDATED). tool_choice "auto"
    is dropped so two toolConfig sources are never merged."""
    if data.get("tool_choice") == "auto":
        data.pop("tool_choice")
    extra = dict(data.get("extra_body") or {})
    tool_config = dict(extra.get("toolConfig") or {})
    fcc = dict(tool_config.get("functionCallingConfig") or {})
    fcc["mode"] = "VALIDATED"
    tool_config["functionCallingConfig"] = fcc
    extra["toolConfig"] = tool_config
    data["extra_body"] = extra


def _is_gemini(request_data: dict) -> bool:
    return "gemini" in str((request_data or {}).get("model", "")).lower()


def _delta_has_output(chunk) -> bool:
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None):
            return True
        if getattr(delta, "tool_calls", None) or getattr(delta, "function_call", None):
            return True
    return False


def _finish_reason(chunk):
    for choice in getattr(chunk, "choices", None) or []:
        reason = getattr(choice, "finish_reason", None)
        if reason:
            return reason
    return None


def _response_is_empty_stop(response) -> bool:
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg is None:
            return False
        if getattr(msg, "content", None) or getattr(msg, "tool_calls", None) \
                or getattr(msg, "function_call", None):
            return False
        if getattr(choice, "finish_reason", None) not in _RETRYABLE_FINISH:
            return False
    return True


class GeminiMalformedRetry(CustomLogger):
    def __init__(self, max_retries: int = MAX_RETRIES):
        if CustomLogger is not object:
            super().__init__()
        self.max_retries = max_retries

    # -- the re-call; overridden in tests -----------------------------------
    async def _recall(self, request_data: dict):
        from litellm.proxy.proxy_server import llm_router

        return await llm_router.acompletion(**request_data)

    def _retry_data(self, request_data: dict, attempt: int, validated: bool = True):
        """Build retry `attempt` (1-based). Returns (data, variant, used_validated).

        Retry 1: the same request + NUDGE_TEXT as a trailing system message.
        Retry 2: the nudge + a sampling change + function-calling mode
                 VALIDATED (when tools are declared and tool_choice allows it).
        The caller's dict is never mutated — LiteLLM still uses it for the
        first attempt's logging.
        """
        data = copy.copy(request_data)
        for key in _PER_ATTEMPT_KEYS:
            data.pop(key, None)
        data["litellm_call_id"] = str(uuid.uuid4())
        # A trailing system message is safe in any position: LiteLLM's Gemini
        # transformation folds every system message into system_instruction
        # (checked against 1.100.0 after both a user turn and a tool result),
        # so it never breaks the user/model/function_response ordering.
        data["messages"] = list(request_data.get("messages") or []) + [
            {"role": "system", "content": NUDGE_TEXT}]
        variant = ["nudge"]
        used_validated = False
        if attempt >= 2:
            data["temperature"] = _nudged_temperature(request_data.get("temperature"))
            variant.append(f"temperature={data['temperature']:g}")
            if validated and _validated_allowed(request_data):
                _add_validated(data)
                used_validated = True
                variant.append("validated")
        return data, "+".join(variant), used_validated

    async def _open_retry(self, request_data: dict, attempt: int, label: str):
        """Send retry `attempt`; if Google rejects VALIDATED (400), re-send the
        same retry without it so the last chance is not lost to an untested
        enum value. Returns the provider reply (stream or response)."""
        data, variant, used_validated = self._retry_data(request_data, attempt)
        log.warning("P169 empty Gemini reply on %s (%s): retry %d/%d (variant=%s)",
                    request_data.get("model"), label, attempt, self.max_retries, variant)
        try:
            return await self._first_ready(await self._recall(data))
        except Exception as exc:
            if not (used_validated and getattr(exc, "status_code", None) == 400):
                raise
            log.warning("P169 VALIDATED mode rejected on %s (%r): re-running retry %d/%d "
                        "without VALIDATED", request_data.get("model"), exc, attempt,
                        self.max_retries)
            data, variant, _ = self._retry_data(request_data, attempt, validated=False)
            return await self._first_ready(await self._recall(data))

    @staticmethod
    async def _first_ready(reply):
        """For a stream, pull the first chunk now so that an error LiteLLM only
        raises on first iteration (Gemini streams open the HTTP call lazily)
        surfaces here, where it can still be handled, instead of mid-way
        through the client's stream. Non-stream replies pass through."""
        if not hasattr(reply, "__aiter__"):
            return reply
        it = reply.__aiter__()
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = _END

        async def chained():
            if first is _END:
                return
            yield first
            async for item in it:
                yield item
        return chained()

    # -- opt-in source-side change ------------------------------------------
    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        """With GEMINI_FC_MODE_VALIDATED=1, every Gemini request that declares
        tools asks for function-calling mode VALIDATED (constrained decoding
        of function calls). Off by default: the mode is documented in Google's
        v1beta discovery doc but has not been exercised on gemini-3.8-flash —
        turn it on only after one live probe shows it is accepted."""
        if os.environ.get("GEMINI_FC_MODE_VALIDATED") != "1":
            return data
        if _is_gemini(data) and _validated_allowed(data):
            _add_validated(data)
        return data

    # -- streaming (chat turns from Open WebUI) -----------------------------
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response,
                                                      request_data: dict):
        if not _is_gemini(request_data):
            async for item in response:
                yield item
            return

        stream = response
        attempt = 0
        while True:
            saw_output = False
            held = []  # finish chunk and anything after it, for this attempt
            async for item in stream:
                if held:
                    held.append(item)
                    continue
                if _delta_has_output(item):
                    saw_output = True
                if _finish_reason(item) and not saw_output:
                    held.append(item)
                    continue
                yield item

            reason = next((_finish_reason(c) for c in held if _finish_reason(c)), None)
            retryable = not saw_output and reason in _RETRYABLE_FINISH
            if not retryable:
                for item in held:
                    yield item
                return
            if attempt >= self.max_retries:
                log.warning("P169 empty Gemini reply on %s: giving up after %d retries, "
                            "passing the blank turn through", request_data.get("model"), attempt)
                for item in held:
                    yield item
                return
            attempt += 1
            try:
                stream = await self._open_retry(request_data, attempt,
                                                f"finish_reason={reason}, no text/tool call")
            except Exception as exc:  # degrade: release the original ending
                log.warning("P169 retry failed on %s: %r - passing the blank turn through",
                            request_data.get("model"), exc)
                for item in held:
                    yield item
                return

    # -- non-streaming (title / follow-up / tag tasks) ----------------------
    async def async_post_call_success_hook(self, data: dict, user_api_key_dict, response):
        if not _is_gemini(data) or not _response_is_empty_stop(response):
            return None
        for attempt in range(1, self.max_retries + 1):
            try:
                retried = await self._open_retry(data, attempt, "non-stream")
            except Exception as exc:
                log.warning("P169 retry failed on %s: %r - keeping the blank reply",
                            data.get("model"), exc)
                return None
            if not _response_is_empty_stop(retried):
                return retried
        log.warning("P169 empty Gemini reply on %s: giving up after %d retries",
                    data.get("model"), self.max_retries)
        return None


# Referenced from litellm-config.yaml: callbacks: gemini_malformed_retry.proxy_handler_instance
proxy_handler_instance = GeminiMalformedRetry()
