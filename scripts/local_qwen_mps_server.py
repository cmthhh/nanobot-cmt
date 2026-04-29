#!/usr/bin/env python3
"""Small OpenAI-compatible Qwen server backed by Transformers + MPS."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import json_repair
except Exception:  # pragma: no cover - optional robustness only
    json_repair = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen" / "Qwen2___5-7B-Instruct"
DEFAULT_MODEL_NAME = "qwen2.5-7b"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ServerSettings:
    model_dir: Path
    served_model_name: str
    device: str
    dtype: str
    max_input_tokens: int
    max_new_tokens: int


@dataclass
class GenerationResult:
    id: str
    created: int
    model: str
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    usage: dict[str, int]


@dataclass
class PreparedGeneration:
    tokenizer: Any
    model: Any
    model_inputs: dict[str, Any]
    prompt_tokens: int
    max_new_tokens: int
    generation_kwargs: dict[str, Any]


class Runtime:
    def __init__(self) -> None:
        self.settings = ServerSettings(
            model_dir=Path(os.environ.get("NANOBOT_LOCAL_MODEL_DIR", DEFAULT_MODEL_DIR)),
            served_model_name=os.environ.get("NANOBOT_LOCAL_MODEL_NAME", DEFAULT_MODEL_NAME),
            device=os.environ.get("NANOBOT_LOCAL_DEVICE", "auto"),
            dtype=os.environ.get("NANOBOT_LOCAL_DTYPE", "auto"),
            max_input_tokens=int(os.environ.get("NANOBOT_LOCAL_MAX_INPUT_TOKENS", "8192")),
            max_new_tokens=int(os.environ.get("NANOBOT_LOCAL_MAX_NEW_TOKENS", "1024")),
        )
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"
        self.lock = threading.Lock()


runtime = Runtime()


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    repetition_penalty: float | None = None

    model_config = ConfigDict(extra="allow")


def _select_device(requested: str) -> str:
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested, but this PyTorch build does not include MPS.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is false. "
                "Check that this conda environment is using an Apple Silicon PyTorch build."
            )
        return "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _select_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def load_model() -> None:
    settings = runtime.settings
    model_dir = settings.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")

    runtime.device = _select_device(settings.device)
    dtype = _select_dtype(runtime.device, settings.dtype)
    print(
        f"Loading {settings.served_model_name} from {model_dir} "
        f"on {runtime.device} ({dtype})",
        flush=True,
    )

    runtime.tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    runtime.model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    runtime.model.to(runtime.device)
    runtime.model.eval()
    print("Local Qwen API is ready.", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Nanobot Local Qwen MPS API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime.model is not None else "loading",
        "model": runtime.settings.served_model_name,
        "device": runtime.device,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.settings.served_model_name,
                "object": "model",
                "created": now,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> Any:
    if runtime.model is None or runtime.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    if request.stream:
        return StreamingResponse(
            _stream_generate_openai_chunks(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await asyncio.to_thread(_generate, request)
    return JSONResponse(_completion_response(result))


def _prepare_generation(request: ChatCompletionRequest) -> PreparedGeneration:
    tokenizer = runtime.tokenizer
    model = runtime.model
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded")

    messages = _normalize_messages(request.messages)
    tools = _normalize_tools(request.tools, request.tool_choice)
    prompt = _apply_chat_template(tokenizer, messages, tools)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if input_ids.shape[-1] > runtime.settings.max_input_tokens:
        input_ids = input_ids[:, -runtime.settings.max_input_tokens :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -runtime.settings.max_input_tokens :]

    model_inputs: dict[str, Any] = {"input_ids": input_ids.to(runtime.device)}
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(runtime.device)

    prompt_tokens = int(input_ids.shape[-1])
    max_new_tokens = _bounded_max_tokens(request.max_tokens, runtime.settings.max_new_tokens)
    generation_kwargs = _generation_kwargs(request, tokenizer, max_new_tokens)

    return PreparedGeneration(
        tokenizer=tokenizer,
        model=model,
        model_inputs=model_inputs,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        generation_kwargs=generation_kwargs,
    )


def _generate(request: ChatCompletionRequest) -> GenerationResult:
    prepared = _prepare_generation(request)
    with runtime.lock, torch.inference_mode():
        try:
            outputs = prepared.model.generate(
                **prepared.model_inputs,
                **prepared.generation_kwargs,
            )
        except RuntimeError as exc:
            if (
                not _is_invalid_sampling_error(exc)
                or not prepared.generation_kwargs.get("do_sample")
            ):
                raise
            fallback_kwargs = {
                key: value
                for key, value in prepared.generation_kwargs.items()
                if key not in {"do_sample", "temperature", "top_p"}
            }
            fallback_kwargs["do_sample"] = False
            outputs = prepared.model.generate(**prepared.model_inputs, **fallback_kwargs)

    new_token_ids = outputs[0][prepared.prompt_tokens:]
    completion_tokens = int(new_token_ids.shape[-1])
    text = prepared.tokenizer.decode(new_token_ids, skip_special_tokens=True)
    text, stop_hit = _apply_stop(text, request.stop)
    content, tool_calls = _extract_tool_calls(text)

    hit_length = completion_tokens >= prepared.max_new_tokens and not stop_hit and not tool_calls
    finish_reason = "tool_calls" if tool_calls else ("length" if hit_length else "stop")
    if runtime.device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    return GenerationResult(
        id=f"chatcmpl-local-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model or runtime.settings.served_model_name,
        content=content or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage={
            "prompt_tokens": prepared.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prepared.prompt_tokens + completion_tokens,
        },
    )


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        data = message.model_dump(exclude_none=True)
        role = str(data.get("role") or "user")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        item: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(data.get("content")),
        }
        if role == "assistant" and data.get("tool_calls"):
            item["tool_calls"] = _normalize_prior_tool_calls(data["tool_calls"])
        if role == "tool":
            if data.get("tool_call_id"):
                item["tool_call_id"] = data["tool_call_id"]
            if data.get("name"):
                item["name"] = data["name"]
        normalized.append(item)
    return normalized


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type"):
                    parts.append(f"[{item['type']} omitted]")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_prior_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            function = call if isinstance(call, dict) else {}
        name = str(function.get("name") or "")
        args = _coerce_arguments(function.get("arguments", {}))
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        )
    return normalized


def _normalize_tools(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not _env_bool("NANOBOT_LOCAL_ENABLE_TOOLS", default=False):
        return None
    if not tools or tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        requested = (tool_choice.get("function") or {}).get("name")
        if requested:
            narrowed = [
                tool for tool in tools
                if (tool.get("function") or {}).get("name") == requested
            ]
            return narrowed or tools
    return tools


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_max_tokens(requested: int | None, server_limit: int) -> int:
    if requested is None:
        return min(512, server_limit)
    return max(1, min(int(requested), server_limit))


def _generation_kwargs(
    request: ChatCompletionRequest,
    tokenizer: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    temperature = 0.7 if request.temperature is None else float(request.temperature)
    top_p = 0.9 if request.top_p is None else max(0.01, min(float(request.top_p), 1.0))
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }
    if request.repetition_penalty:
        kwargs["repetition_penalty"] = float(request.repetition_penalty)
    if temperature <= 0.2:
        kwargs["do_sample"] = False
    else:
        kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    return kwargs


def _is_invalid_sampling_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "probability tensor contains" in message
        or "inf" in message
        or "nan" in message
    )


def _apply_stop(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    if not stop:
        return text, False
    stops = [stop] if isinstance(stop, str) else [s for s in stop if s]
    positions = [text.find(s) for s in stops if s in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        payload = _loads_jsonish(match.group(1).strip())
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not name and isinstance(payload.get("function"), dict):
            name = payload["function"].get("name")
        if not name:
            continue
        arguments = _coerce_arguments(payload.get("arguments", {}))
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    if not calls:
        return text.strip(), []
    content = TOOL_CALL_RE.sub("", text).strip()
    return content, calls


def _coerce_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _loads_jsonish(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _loads_jsonish(raw: str) -> Any:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        if json_repair is not None:
            try:
                return json_repair.loads(cleaned)
            except Exception:
                return None
    return None


def _completion_response(result: GenerationResult) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.content,
    }
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls

    return {
        "id": result.id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": result.usage,
    }


_STREAM_DONE = object()


async def _stream_generate_openai_chunks(request: ChatCompletionRequest) -> AsyncIterator[str]:
    request_id = f"chatcmpl-local-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = request.model or runtime.settings.served_model_name
    completion_parts: list[str] = []
    sent_any_content = False

    yield _sse(_stream_chunk(request_id, created, model_name, {"role": "assistant"}, None))
    await asyncio.sleep(0)

    try:
        prepared = _prepare_generation(request)
    except Exception as exc:
        yield _sse(_stream_chunk(
            request_id,
            created,
            model_name,
            {"content": f"Error preparing local generation: {exc}"},
            "stop",
        ))
        yield "data: [DONE]\n\n"
        return

    streamer = TextIteratorStreamer(
        prepared.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = {
        **prepared.model_inputs,
        **prepared.generation_kwargs,
        "streamer": streamer,
    }
    errors: list[BaseException] = []

    def _worker() -> None:
        with runtime.lock, torch.inference_mode():
            try:
                prepared.model.generate(**generation_kwargs)
            except BaseException as exc:
                errors.append(exc)
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass
            finally:
                if runtime.device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    pending = ""
    iterator = iter(streamer)
    while True:
        item = await asyncio.to_thread(_next_stream_item, iterator)
        if item is _STREAM_DONE:
            break
        text = str(item)
        if not text:
            continue
        completion_parts.append(text)
        pending += text
        if _could_be_tool_call_prefix(pending):
            continue
        if pending:
            sent_any_content = True
            yield _sse(_stream_chunk(request_id, created, model_name, {"content": pending}, None))
            pending = ""
            await asyncio.sleep(0)

    thread.join(timeout=0)
    full_text = "".join(completion_parts)
    full_text, stop_hit = _apply_stop(full_text, request.stop)
    content, tool_calls = _extract_tool_calls(full_text)
    completion_tokens = _count_completion_tokens(prepared.tokenizer, full_text)
    hit_length = (
        completion_tokens >= prepared.max_new_tokens
        and not stop_hit
        and not tool_calls
        and not errors
    )
    finish_reason = "tool_calls" if tool_calls else ("length" if hit_length else "stop")

    if errors:
        err = errors[0]
        if _is_invalid_sampling_error(err) and generation_kwargs.get("do_sample"):
            finish_reason = "stop"
            if not sent_any_content:
                yield _sse(_stream_chunk(
                    request_id,
                    created,
                    model_name,
                    {"content": "Local generation hit a sampling instability; retry with temperature <= 0.2."},
                    None,
                ))
        else:
            finish_reason = "stop"
            if not sent_any_content:
                yield _sse(_stream_chunk(
                    request_id,
                    created,
                    model_name,
                    {"content": f"Error calling local model: {err}"},
                    None,
                ))
    elif tool_calls and not sent_any_content:
        for index, call in enumerate(tool_calls):
            yield _sse(_stream_chunk(
                request_id,
                created,
                model_name,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    ]
                },
                None,
            ))
            await asyncio.sleep(0)
    elif pending:
        yield _sse(_stream_chunk(request_id, created, model_name, {"content": pending}, None))

    usage = {
        "prompt_tokens": prepared.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prepared.prompt_tokens + completion_tokens,
    }
    yield _sse(_stream_chunk(request_id, created, model_name, {}, finish_reason))
    yield _sse({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [],
        "usage": usage,
    })
    yield "data: [DONE]\n\n"


def _next_stream_item(iterator: Any) -> str | object:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_DONE


def _could_be_tool_call_prefix(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return True
    return "<tool_call".startswith(stripped) or stripped.startswith("<tool_call")


def _count_completion_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


async def _stream_openai_chunks(result: GenerationResult) -> AsyncIterator[str]:
    yield _sse(_chunk(result, {"role": "assistant"}, None))
    await asyncio.sleep(0)

    if result.tool_calls:
        for index, call in enumerate(result.tool_calls):
            delta = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        },
                    }
                ]
            }
            yield _sse(_chunk(result, delta, None))
            await asyncio.sleep(0)
    elif result.content:
        for part in _split_stream_text(result.content):
            yield _sse(_chunk(result, {"content": part}, None))
            await asyncio.sleep(0)

    yield _sse(_chunk(result, {}, result.finish_reason))
    yield _sse({
        "id": result.id,
        "object": "chat.completion.chunk",
        "created": result.created,
        "model": result.model,
        "choices": [],
        "usage": result.usage,
    })
    yield "data: [DONE]\n\n"


def _chunk(
    result: GenerationResult,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": result.id,
        "object": "chat.completion.chunk",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _stream_chunk(
    request_id: str,
    created: int,
    model_name: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _split_stream_text(text: str, size: int = 80) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=runtime.settings.model_dir)
    parser.add_argument("--served-model-name", default=runtime.settings.served_model_name)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--device", default=runtime.settings.device, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default=runtime.settings.dtype,
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-input-tokens", type=int, default=runtime.settings.max_input_tokens)
    parser.add_argument("--max-new-tokens", type=int, default=runtime.settings.max_new_tokens)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime.settings = ServerSettings(
        model_dir=args.model_dir,
        served_model_name=args.served_model_name,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
