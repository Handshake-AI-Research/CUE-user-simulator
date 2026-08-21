"""Shared prompt-embedding inference helper for evaluation."""

import json
import os
import urllib.error
import urllib.request

from cue_training.runlog.log import warn


def post_prompt_embeds(
    base_url: str,
    model: str,
    b64: str,
    *,
    api_key_env: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    timeout: float = 600.0,
    tag: str = "decoder",
    include_empty_prompt: bool = False,
) -> str:
    body: dict[str, object] = {
        "model": model,
        "prompt_embeds": b64,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if include_empty_prompt:
        body["prompt"] = ""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get(api_key_env, '') or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            output = json.loads(response.read().decode("utf-8"))
        return (output["choices"][0]["text"] or "").strip()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:  # noqa: BLE001
            pass
        warn(tag, f"vllm completion HTTP {exc.code}: {detail}")
        return ""
    except Exception as exc:  # noqa: BLE001
        warn(tag, f"vllm completion failed: {exc}")
        return ""
