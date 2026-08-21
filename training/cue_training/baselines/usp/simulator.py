"""Published USP model served through its documented chat-template interface."""

from __future__ import annotations

from typing import Any

from cue_training.baselines.common.simulator import BaseUserSimulator
from cue_training.baselines.usp.sampler import ProfileSampler

Turn = dict[str, str]


class _USPHTTPError(Exception):
    """Carries the vLLM error body so callers can parse the reported token counts."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"USP vLLM {code}")
        self.code = code
        self.detail = detail

    def reported_max_len(self) -> int | None:
        import re

        match = re.search(r"maximum context length is (\d+) tokens", self.detail)
        return int(match.group(1)) if match else None

    def reported_prompt_tokens(self) -> int | None:
        """The exact prompt size vLLM measured, in tokens.

        vLLM uses two body shapes: token counts (``prompt contains at least N input
        tokens``) or, when 0 input tokens are allowed, a character count (``prompt
        contains N characters``). Convert characters conservatively (~3 chars/token)."""

        import math
        import re

        match = re.search(r"prompt contains (?:at least )?(\d+) input tokens", self.detail)
        if match:
            return int(match.group(1))
        match = re.search(r"prompt contains (\d+) characters", self.detail)
        if match:
            return math.ceil(int(match.group(1)) / 3)
        return None


class USPOfficialVLLMSimulator(BaseUserSimulator):
    """HuggingFace ``wangkevin02/USP`` via a local vLLM OpenAI chat server.

    Uses the published model-card chat template (profile as system content, then
    conversation turns). USP generates customer text in the assistant slot, so
    benchmark roles are inverted for agent-first histories (e.g. tau2).
    ``batched=True`` tells the sidecar to skip its GPU lock.
    """

    name = "usp"
    batched = True

    def __init__(
        self,
        profiles: list[dict],
        *,
        base_url: str,
        model: str,
        api_key_env: str = "HOSTED_VLLM_API_KEY",
        seed: int = 0,
        max_new_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        timeout: float = 600.0,
        max_context_chars: int = 9000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.timeout = timeout
        self.max_context_chars = max_context_chars
        self.sampler = ProfileSampler(profiles, seed=seed)
        self._profile_text = self.sampler.sample_text()
        # Episode task/scenario (Tau2 identity, SimArena goal, …). Kept separate from the
        # soft persona so the published "Your profile is:" system message stays the sole
        # system content while still grounding the customer in private task facts.
        self._task_text = ""
        self._server_max_len: int | None | bool = False  # False = not yet queried

    # Fallback window before the server tells us otherwise. vLLM rejects
    # prompt+max_tokens > max_model_len up front, so requesting the 4096 card ceiling
    # 400s on every call; assume the observed 4096 window until an error corrects us.
    _DEFAULT_WINDOW = 4096

    def _on_episode_start(self, task: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        """Resample the pool profile; harness may overwrite ``_profile_text`` with a sidecar persona."""

        self._task_text = str(task or "").strip()
        self._profile_text = self.sampler.sample_text()

    def _profile_block(self) -> str:
        """Persona (+ optional task facts) under the published profile slot."""

        profile = str(self._profile_text or "").rstrip()
        task = str(self._task_text or "").strip()
        if not task:
            return profile
        # Keep task facts inside the profile block — never a second system message —
        # so the HF chat-template contract stays intact.
        return (
            f"{profile}\n\n"
            "Private facts for this conversation (you already know these as the customer; "
            "use them when needed, do not recite verbatim unless asked):\n"
            f"{task}"
        )

    def _window(self) -> int:
        """Server context window: the ``max_model_len`` from ``/v1/models`` if exposed,
        otherwise the default (later corrected from the first context-length error)."""

        if self._server_max_len is False:
            import json
            import os
            import urllib.request

            self._server_max_len = None
            api_key = os.environ.get(self.api_key_env, "") or "EMPTY"
            probe = urllib.request.Request(
                f"{self.base_url}/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            try:
                with urllib.request.urlopen(probe, timeout=10) as resp:  # noqa: S310
                    data = json.loads(resp.read().decode("utf-8")).get("data") or []
                for row in data:
                    if isinstance(row, dict) and row.get("max_model_len"):
                        self._server_max_len = int(row["max_model_len"])
                        break
            except Exception:  # noqa: BLE001
                self._server_max_len = None
        return int(self._server_max_len) if self._server_max_len else self._DEFAULT_WINDOW

    def _completion_budget(self, messages: list[dict[str, str]]) -> int:
        # Overestimate the prompt (~3 chars/token) and keep a wide margin so the first
        # attempt almost always fits; the retry corrects any remaining slack exactly.
        approx_prompt = sum(len(m.get("content") or "") // 3 + 8 for m in messages)
        return max(64, min(self.max_new_tokens, self._window() - approx_prompt - 256))

    def first_turn(self, task: str, metadata: dict[str, Any]) -> str:
        self._task_text = str(task or "").strip()
        # Do not resample here: ``_on_episode_start`` (or the harness persona inject)
        # already set ``_profile_text`` for this episode.
        return self.next_turn(task, [], metadata)

    def _messages(self, history: list[Turn]) -> list[dict[str, str]]:
        # Match the published model card: the profile is the sole system content.
        system = (
            "You are engaging in a conversation with an AI assistant. "
            f"Your profile is:\n{self._profile_block()}\n"
            "You can say anything you want, either based on the profile or something brand new.\n"
        )
        budget = max(0, self.max_context_chars - len(system))
        if not history:
            return [{"role": "system", "content": system}]

        # USP generates customer text in the assistant slot, so invert benchmark
        # roles to keep agent-first conversations valid for its chat template.
        opening = dict(history[0])
        opening["role"] = "assistant" if opening.get("role") == "user" else "user"
        opening["content"] = str(opening.get("content") or "")[:budget]
        kept = [opening]
        used = len(opening["content"])
        for turn in reversed(history[1:]):
            content = str(turn.get("content") or "")
            if used + len(content) > budget:
                continue
            role = "assistant" if turn.get("role") == "user" else "user"
            kept.append({"role": role, "content": content})
            used += len(content)
        kept[1:] = reversed(kept[1:])
        return [{"role": "system", "content": system}, *kept]

    def _post(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        import json
        import os
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        api_key = os.environ.get(self.api_key_env, "") or "EMPTY"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # vLLM puts the actual reason (e.g. context length) in the body; the bare
            # "HTTP Error 400: Bad Request" alone is undiagnosable.
            detail = exc.read().decode("utf-8", "replace")
            raise _USPHTTPError(exc.code, detail) from exc

    def next_turn(self, task: str, history: list[Turn], metadata: dict[str, Any]) -> str:  # noqa: ARG002
        if task is not None:
            self._task_text = str(task).strip()
        messages = self._messages(history)
        try:
            out = self._post(messages, self._completion_budget(messages))
        except _USPHTTPError as exc:
            # The estimate can undershoot the tokenizer; the error body reports the exact
            # window and prompt size, so cache the window and retry with a budget it accepts.
            window = exc.reported_max_len()
            if window:
                self._server_max_len = window
            window = window or self._window()
            prompt = exc.reported_prompt_tokens()
            budget = window - prompt - 8 if prompt is not None else 0
            if budget < 16:
                raise RuntimeError(f"USP vLLM {exc.code}: {exc.detail[:500]}") from exc
            out = self._post(messages, budget)
        return (out["choices"][0].get("message", {}).get("content") or "").strip()
