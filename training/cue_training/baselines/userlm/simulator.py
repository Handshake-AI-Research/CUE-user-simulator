"""UserLM-8b user simulator with the paper's generation guardrails (Appendix C.1).

UserLM-8b (https://huggingface.co/microsoft/UserLM-8b) predicts the *user* side of a
conversation. Guardrails (paper extrinsic simulation):

1. Filter first tokens I / You / Here (lower + capitalized) at position 0
2. Suppress ``<|endconversation|>`` on the *first user turn* only (CUE keeps this
   prevention because tau2 is often agent-first; after turn 1 the end token is the
   paper termination method)
3. Resample until utterance is 3–25 words
4. Resample until the turn is novel vs prior user turns / the intent string

Termination after the first user turn is via ``<|endconversation|>`` (``is_done``),
honored by ``_StopGuard`` (which still blocks end on the first user turn).
"""

from __future__ import annotations

from typing import Any

from cue_training.baselines.common.simulator import BaseUserSimulator

Turn = dict[str, str]

_MODEL_ID = "microsoft/UserLM-8b"
_END_TOKEN = "<|endconversation|>"
_BANNED_FIRST = ("I", "You", "Here", "i", "you", "here")
_MIN_WORDS = 3
_MAX_WORDS = 25
_MAX_RESAMPLES = 12


def _strip_role_prefix(text: str) -> str:
    for prefix in ("user:", "User:", "USER:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _word_count(text: str) -> int:
    return len(text.split())


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _starts_with_banned(text: str) -> bool:
    first = (text.split() or [""])[0]
    # Strip leading punctuation so "'I" / "I," still count.
    core = "".join(ch for ch in first if ch.isalpha())
    return core in _BANNED_FIRST


def _is_novel(text: str, history: list[Turn], intent: str) -> bool:
    target = _norm(text)
    if not target:
        return False
    if intent and target == _norm(intent):
        return False
    for turn in history:
        if turn.get("role") == "user" and _norm(str(turn.get("content") or "")) == target:
            return False
    return True


def _passes_guardrails(text: str, history: list[Turn], intent: str) -> bool:
    if not text:
        return False
    if _starts_with_banned(text):
        return False
    n = _word_count(text)
    if n < _MIN_WORDS or n > _MAX_WORDS:
        return False
    return _is_novel(text, history, intent)


class UserLMSimulator(BaseUserSimulator):
    name = "userlm"

    def __init__(
        self,
        *,
        model_id: str = _MODEL_ID,
        device: str = "cuda",
        temperature: float = 1.0,
        top_p: float = 0.8,
        max_new_tokens: int = 256,
        never_end_first_turn: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

        self.never_end_first_turn = never_end_first_turn
        self.device = device if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.model.generation_config.max_length = None
        self._end_id = self.tokenizer.convert_tokens_to_ids(_END_TOKEN)
        if self._end_id is None or self._end_id == self.tokenizer.unk_token_id:
            self._end_id = None
        # HF README: eos = <|eot_id|>
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self._eos_id = eot if eot is not None and eot != self.tokenizer.unk_token_id else self.tokenizer.eos_token_id
        banned_ids: list[int] = []
        for word in _BANNED_FIRST:
            tid = self.tokenizer.convert_tokens_to_ids(word)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                banned_ids.append(int(tid))
            # Also try leading-space BPE variants.
            tid_sp = self.tokenizer.encode(f" {word}", add_special_tokens=False)
            if len(tid_sp) == 1:
                banned_ids.append(int(tid_sp[0]))
        self._banned_first_ids = sorted(set(banned_ids))

        class _FirstTokenBan(LogitsProcessor):
            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

            def __call__(self, input_ids, scores):  # noqa: ANN001
                if not self.ids:
                    return scores
                # Only the first generated token: input already has the prompt; scores is for next.
                # Transformers calls this each step; ban only when no new tokens yet relative to prompt
                # is handled by checking scores batch — we ban whenever generated length is 0 by
                # comparing to a captured prompt length set on the processor before generate.
                prompt_len = getattr(self, "prompt_len", None)
                if prompt_len is not None and input_ids.shape[-1] == prompt_len:
                    scores[:, self.ids] = float("-inf")
                return scores

        self._FirstTokenBan = _FirstTokenBan
        self._done = False
        self._user_turns = 0

    def _build_messages(self, task: str, history: list[Turn]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": task or "Have a conversation."}]
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        return messages

    def _generate(self, messages: list[dict[str, str]], allow_end: bool) -> str:
        import torch
        from transformers import LogitsProcessorList

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        suppress = []
        if not allow_end and self._end_id is not None:
            suppress = [self._end_id]

        ban = self._FirstTokenBan(self._banned_first_ids)
        ban.prompt_len = input_ids.shape[-1]
        processors = LogitsProcessorList([ban]) if self._banned_first_ids else None

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                suppress_tokens=suppress or None,
                logits_processor=processors,
                eos_token_id=self._eos_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1] :]
        if self._end_id is not None and self._end_id in new_tokens.tolist():
            self._done = True
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = text.replace(_END_TOKEN, "").strip()
        return _strip_role_prefix(text)

    def first_turn(self, task: str, metadata: dict[str, Any]) -> str:
        self._done = False
        self._user_turns = 0
        return self.next_turn(task, [], metadata, _opening=True)

    def next_turn(
        self,
        task: str,
        history: list[Turn],
        metadata: dict[str, Any],
        _opening: bool = False,
    ) -> str:
        prior_users = sum(1 for t in history if t.get("role") == "user")
        is_first_user_turn = prior_users == 0 or _opening
        if is_first_user_turn:
            self._done = False
            self._user_turns = 0
        messages = self._build_messages(task, history)
        allow_end = not (is_first_user_turn and self.never_end_first_turn)
        intent = task or ""
        text = ""
        for _ in range(_MAX_RESAMPLES):
            self._done = False if not allow_end else self._done
            candidate = self._generate(messages, allow_end=allow_end)
            # If the model ended without text, treat as empty and resample (unless allowed end).
            if self._done and not candidate and allow_end:
                self._user_turns += 1
                return ""
            if _passes_guardrails(candidate, history, intent):
                text = candidate
                break
            # Rejected: clear accidental end flag from a bad sample.
            self._done = False
            text = candidate  # keep last attempt if all fail
        if not text:
            # Final forced resample with end suppressed.
            text = self._generate(messages, allow_end=False)
            self._done = False
        self._user_turns += 1
        if is_first_user_turn:
            self._done = False
        return text

    def is_done(self, history: list[Turn]) -> bool:  # noqa: ARG002
        return self._done


class UserLMVLLMSimulator(BaseUserSimulator):
    """UserLM-8b via vLLM with the same Appendix C.1 guardrails (approximate)."""

    name = "userlm"
    batched = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "HOSTED_VLLM_API_KEY",
        temperature: float = 1.0,
        top_p: float = 0.8,
        max_new_tokens: int = 256,
        timeout: float = 600.0,
        never_end_first_turn: bool = True,
        context_len: int = 8192,
    ) -> None:
        self.never_end_first_turn = never_end_first_turn
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.context_len = int(context_len)
        self._done = False
        self._user_turns = 0

    def _build_messages(self, task: str, history: list[Turn]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": task or "Have a conversation."}]
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        return self._fit_messages(messages)

    def _fit_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        budget = max(256, self.context_len - self.max_new_tokens - 64)

        def est(msg: dict[str, str]) -> int:
            return len(msg.get("content", "")) // 3 + 8

        if not messages:
            return messages
        system, rest = messages[:1], messages[1:]
        total = est(system[0])
        kept: list[dict[str, str]] = []
        for msg in reversed(rest):
            t = est(msg)
            if kept and total + t > budget:
                break
            kept.append(msg)
            total += t
        kept.reverse()
        return system + kept

    def _post(self, messages: list[dict[str, str]], allow_end: bool) -> tuple[str, bool]:
        import json
        import os
        import urllib.error
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "truncate_prompt_tokens": max(1, self.context_len - self.max_new_tokens),
        }
        if allow_end:
            body["stop"] = [_END_TOKEN]
        else:
            body["bad_words"] = [_END_TOKEN]
        api_key = os.environ.get(self.api_key_env, "") or "EMPTY"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:500]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"UserLM vLLM {exc.code} {exc.reason} at {self.base_url}: {detail}") from exc
        choice = out["choices"][0]
        text = (choice.get("message", {}).get("content") or "").strip()
        ended = allow_end and (choice.get("stop_reason") == _END_TOKEN or _END_TOKEN in text)
        return _strip_role_prefix(text.replace(_END_TOKEN, "").strip()), ended

    def first_turn(self, task: str, metadata: dict[str, Any]) -> str:
        self._done = False
        self._user_turns = 0
        return self.next_turn(task, [], metadata, _opening=True)

    def next_turn(
        self,
        task: str,
        history: list[Turn],
        metadata: dict[str, Any],  # noqa: ARG002
        _opening: bool = False,
    ) -> str:
        prior_users = sum(1 for t in history if t.get("role") == "user")
        is_first_user_turn = prior_users == 0 or _opening
        if is_first_user_turn:
            self._done = False
            self._user_turns = 0
        messages = self._build_messages(task, history)
        allow_end = not (is_first_user_turn and self.never_end_first_turn)
        intent = task or ""
        text = ""
        ended = False
        for _ in range(_MAX_RESAMPLES):
            candidate, ended = self._post(messages, allow_end=allow_end)
            if ended and not candidate and allow_end:
                self._done = True
                self._user_turns += 1
                return ""
            if _passes_guardrails(candidate, history, intent):
                text = candidate
                self._done = bool(ended and allow_end)
                break
            ended = False
            text = candidate
        if not text:
            text, _ = self._post(messages, allow_end=False)
            self._done = False
        self._user_turns += 1
        if is_first_user_turn:
            self._done = False
        return text

    def is_done(self, history: list[Turn]) -> bool:  # noqa: ARG002
        return self._done
