"""Pairwise context-conditioned OSS hidden-state probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_TEMPLATE = (
    "[History]\n{history}\n\n[Response A]\n{a}\n[Response B]\n{b}\n\n"
    "Which is more likely to have been produced by the person in the conversation?\nAnswer:"
)


def format_probe_input(history: str, a: str, b: str) -> str:
    """Format one pair exactly as presented to the judge model."""
    return _TEMPLATE.format(history=history, a=a, b=b)


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC with average ranks for ties."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


@dataclass(frozen=True)
class PairwiseScore:
    """Both the randomized classifier output and its candidate-oriented value."""

    probability_a_human: float
    probability_candidate_human: float
    candidate_is_a: bool


class PairwiseOSSHiddenStateProbe:
    """Logistic probe over an OSS causal model's hidden state at ``Answer``."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cuda",
        dtype: str = "float16",
        *,
        batch_size: int = 16,
        max_length: int = 1024,
        layer: int = -1,
        seed: int = 0,
        tokenizer=None,
        model=None,
    ) -> None:
        import torch

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self.layer = layer
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = model or AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=getattr(torch, dtype),
                trust_remote_code=True,
            )
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._b = 0.0
        self.train_diagnostics: dict[str, float] | None = None

    def _embed(self, prompts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        torch = self._torch
        chunks: list[np.ndarray] = []
        steps = range(0, len(prompts), self.batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            steps = tqdm(steps, desc="refinement probe embed", unit="batch", leave=False)

        for start in steps:
            batch = list(prompts[start : start + self.batch_size])
            if any(not prompt.endswith("\nAnswer:") for prompt in batch):
                raise ValueError("probe prompts must end with the Answer: marker")
            answer_positions = None
            try:
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                offsets = encoded.pop("offset_mapping")
                positions = []
                for row, prompt in enumerate(batch):
                    answer_start = prompt.rfind("Answer")
                    answer_end = answer_start + len("Answer")
                    overlap = (
                        (offsets[row, :, 0] < answer_end)
                        & (offsets[row, :, 1] > answer_start)
                    ).nonzero(as_tuple=False)
                    if not len(overlap):
                        raise ValueError("tokenizer truncated the Answer marker")
                    answer_ids = encoded["input_ids"][row, overlap[:, 0]].tolist()
                    if "answer" not in self.tokenizer.decode(answer_ids).strip().lower():
                        raise ValueError("failed to locate lexical Answer token span")
                    positions.append(int(overlap[-1, 0]))
                answer_positions = torch.tensor(positions, dtype=torch.long)
            except (KeyError, TypeError, NotImplementedError) as exc:
                raise ValueError(
                    "pairwise probe requires a fast tokenizer with offset mappings "
                    "to pool exactly at the lexical Answer token"
                ) from exc
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            answer_positions = answer_positions.to(self.device)
            with torch.no_grad():
                output = self.model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[self.layer]
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), answer_positions
            ].float()
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            chunks.append(pooled.cpu().numpy())

        hidden_size = int(self.model.config.hidden_size)
        return np.concatenate(chunks) if chunks else np.zeros((0, hidden_size), dtype=np.float32)

    def _training_rows(
        self,
        examples: Sequence[tuple[str, str, str]],
        max_per_class: int | None,
    ) -> tuple[list[str], np.ndarray]:
        """Build balanced, seeded orientations from (history, human, simulator)."""
        shuffled = list(examples)
        self._rng.shuffle(shuffled)
        if max_per_class is not None:
            shuffled = shuffled[: 2 * max_per_class]
        # One randomized orientation per pair; alternate labels before a final
        # shuffle so neither response position is a shortcut.
        labels = np.arange(len(shuffled), dtype=np.int8) % 2
        if bool(self._rng.integers(0, 2)):
            labels = 1 - labels
        prompts = [
            format_probe_input(history, human, simulator)
            if label
            else format_probe_input(history, simulator, human)
            for (history, human, simulator), label in zip(
                shuffled, labels, strict=True
            )
        ]
        order = self._rng.permutation(len(prompts))
        return [prompts[index] for index in order], labels[order]

    def fit(
        self,
        examples: Sequence[tuple[str, str, str]],
        *,
        max_per_class: int | None = None,
        diagnostics: bool = False,
        iterations: int = 200,
        learning_rate: float = 0.1,
        l2: float = 0.01,
    ) -> "PairwiseOSSHiddenStateProbe":
        """Fit from ``(history, human_response, simulator_response)`` triples."""
        if not examples:
            raise ValueError("probe needs at least one training example")
        prompts, labels = self._training_rows(examples, max_per_class)
        features = self._embed(prompts, show_progress=diagnostics)
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std <= 1e-8] = 1.0
        scaled = (features - self._mean) / self._std
        self._w, self._b = _train_logistic(
            scaled,
            labels,
            iterations=iterations,
            learning_rate=learning_rate,
            l2=l2,
        )
        if diagnostics:
            probabilities = _sigmoid(scaled @ self._w + self._b)
            self.train_diagnostics = {
                "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
                "auc": binary_auc(labels, probabilities),
                "examples": float(len(labels)),
            }
        else:
            self.train_diagnostics = None
        return self

    def predict_proba_a(self, history: str, a: str, b: str) -> float:
        """Return the continuous classifier probability that response A is human."""
        self._require_fitted()
        features = self._embed([format_probe_input(history, a, b)])
        scaled = (features - self._mean) / self._std
        return float(_sigmoid(scaled @ self._w + self._b)[0])

    def score_candidate(self, history: str, candidate: str, human: str) -> PairwiseScore:
        """Randomize order and return both P(A human) and de-randomized P(candidate human)."""
        candidate_is_a = bool(self._rng.integers(0, 2))
        a, b = (candidate, human) if candidate_is_a else (human, candidate)
        probability_a = self.predict_proba_a(history, a, b)
        probability_candidate = probability_a if candidate_is_a else 1.0 - probability_a
        return PairwiseScore(probability_a, probability_candidate, candidate_is_a)

    def score_candidates(
        self, examples: Sequence[tuple[str, str, str]]
    ) -> list[PairwiseScore]:
        """Batch score ``(history, candidate, human)`` with randomized positions."""

        self._require_fitted()
        prompts: list[str] = []
        orientations: list[bool] = []
        for history, candidate, human in examples:
            candidate_is_a = bool(self._rng.integers(0, 2))
            a, b = (candidate, human) if candidate_is_a else (human, candidate)
            prompts.append(format_probe_input(history, a, b))
            orientations.append(candidate_is_a)
        features = self._embed(prompts)
        probabilities = _sigmoid(
            ((features - self._mean) / self._std) @ self._w + self._b
        )
        return [
            PairwiseScore(
                float(probability_a),
                float(probability_a if candidate_is_a else 1.0 - probability_a),
                candidate_is_a,
            )
            for probability_a, candidate_is_a in zip(
                probabilities, orientations, strict=True
            )
        ]

    def save(self, path: str | Path) -> None:
        """Save logistic weights and standardization statistics."""
        self._require_fitted()
        np.savez(path, mean=self._mean, std=self._std, weights=self._w, bias=self._b)

    def load(self, path: str | Path) -> "PairwiseOSSHiddenStateProbe":
        """Load logistic weights and standardization statistics."""
        with np.load(path) as data:
            self._mean = data["mean"]
            self._std = data["std"]
            self._w = data["weights"]
            self._b = float(data["bias"])
        return self

    def _require_fitted(self) -> None:
        if self._w is None or self._mean is None or self._std is None:
            raise RuntimeError("probe must be fit or loaded before scoring")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def _train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    x = features.astype(np.float64)
    y = labels.astype(np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(iterations):
        probabilities = _sigmoid(x @ weights + bias)
        error = probabilities - y
        weights -= learning_rate * ((x.T @ error) / len(x) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return weights, bias
