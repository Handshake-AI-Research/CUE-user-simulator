"""Total variation distance over failure-mode mixtures (Figure 4 / Table 3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.fma.paths import distributions_path, tagged_path
from cue_training.evaluation.fma.sim_errors import (
    ENVIRONMENT_ERROR_LABEL,
    PAPER_CUT_MERGE,
    USER_ERROR_LABELS,
)
from cue_training.evaluation.fma.store import load_taxonomy, write_json

# Fixed order for attribution TVD: user / env / agent shares of all failures.
ATTRIBUTION_BUCKETS = ("user_error", "env_error", "agent_error")


def total_variation(p: Iterable[float], q: Iterable[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _counts(rows: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("label") or "Uncategorized").strip() or "Uncategorized"
        c[label] += 1
    return c


def _probs(counts: Counter[str], vocabulary: list[str]) -> list[float]:
    n = sum(counts.get(v, 0) for v in vocabulary)
    if n <= 0:
        return [0.0] * len(vocabulary)
    return [counts.get(v, 0) / n for v in vocabulary]


def _n_in(counts: Counter[str], labels: Iterable[str]) -> int:
    return sum(int(counts.get(name, 0)) for name in labels)


def _attribution_counts(counts: Counter[str]) -> dict[str, int]:
    """Collapse failure labels into user / env / agent attribution buckets."""

    n = int(sum(counts.values()))
    n_user = _n_in(counts, USER_ERROR_LABELS)
    n_env = _n_in(counts, (ENVIRONMENT_ERROR_LABEL,))
    n_agent = max(0, n - n_user - n_env)
    return {"user_error": n_user, "env_error": n_env, "agent_error": n_agent}


def _attribution_probs(counts: Counter[str]) -> list[float]:
    buckets = _attribution_counts(counts)
    n = sum(buckets.values())
    if n <= 0:
        return [0.0] * len(ATTRIBUTION_BUCKETS)
    return [buckets[name] / n for name in ATTRIBUTION_BUCKETS]


def _apply_merge(
    rows: list[dict[str, Any]], merge: dict[str, str]
) -> list[dict[str, Any]]:
    if not merge:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label") or "Uncategorized").strip() or "Uncategorized"
        if label in merge:
            row = dict(row)
            row["label"] = merge[label]
        out.append(row)
    return out


def _filtered_vocab_tvd(
    counts_a: Counter[str],
    counts_b: Counter[str],
    vocabulary: list[str],
    exclude: set[str],
) -> float | None:
    """Multinomial TVD after dropping ``exclude`` labels and renormalizing."""

    vocab = [v for v in vocabulary if v not in exclude]
    if not vocab:
        return None
    if (
        sum(counts_a.get(v, 0) for v in vocab) <= 0
        or sum(counts_b.get(v, 0) for v in vocab) <= 0
    ):
        return None
    return total_variation(_probs(counts_a, vocab), _probs(counts_b, vocab))


def compute_distributions(
    run_dir: Path,
    *,
    human_source: str = "human",
    exclude_labels: Iterable[str] | None = None,
    merge_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    tagged = read_jsonl(str(tagged_path(run_dir)))
    merge = {
        str(k).strip(): str(v).strip()
        for k, v in (merge_labels or {}).items()
        if str(k).strip() and str(v).strip() and str(k).strip() != str(v).strip()
    }
    # Same default paper cut as ``cue-fma plot`` so sim/env rates match the User/Env bars.
    for src, dst in PAPER_CUT_MERGE.items():
        if src not in merge:
            merge[src] = dst
    tagged = _apply_merge(tagged, merge)
    drop = {str(x).strip() for x in (exclude_labels or []) if str(x).strip()}
    drop.add("Uncategorized")
    # Merged-away sources should not stay in the vocabulary.
    drop |= set(merge)
    if drop:
        tagged = [
            r
            for r in tagged
            if str(r.get("label") or "Uncategorized").strip() not in drop
        ]
    taxonomy = load_taxonomy(run_dir)
    vocabulary = [
        m["name"] for m in taxonomy if m.get("name") and m["name"] not in drop
    ]
    # Ensure merge targets stay in the vocabulary even if absent from taxonomy.
    for target in merge.values():
        if target not in drop and target not in vocabulary:
            vocabulary.append(target)
    observed = sorted({str(r.get("label") or "Uncategorized") for r in tagged})
    for name in observed:
        if name not in vocabulary and name not in drop:
            vocabulary.append(name)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in tagged:
        sid = str(row.get("source_id") or "unknown")
        by_source.setdefault(sid, []).append(row)

    human_rows = by_source.get(human_source) or []
    human_counts = _counts(human_rows)
    human_n = len(human_rows)
    human_attr = _attribution_counts(human_counts)
    human_attr_p = _attribution_probs(human_counts)
    human_sim = human_attr["user_error"]
    human_env = human_attr["env_error"]
    human_sim_rate = (human_sim / human_n) if human_n else 0.0
    human_env_rate = (human_env / human_n) if human_n else 0.0

    report: dict[str, Any] = {
        "vocabulary": vocabulary,
        "human_source": human_source,
        "exclude_labels": sorted(drop - set(merge)),
        "merge_labels": merge,
        "human": {
            "n": human_n,
            "counts": dict(human_counts),
            "n_sim_error": human_sim,
            "sim_error_rate": human_sim_rate,
            "n_env_error": human_env,
            "env_error_rate": human_env_rate,
            "attribution": human_attr,
            "attribution_proportions": dict(zip(ATTRIBUTION_BUCKETS, human_attr_p)),
        },
        "comparisons": {},
    }
    for sid, rows in sorted(by_source.items()):
        if sid == human_source:
            continue
        counts = _counts(rows)
        n = len(rows)
        attr = _attribution_counts(counts)
        attr_p = _attribution_probs(counts)
        n_sim = attr["user_error"]
        n_env = attr["env_error"]
        sim_rate = (n_sim / n) if n else 0.0
        env_rate = (n_env / n) if n else 0.0
        report["comparisons"][sid] = {
            "n": n,
            "counts": dict(counts),
            # Proportions are over raw tagged counts for this source_id. When the corpus
            # spans multiple seeds, ingest keeps one source_id per method/sim/arm, so ``n``
            # and ``counts`` are already pooled across seeds.
            "n_sim_error": n_sim,
            "sim_error_rate": sim_rate,
            "n_env_error": n_env,
            "env_error_rate": env_rate,
            "attribution": attr,
            "attribution_proportions": dict(zip(ATTRIBUTION_BUCKETS, attr_p)),
            # (b) TVD over {user_error, env_error, agent_error} vs human.
            "tvd_attribution": (
                total_variation(human_attr_p, attr_p) if human_rows else None
            ),
            # (c) TVD over agent failure modes only (drop user/sim + env; renormalized).
            "tvd_agent": (
                _filtered_vocab_tvd(
                    human_counts,
                    counts,
                    vocabulary,
                    set(USER_ERROR_LABELS) | {ENVIRONMENT_ERROR_LABEL},
                )
                if human_rows
                else None
            ),
            "n_seeds": len({r.get("seed") for r in rows if r.get("seed") is not None})
            or None,
        }
    out = distributions_path(run_dir)
    # Keep distributions.json as the paper-cut (plot-aligned) report. Only suffix the
    # path when the caller adds merges/excludes beyond that default.
    paper_cut = merge == PAPER_CUT_MERGE and drop - set(merge) <= {"Uncategorized"}
    if not paper_cut:
        parts: list[str] = []
        if merge:
            parts.append("merged")
        if drop - set(merge):
            parts.append(
                "excl_"
                + "_".join(
                    "".join(ch if ch.isalnum() else "_" for ch in lab)[:24]
                    for lab in sorted(drop - set(merge))
                )
            )
        if parts:
            out = run_dir / f"distributions_{'_'.join(parts)}.json"
    write_json(out, report)
    report["path"] = str(out)
    return report
