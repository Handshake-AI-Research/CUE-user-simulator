"""Tests for annotation-delta export / rehydrate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cue_training.data.annotation_release import (
    build_delta_row,
    export_delta_rows,
    export_delta_rows_multi,
    group_delta_by_dataset_split,
    load_delta_jsonl,
    match_source_record,
    rehydrate_delta_rows,
    rehydrate_record,
    resolve_source,
    load_source_records,
    source_native_id,
    validate_delta_row,
    write_delta_jsonl,
    write_rehydrated_dataset,
)
from cue_training.data.schema import validate_record
from cue_training.data.streaming import RoundRobinStreamingDataset, write_jsonl_atomic
from cue_training.preprocessing.raw import process_dialoguestudio


def _manual():
    return {
        "schema_version": 1,
        "commands": [
            {
                "text": f"Rule {i}",
                "kind": "sim_contrast" if i < 5 else "human_contrast",
                "form": "general",
                "examples": ["say hi"] if i == 0 else [],
                "evidence_turn_ids": [f"s1:user:{i}"] if i < 2 else [],
            }
            for i in range(10)
        ],
    }


def _annotated(dataset: str, sid: str, text: str, *, split: str = "train") -> dict:
    return validate_record(
        {
            "id": sid,
            "dataset": dataset,
            "split": split,
            "turns": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "ok"},
            ],
            "persona_manual": _manual(),
            "provenance": {
                "proposer": "gpt-test",
                "sim_models": ["m1"],
                "human_negative_ids": ["secret-id"],
                "gate_notes": ["quoted user text should not ship"],
                "max_user_turns": 15,
            },
            "metadata": {"language": "english"},
        }
    )


def test_resolve_source_registry():
    assert resolve_source("MultiWOZ_2.1").repo == "Salesforce/dialogstudio"
    assert resolve_source("MultiWOZ_2.1").config == "MultiWOZ_2.1"
    assert resolve_source("lmsys-chat-1m").kind == "lmsys"
    assert resolve_source("wildchat").kind == "wildchat"
    # Anything else (except LMSYS/WildChat) maps to DialogStudio by config name.
    other = resolve_source("SomeDialogStudioConfig")
    assert other.repo == "Salesforce/dialogstudio"
    assert other.config == "SomeDialogStudioConfig"
    assert other.kind == "dialogstudio"


def test_dataset_aliases_canonicalize():
    assert resolve_source("lmsys").kind == "lmsys"
    assert resolve_source("wildchat-1m").kind == "wildchat"
    assert resolve_source("multiwoz_2.1").config == "MultiWOZ_2.1"
    assert resolve_source("multiwoz2_2").config == "MULTIWOZ2_2"
    row = build_delta_row(_annotated("lmsys", "abc", "hi"))
    assert row["dataset"] == "lmsys-chat-1m"
    assert row["source_repo"] == "lmsys/lmsys-chat-1m"
    ds_row = build_delta_row(_annotated("ABCD", "d1", "hi"))
    assert ds_row["source_repo"] == "Salesforce/dialogstudio"
    assert ds_row["source_config"] == "ABCD"


def test_build_delta_row_strips_conversation_and_sensitive_provenance():
    row = build_delta_row(_annotated("ABCD", "dlg-1", "book a flight"))
    assert "turns" not in row
    assert row["source_repo"] == "Salesforce/dialogstudio"
    assert row["source_config"] == "ABCD"
    assert row["source_id"] == "dlg-1"
    assert row["persona_manual"]["commands"][0]["text"] == "Rule 0"
    assert "human_negative_ids" not in row["provenance"]
    assert "gate_notes" not in row["provenance"]
    assert row["provenance"]["proposer"] == "gpt-test"
    validate_delta_row(row)


def test_wildchat_native_id():
    assert source_native_id("wildchat", "abc123-42") == "abc123"
    assert source_native_id("lmsys-chat-1m", "conv-9") == "conv-9"


def test_export_and_rehydrate_roundtrip(tmp_path: Path):
    ann = tmp_path / "ann"
    write_jsonl_atomic(
        ann / "ABCD" / "train.jsonl",
        [_annotated("ABCD", "a1", "hello a"), _annotated("ABCD", "a2", "hello b")],
    )
    write_jsonl_atomic(
        ann / "ABCD" / "validation.jsonl",
        [_annotated("ABCD", "a1v", "hello val", split="validation")],
    )
    write_jsonl_atomic(
        ann / "lmsys-chat-1m" / "train.jsonl",
        [_annotated("lmsys-chat-1m", "c1", "hello c")],
    )
    rows = export_delta_rows_multi({"train": ann, "validation": ann})
    assert len(rows) == 4
    grouped = group_delta_by_dataset_split(rows)
    assert set(grouped["ABCD"]) == {"train", "validation"}
    assert grouped["ABCD"]["train"] and grouped["ABCD"]["validation"]
    assert set(grouped["lmsys-chat-1m"]) == {"train"}

    delta_path = tmp_path / "delta.jsonl"
    write_delta_jsonl(delta_path, rows)
    loaded = load_delta_jsonl(delta_path)
    assert len(loaded) == 4

    sources = {
        "a1": _annotated("ABCD", "a1", "hello a"),
        "a2": _annotated("ABCD", "a2", "hello b"),
        "a1v": _annotated("ABCD", "a1v", "hello val", split="validation"),
        "c1": _annotated("lmsys-chat-1m", "c1", "hello c"),
    }
    # Drop manuals from "source" side to prove rehydrate attaches them.
    for rec in sources.values():
        rec.pop("persona_manual", None)

    def fake_loader(spec, *, revision=None, max_records=None, target_rows=None):
        if spec.kind == "dialogstudio":
            wanted = {str(r["source_id"]) for r in (target_rows or [])}
            return [sources[k] for k in ("a1", "a2", "a1v") if k in wanted or not wanted]
        return [sources["c1"]]

    out, errors = rehydrate_delta_rows(loaded, source_loader=fake_loader)
    assert not errors
    assert len(out) == 4
    assert all(r.get("persona_manual") for r in out)
    assert all(r["provenance"].get("rehydrated") for r in out)

    out_root = tmp_path / "rehydrated"
    written = write_rehydrated_dataset(out, out_root)
    assert {p.name for p in written} == {"train.jsonl", "validation.jsonl"}
    assert (out_root / "ABCD" / "validation.jsonl").exists()

    ds = RoundRobinStreamingDataset(out_root, split="train", require_manual=True, shuffle_buffer=0)
    ids = {r["id"] for r in ds}
    assert ids == {"a1", "a2", "c1"}


def test_push_delta_builds_per_dataset_train_validation(monkeypatch):
    pushed: list[tuple[str, set[str]]] = []

    class FakeDatasetDict(dict):
        def push_to_hub(self, repo_id, *, config_name=None, private=True, token=None, revision=None):
            pushed.append((config_name, set(self.keys())))

    def fake_from_splits(splits):
        return FakeDatasetDict({name: list(items) for name, items in splits.items() if items})

    monkeypatch.setattr(
        "cue_training.data.annotation_release._dataset_dict_from_splits",
        fake_from_splits,
    )
    from cue_training.data.annotation_release import push_delta_to_hub

    rows = [
        build_delta_row(_annotated("ABCD", "a1", "t1")),
        build_delta_row(_annotated("ABCD", "a2", "t2", split="validation")),
        build_delta_row(_annotated("wildchat", "w1", "t3")),
    ]
    push_delta_to_hub(rows, "org/cue-delta", private=True)
    assert ("default", {"train", "validation"}) in pushed
    assert ("ABCD", {"train", "validation"}) in pushed
    assert ("wildchat", {"train"}) in pushed


def test_match_falls_back_to_native_id():
    """WildChat ids carry a scan-position suffix, so the bare hash is the stable key."""

    source = validate_record(
        {
            "id": "wildchat-hash-99",
            "dataset": "wildchat",
            "turns": [
                {"role": "user", "content": "same text"},
                {"role": "assistant", "content": "ok"},
            ],
        }
    )
    delta = build_delta_row(_annotated("wildchat", "wildchat-hash-1", "same text"))
    matched = match_source_record(delta, by_id={}, by_native={"wildchat-hash": [source]})
    assert matched["id"] == "wildchat-hash-99"
    joined = rehydrate_record(delta, matched)
    assert joined["id"] == delta["source_id"]
    assert joined["persona_manual"]


def test_match_keeps_record_whose_text_drifted():
    """The join is on ids, so differing upstream text is not a reason to drop a row."""

    source = validate_record(
        {"id": "a1", "dataset": "ABCD", "turns": [{"role": "user", "content": "upstream text"}]}
    )
    delta = build_delta_row(_annotated("ABCD", "a1", "annotated text"))
    matched = match_source_record(delta, by_id={"a1": source}, by_native={})
    assert matched["id"] == "a1"
    joined = rehydrate_record(delta, matched)
    assert joined["persona_manual"]
    assert joined["turns"][0]["content"] == "upstream text"


def test_source_loader_matches_by_id_and_exits_early(monkeypatch):
    """Rows whose text drifted are still found by id, and the scan stops there."""

    consumed: list[str] = []
    source_rows = [
        {
            "conversation_id": f"c{i}",
            "conversation": [{"role": "user", "content": f"drifted {i}"}],
        }
        for i in range(6)
    ]

    def fake_load_dataset(*args, **kwargs):
        def stream():
            for row in source_rows:
                consumed.append(row["conversation_id"])
                yield row

        return stream()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    targets = [
        build_delta_row(_annotated("lmsys-chat-1m", "c0", "original 0")),
        build_delta_row(_annotated("lmsys-chat-1m", "c1", "original 1")),
    ]
    records = load_source_records(resolve_source("lmsys-chat-1m"), target_rows=targets)
    assert [record["id"] for record in records] == ["c0", "c1"]
    assert consumed == ["c0", "c1"]


def test_dialogstudio_rows_come_from_json_not_the_removed_script(monkeypatch, tmp_path: Path):
    """DialogStudio ships a loading script and no parquet, so rows are read from JSON."""

    import huggingface_hub

    from cue_training.data.annotation_release import iter_dialogstudio_rows

    files = {
        "task_oriented/SGD/train/dialogues_2.json": {
            "SGD--train--2": {"original dialog id": "o2", "dialog index": 2, "prompt": ["p"],
                              "log": {"turn id": "1", "user utterance": "hi", "system response": "yo"}},
        },
        "task_oriented/SGD/train/dialogues_10.json": {
            "SGD--train--10": {"original dialog id": "o10", "dialog index": 10, "prompt": ["p"],
                               "log": [{"turn id": "1", "user utterance": "hey", "system response": "sup"}]},
        },
    }
    for name, payload in files.items():
        path = tmp_path / name.replace("/", "_")
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda *a, **k: [*files, "task_oriented/SGD/val/dialogues_1.json"],
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, name, **k: str(tmp_path / name.replace("/", "_")),
    )
    rows = list(iter_dialogstudio_rows(resolve_source("SGD", split="train")))
    # dialogues_2 before dialogues_10: numeric order, not lexicographic.
    assert [row["new dialog id"] for row in rows] == ["SGD--train--2", "SGD--train--10"]
    # A single-turn dialog is stored as a dict and must still become a list.
    assert rows[0]["log"] == [{"turn id": "1", "user utterance": "hi", "system response": "yo"}]

    record = validate_record(
        process_dialoguestudio.convert_dialog(rows[0], "SGD", split="train", index=0)
    )
    assert record["id"] == "SGD--train--2"
    assert [turn["role"] for turn in record["turns"]] == ["system", "user", "assistant"]


def test_coverage_report_flags_missing():
    from cue_training.data.annotation_release import coverage_report

    rows = [
        build_delta_row(_annotated("ABCD", "a1", "hi")),
        build_delta_row(_annotated("ABCD", "a2", "there")),
    ]
    hydrated = rehydrate_record(
        rows[0],
        validate_record({"id": "a1", "dataset": "ABCD", "turns": [{"role": "user", "content": "drift"}]}),
    )
    report = coverage_report(rows, [hydrated], [{"source_id": "a2", "error": "no source match"}])
    assert report["sources"]["ABCD"]["coverage"] == 0.5
    assert report["sources"]["ABCD"]["missing_ids"] == ["a2"]


def test_load_delta_from_hub_skips_mixed_configs(monkeypatch):
    """'default' and 'all' mirror the per-dataset configs; loading them doubles every row."""

    import datasets as hf_datasets

    from cue_training.data.annotation_release import load_delta_from_hub

    rows = {"ABCD": [build_delta_row(_annotated("ABCD", "a1", "hi"))]}
    rows["all"] = rows["ABCD"]
    rows["default"] = rows["ABCD"]
    loaded: list[str] = []

    monkeypatch.setattr(hf_datasets, "get_dataset_config_names", lambda *a, **k: ["ABCD", "all", "default"])
    monkeypatch.setattr(hf_datasets, "get_dataset_split_names", lambda *a, **k: ["train"])

    def fake_load_dataset(repo_id, name, *, split=None, **kwargs):
        loaded.append(name)
        return rows[name]

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    out = load_delta_from_hub("org/delta")
    assert loaded == ["ABCD"]
    assert len(out) == 1


def test_pull_fails_closed_below_coverage(monkeypatch, tmp_path: Path):
    from cue_training.preprocessing import hydrate_annotations as module

    rows = [
        build_delta_row(_annotated("ABCD", "a1", "hi")),
        build_delta_row(_annotated("ABCD", "a2", "there")),
    ]
    hydrated = rehydrate_record(
        rows[0],
        validate_record({"id": "a1", "dataset": "ABCD", "turns": [{"role": "user", "content": "hi"}]}),
    )
    monkeypatch.setattr(module, "load_delta_from_hub", lambda *a, **k: rows)
    monkeypatch.setattr(module, "rehydrate_delta_rows", lambda *a, **k: ([hydrated], []))

    out_root = tmp_path / "out"
    with pytest.raises(ValueError, match="coverage below"):
        module.hydrate_annotations(repo_id="org/delta", out_root=out_root, min_coverage=0.95)
    assert not (out_root / "ABCD").exists()
    audit = json.loads((out_root / "hydration_audit.json").read_text())
    assert audit["sources"]["ABCD"]["missing_ids"] == ["a2"]

    module.hydrate_annotations(repo_id="org/delta", out_root=out_root, min_coverage=0.4)
    assert (out_root / "ABCD" / "train.jsonl").exists()


def test_pull_resumes_and_only_hydrates_missing(monkeypatch, tmp_path: Path):
    from cue_training.preprocessing import hydrate_annotations as module

    rows = [
        build_delta_row(_annotated("ABCD", "a1", "hi")),
        build_delta_row(_annotated("ABCD", "a2", "there")),
    ]
    sources = {
        "a1": validate_record({"id": "a1", "dataset": "ABCD", "turns": [{"role": "user", "content": "hi"}]}),
        "a2": validate_record({"id": "a2", "dataset": "ABCD", "turns": [{"role": "user", "content": "there"}]}),
    }
    asked: list[list[str]] = []

    def fake_rehydrate(pending, **kwargs):
        asked.append([str(row["source_id"]) for row in pending])
        return [rehydrate_record(row, sources[str(row["source_id"])]) for row in pending], []

    monkeypatch.setattr(module, "load_delta_from_hub", lambda *a, **k: rows)
    monkeypatch.setattr(module, "rehydrate_delta_rows", fake_rehydrate)

    out_root = tmp_path / "out"
    module.hydrate_annotations(repo_id="org/delta", out_root=out_root)
    assert asked == [["a1", "a2"]]

    # Second pull sees both on disk, hydrates nothing, and leaves the file intact.
    module.hydrate_annotations(repo_id="org/delta", out_root=out_root)
    assert asked == [["a1", "a2"]]
    written = (out_root / "ABCD" / "train.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["id"] for line in written] == ["a1", "a2"]

    # Dropping one record makes the next pull fetch only that one, and append it back.
    (out_root / "ABCD" / "train.jsonl").write_text(written[0] + "\n", encoding="utf-8")
    module.hydrate_annotations(repo_id="org/delta", out_root=out_root)
    assert asked[-1] == ["a2"]
    ids = [json.loads(line)["id"] for line in (out_root / "ABCD" / "train.jsonl").read_text().strip().splitlines()]
    assert sorted(ids) == ["a1", "a2"]


def test_audit_reports_first_difference(tmp_path: Path):
    from cue_training.data.annotation_release import audit_annotated_records

    ann = tmp_path / "ann"
    write_jsonl_atomic(ann / "ABCD" / "train.jsonl", [_annotated("ABCD", "a1", "annotated text")])
    source = validate_record(
        {"id": "a1", "dataset": "ABCD", "turns": [
            {"role": "user", "content": "annotated txt"},
            {"role": "assistant", "content": "ok"},
        ]}
    )
    report = audit_annotated_records(
        ann,
        source_loader=lambda spec, **kwargs: [source],
    )
    diff = report["ABCD"]["differences"][0]
    assert report["ABCD"]["found"] == 1
    assert report["ABCD"]["identical"] == 0
    assert diff["kind"] == "content"
    assert diff["annotated"].startswith("e")


def test_data_annotation_loader_accepts_rehydrated_records(tmp_path: Path):
    """data annotation / training JSONL discovery accepts hydrate output."""

    from cue_training.data.streaming import discover_jsonl_files, iter_jsonl

    rec = rehydrate_record(
        build_delta_row(_annotated("ABCD", "a1", "hi")),
        validate_record(
            {
                "id": "a1",
                "dataset": "ABCD",
                "turns": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
        ),
    )
    root = tmp_path / "data"
    write_jsonl_atomic(root / "ABCD" / "train.jsonl", [rec])
    files = discover_jsonl_files(root, split="train")
    assert len(files) == 1
    loaded = list(iter_jsonl(files[0], validate=True))
    assert loaded[0]["persona_manual"]
    assert loaded[0]["metadata"]["rehydrated_from"]["source_id"] == "a1"
