import json

import pytest
import torch

from cue_training.infer import export


@pytest.mark.parametrize("conversation_key", ["real_conversation", "conversation", "turns"])
def test_normalize_export_record_conversation_aliases(conversation_key):
    record = export.normalize_export_record(
        {
            "episode_id": "episode-1",
            conversation_key: [{"role": "user", "content": "hello"}],
            "metadata": {"domain": "test"},
        }
    )
    assert record["session_id"] == "episode-1"
    assert record["turns"][0]["content"] == "hello"
    assert record["metadata"] == {"domain": "test"}


def test_export_embeddings_writes_harness_rows(tmp_path, monkeypatch):
    class FakeModel:
        def encode(self, sessions):
            return torch.tensor([[len(session), index] for index, session in enumerate(sessions)])

    monkeypatch.setattr(export, "load_checkpoint", lambda checkpoint, device: FakeModel())
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "s1",
                        "real_conversation": [{"role": "user", "content": "one"}],
                        "metadata": {"arm": "a"},
                    }
                ),
                json.dumps(
                    {
                        "id": "s2",
                        "conversation": [
                            {"role": "assistant", "content": "hi"},
                            {"role": "user", "content": "two"},
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "embeddings.jsonl"

    assert export.export_embeddings("checkpoint", source, output, batch_size=2) == output

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "trajectory_embedding": [1.0, 0.0],
            "session_id": "s1",
            "metadata": {"arm": "a"},
        },
        {
            "trajectory_embedding": [2.0, 1.0],
            "session_id": "s2",
            "metadata": {},
        },
    ]


def test_export_embedding_records_writes_in_memory_rows(tmp_path, monkeypatch):
    class FakeModel:
        def encode(self, sessions):
            return torch.tensor([[float(len(session)), 2.0] for session in sessions])

    monkeypatch.setattr(export, "load_checkpoint", lambda checkpoint, device: FakeModel())
    output = tmp_path / "embeddings.jsonl"

    export.export_embedding_records(
        "checkpoint",
        [
            {
                "episode_id": "paired-1",
                "real_conversation": [{"role": "user", "content": "hello"}],
                "metadata": {"domain": "test"},
            }
        ],
        output,
        batch_size=4,
    )

    assert json.loads(output.read_text()) == {
        "trajectory_embedding": [1.0, 2.0],
        "session_id": "paired-1",
        "metadata": {"domain": "test"},
    }


def test_iter_export_records_reports_bad_line(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"session_id": "ok", "turns": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"input\.jsonl:1"):
        list(export.iter_export_records(source))
