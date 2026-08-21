import json

from cue_training.preprocessing.raw import process_dialoguestudio, process_lmsys
from cue_training.preprocessing.raw.process_locomo import convert_sample
from cue_training.preprocessing.raw.process_ms_tod import convert_dataset as convert_ms_tod
from cue_training.preprocessing.raw.utils import write_dataset_split


def test_dialogstudio_conversion_is_canonical():
    record = process_dialoguestudio.convert_dialog(
        {
            "new dialog id": "d1",
            "prompt": ["Book travel", "Be helpful"],
            "log": [
                {"user utterance": "Paris, please", "system response": "What date?"},
                {"user utterance": "Tomorrow", "system response": "Booked."},
            ],
        },
        "MultiWOZ",
        split="validation",
    )
    assert record is not None
    assert (record["id"], record["session_id"], record["split"]) == (
        "d1",
        "d1",
        "validation",
    )
    assert [turn["role"] for turn in record["turns"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(turn["turn_id"].startswith("d1:") for turn in record["turns"])


def test_wildchat_timestamp_metadata_is_json_writable():
    """WildChat yields a datetime for `timestamp`, which no JSONL writer can encode."""

    from datetime import datetime, timezone

    from cue_training.preprocessing.raw import process_wildchat

    records = process_wildchat.convert_rows(
        [
            {
                "conversation_hash": "abc",
                "timestamp": datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc),
                "conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        ]
    )
    assert records[0]["metadata"]["timestamp"] == "2024-05-01T12:30:00+00:00"
    json.dumps(records[0])


def test_lmsys_conversion_handles_alias_roles_and_filters():
    records = process_lmsys.convert_rows(
        [
            {
                "conversation_id": "c1",
                "language": "en",
                "conversation": [
                    {"role": "system", "content": "Answer briefly."},
                    {"role": "human", "content": [{"text": "Help"}]},
                    {"role": "gpt", "content": "Sure."},
                ],
            },
            {
                "conversation_id": "c2",
                "language": "French",
                "conversation": [{"role": "user", "content": "Bonjour"}],
            },
        ],
        english_only=True,
    )
    assert [record["id"] for record in records] == ["c1"]
    assert [turn["role"] for turn in records[0]["turns"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_ms_tod_flattens_sessions_and_preserves_persona_metadata():
    records = convert_ms_tod(
        {
            "personas": [
                {
                    "persona_id": 0,
                    "sessions": [
                        {
                            "session_id": "s1",
                            "service": "hotel",
                            "intent": "book",
                            "turns": [
                                {"speaker": "user", "utterance": "Need a room"},
                                {"speaker": "assistant", "utterance": "For when?"},
                            ],
                        }
                    ],
                }
            ]
        }
    )
    assert len(records) == 1
    assert records[0]["metadata"]["persona_id"] == "0"
    assert records[0]["metadata"]["service"] == "hotel"
    assert [turn["role"] for turn in records[0]["turns"]] == ["user", "assistant"]


def test_locomo_creates_one_session_record_per_speaker():
    records = convert_sample(
        {
            "sample_id": "sample",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {"speaker": "Bob", "text": "How are you?"},
                    {"speaker": "Alice", "text": "Busy."},
                    {"speaker": "Bob", "text": "Same."},
                ],
            },
        },
        0,
    )
    assert {record["metadata"]["speaker"] for record in records} == {"Alice", "Bob"}
    alice = next(record for record in records if record["metadata"]["speaker"] == "Alice")
    assert [turn["role"] for turn in alice["turns"]] == ["assistant", "user"]
    assert alice["metadata"]["user_id"] == "sample:Alice"


def test_dataset_writer_creates_jsonl_and_streaming_index(tmp_path):
    records = process_lmsys.convert_rows(
        [
            {
                "conversation_id": "c1",
                "conversation": [{"role": "user", "content": "hello"}],
            }
        ]
    )
    path = write_dataset_split(records, tmp_path, "lmsys-chat-1m", "train")
    assert json.loads(path.read_text().strip())["id"] == "c1"
    index = json.loads((tmp_path / "streaming_index.json").read_text())
    assert index["files"]["lmsys-chat-1m/train.jsonl"]["count"] == 1
