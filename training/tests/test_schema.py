from cue_training.data.schema import (
    canonicalize_manual,
    canonicalize_profile,
    filter_joint_training_target,
    merge_dual_manual_blocks,
    parse_manual,
    split_manual_targets,
    to_profile,
    validate_manual,
    validate_record,
)


def test_validate_manual_kinds_and_forms():
    manual = validate_manual(
        {
            "commands": [
                {"text": "Be terse.", "kind": "sim_contrast", "form": "general", "examples": ["ok"]},
                {
                    "text": "If the agent apologizes, restate the ask.",
                    "kind": "human_contrast",
                    "form": "if_then",
                    "examples": ["just fix it"],
                },
            ]
        }
    )
    assert manual["schema_version"] == 1
    assert manual["commands"][0]["kind"] == "sim_contrast"
    assert manual["commands"][1]["form"] == "if_then"
    text = canonicalize_manual(manual)
    assert parse_manual(text) is not None


def test_canonicalize_profile_drops_evidence_and_kinds():
    profile = to_profile(
        {
            "commands": [
                {
                    "text": "Be terse.",
                    "kind": "sim_contrast",
                    "form": "general",
                    "examples": ["ok", "fix it"],
                    "evidence_turn_ids": ["ABCD--train--1:user:0"],
                },
                {
                    "text": "If the agent apologizes, restate the ask.",
                    "kind": "human_contrast",
                    "form": "if_then",
                    "examples": ["just fix it"],
                    "evidence_turn_ids": ["ABCD--train--1:user:1"],
                },
            ]
        }
    )
    assert profile == {
        "commands": ["Be terse.", "If the agent apologizes, restate the ask."],
        "examples": ["ok", "fix it", "just fix it"],
    }
    text = canonicalize_profile(profile)
    assert "evidence_turn_ids" not in text
    assert "sim_contrast" not in text
    assert "demographics" not in text
    parsed = parse_manual(text)
    assert parsed is not None
    assert parsed["commands"] == profile["commands"]
    assert parsed["examples"] == profile["examples"]
    assert "demographics" not in parsed


def test_split_manual_targets_uses_annotation_command_kinds():
    general, specific, style = split_manual_targets(
        {
            "commands": [
                {"text": "Be terse and skip pleasantries.", "kind": "sim_contrast", "examples": ["ok"]},
                {"text": "Use lowercase and few commas.", "kind": "human_contrast", "examples": ["fix it"]},
                {
                    "text": "If the agent apologizes, restate the ask.",
                    "kind": "human_contrast",
                    "form": "if_then",
                    "examples": ["just fix it"],
                },
                {
                    "text": "Prefer lowercase fragments without end punctuation.",
                    "kind": "style",
                    "form": "general",
                    "examples": ["ok now"],
                },
            ]
        }
    )
    g = parse_manual(general)
    s = parse_manual(specific)
    st = parse_manual(style)
    assert g is not None and s is not None and st is not None
    assert g["commands"] == ["Be terse and skip pleasantries."]
    assert "Use lowercase and few commas." in s["commands"]
    assert st["commands"] == ["Prefer lowercase fragments without end punctuation."]


def test_merge_dual_manual_blocks():
    merged = merge_dual_manual_blocks(
        {"commands": ["a"], "examples": []},
        {"commands": ["b"], "examples": ["ex"]},
        {"commands": ["c"], "examples": []},
    )
    assert merged["commands"] == ["a", "b", "c"]
    assert merged["examples"] == ["ex"]


def test_filter_joint_training_target_keeps_utterances_not_ask_directives():
    filtered = filter_joint_training_target(
        {
            "commands": [
                {
                    "text": "Ask for the next missing detail.",
                    "kind": "sim_contrast",
                    "examples": ["Which one?"],
                },
                {
                    "text": "If the assistant asks for a detail, answer briefly.",
                    "kind": "human_contrast",
                    "examples": [
                        "the cheapest",
                        'if asked to choose, reply "A"',
                        "assistant asks <Q1>, reply with <ENTITY>",
                        "this example is much too long to be useful as a short natural user utterance",
                    ],
                },
            ]
        }
    )
    assert [command["text"] for command in filtered["commands"]] == [
        "If the assistant asks for a detail, answer briefly."
    ]
    assert filtered["commands"][0]["examples"] == ["the cheapest", "A"]
    general, specific, style = split_manual_targets(filtered)
    assert parse_manual(general)["commands"]
    assert parse_manual(specific)["commands"]
    assert parse_manual(style)["commands"] == []


def test_profile_examples_are_unique_and_capped_at_four():
    profile = to_profile(
        {
            "commands": ["Be terse."],
            "examples": ["One", "one", "Two", "Three", "Four", "Five"],
        }
    )
    assert profile["examples"] == ["One", "Two", "Three", "Four"]


def test_validate_record_role_content():
    rec = validate_record(
        {
            "id": "s1",
            "dataset": "toy",
            "turns": [
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "help"},
            ],
            "persona_manual": {"commands": [{"text": "Be terse.", "examples": []}]},
        }
    )
    assert rec["id"] == "s1"
    assert any(t["role"] == "user" for t in rec["turns"])


def test_validate_record_cue_style_turns():
    rec = validate_record(
        {
            "session_id": "s2",
            "turns": [
                {"user_text": "fix it", "system_text": "sure"},
            ],
        }
    )
    roles = [t["role"] for t in rec["turns"]]
    assert "user" in roles
    assert "assistant" in roles
