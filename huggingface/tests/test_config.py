"""CueConfig defaults and mapping from cue_training's unified_config.json."""

from __future__ import annotations

from cue_hf import CueConfig


def test_defaults_match_final_cue_training():
    config = CueConfig()
    assert config.encoder["user_model_name"] == "nomic-ai/modernbert-embed-base"
    assert config.encoder["system_model_name"] == "nomic-ai/modernbert-embed-base"
    assert config.decoder["model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert config.decoder["insert_every"] == 4
    assert (config.dual_decode, config.command_slot_decode) == (True, True)
    assert (
        config.general_command_slots,
        config.user_specific_command_slots,
        config.style_command_slots,
    ) == (5, 5, 5)


def test_decode_defaults_match_rollout_config():
    config = CueConfig()
    assert config.decode_temperature == 0.0  # greedy
    assert config.slot_dedup_jaccard == 0.8
    assert config.example_retrieval_k_sessions == 8
    assert config.example_retrieval_n_general == 2
    assert config.example_retrieval_n_specific == 6


def test_from_cue_config_prefers_model_block_and_fills_from_train():
    config = CueConfig.from_cue_config(
        {
            "train": {
                "dual_decode": True,
                "encoder": {"user_model_name": "stale/name", "session_layers": 4},
                "decoder": {"model_name": "Qwen/Qwen3-0.6B-Base", "dtype": "bfloat16"},
            },
            "model": {
                "encoder": {
                    "user_model_name": "nomic-ai/modernbert-embed-base",
                    "query_prefix": "query: ",
                },
                "decoder": {"model_name": "Qwen/Qwen3-0.6B-Base", "insert_every": 4},
                "command_slot_decode": True,
                "style_command_slots": 4,
            },
        }
    )
    assert config.encoder["user_model_name"] == "nomic-ai/modernbert-embed-base"
    assert config.encoder["session_layers"] == 4
    assert config.encoder["query_prefix"] == "query: "
    # dtype only exists in the train block but is needed to rebuild the decoder.
    assert config.decoder["dtype"] == "bfloat16"
    assert config.command_slot_decode is True
    assert config.style_command_slots == 4


def test_config_json_roundtrip():
    config = CueConfig(session_preprocess="strip_document", example_pool_id="org/pool")
    restored = CueConfig.from_dict(config.to_dict())
    assert restored.session_preprocess == "strip_document"
    assert restored.example_pool_id == "org/pool"
    assert restored.encoder == config.encoder


def test_null_session_preprocess_falls_back_to_full():
    """Exports that omit the mode write null, which must not reach encode as "None"."""

    assert CueConfig.from_dict({"session_preprocess": None}).session_preprocess == "full"
