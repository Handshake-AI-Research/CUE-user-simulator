# CUE: Calibrated User Embeddings for realistic User Simulation

[![CI](https://github.com/Handshake-AI-Research/CUE-user-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Handshake-AI-Research/CUE-user-simulator/actions/workflows/ci.yml)

**CUE** is a user-simulator framework that can utilize any LLM to generate messages as realistic users. Based on a user's dialogue history, CUE produces a **persona manual**: a short list of behavioral commands ("keep requests terse", "ask about price before committing") that can be added to the system prompt of any LLM so it role-plays that user. Underneath the hood, a chat session from user is encoded into a single 1024-d CUE embedding, and a decoder turns that embedding back into text commands. We call this *user-conditioned CUE*.

You can alternatively use CUE to sample a persona manual representing a *novel* user without any existing user dialogue.

This repo contains a production-ready inference runtime to produce text persona manuals using an already-trained CUE model. 
Python import name: `cue`. 
Load a published checkpoint from Hugging Face, build a persona manual from
a dialogue **or** sample one from the prior, and steer **any** chat model as a
user simulator.

| you want | use |
|----------|-----|
| weights + remote code | [handshake-ai-research/cue](https://huggingface.co/handshake-ai-research/cue) |
| research / training | [AnjaliRuban/CUE](https://github.com/AnjaliRuban/CUE) |
| Transformers `AutoModel` plumbing | [AnjaliRuban/cue-hf](https://github.com/AnjaliRuban/cue-hf) |

## Install

```bash
uv tool install cue-simulator
cue --help
# or
cue-simulator --help

# or as a library (import name is still `cue`)
uv add cue-simulator
```

Before the first release, install from a clone with `uv sync`. Optional extras
are `cue-simulator[openai]` for OpenAI-compatible endpoints and
`cue-simulator[retrieval]` for example-pool retrieval.

`Cue.from_pretrained` loads the code bundled in the model repository with
`AutoModel.from_pretrained(..., trust_remote_code=True)`. If `cue-hf` is already
installed, it uses that package directly.

Private Hub weights need `HF_TOKEN` (or `huggingface-cli login`).

## Quickstart

The repository ships a runnable config under `examples/quickstart/`:

```bash
# 1. Install
uv sync

# 2. Provide access to the private model (not needed once it is public)
export HF_TOKEN=hf_...

# 3. Run from the repository root
uv run cue --config examples/quickstart/cue.toml

# 4. Inspect the stable application outputs
cat examples/quickstart/output/steering_prompts.txt
cat examples/quickstart/output/manuals.json
```

Change `mode = "conditioned"` to `mode = "sampled"` to draw a user from the
diffusion prior instead. Add the three `simulator_*` fields shown in the example
to also execute one user turn against any configured base simulator.

## Configuration

`cue --config` is the recommended production interface.

| field | required | default | description |
|-------|----------|---------|-------------|
| `mode` | yes | — | `conditioned` or `sampled` |
| `output_dir` | yes | — | Output artifact directory |
| `model` | no | `handshake-ai-research/cue` | Hugging Face repo or local model path |
| `device` | no | `cpu` | Torch device (`cpu`, `cuda`, `mps`) |
| `conversation_path` | conditioned only | — | JSON, JSONL, or role-prefixed transcript |
| `session_preprocess` | no | `full` | `full`, `strip_document`, or `user_only` |
| `seed` / `n` | no | `0` / `1` | Sampling seed and number of users |
| `example_pool` | no | — | Separate Hub dataset for style retrieval |
| `example_retrieval` | no | `false` | Retrieve style samples |
| `simulator_backend` | no | — | `openai` or `hf` |
| `simulator_model` | with backend | — | Any model identifier accepted by that backend |
| `scenario` | with backend | — | Goal for the role-played user |
| `history_path` | no | — | Existing simulator conversation |
| `api_key_env` / `base_url` | no | — | OpenAI-compatible endpoint configuration |

Paths are resolved from the invocation's working directory, matching the
config convention used by other Handshake runtime packages.

### Python API

### 1. User-conditioned CUE (from a conversation)

```python
from cue import Cue

cue = Cue.from_pretrained("handshake-ai-research/cue", device="cuda")  # or "cpu" / "mps"

manual = cue.from_conversation([
    {"role": "user", "content": "can you tighten this paragraph"},
    {"role": "assistant", "content": "Sure — here is a shorter version."},
    {"role": "user", "content": "still too long, cut it in half"},
])

print(manual.steering_prompt)   # paste this text snippet into a user-simulator system prompt used with any LLM
print(manual.commands)          # flat command list
```

Accepted conversation formats: OpenAI-style message lists, role-prefixed transcripts
(`user: …` / `assistant: …`), or files via `cue.conversation.load_conversation`.

### 2. Sample a new CUE (no dialogue)

```python
manual = cue.sample(seed=0)           # one user
manuals = cue.sample(n=4, seed=0)     # batch → list[PersonaManual]
```

Requires `sampler.pt` on the Hub repo (shipped with `handshake-ai-research/cue`).

### 3. Steer any base simulator

```python
from cue import OpenAICompatSimulator, HuggingFaceSimulator
from cue.simulator import run_user_turn

# OpenAI, vLLM, OpenRouter, Azure, … — anything Chat Completions–compatible
sim = OpenAICompatSimulator("gpt-4o-mini")
# or: sim = OpenAICompatSimulator("meta-llama/…", base_url="http://localhost:8000/v1", api_key="EMPTY")
# or: sim = HuggingFaceSimulator("meta-llama/Llama-3.1-8B-Instruct")

user_msg = run_user_turn(
    sim,
    scenario="You want help planning a 3-day trip to Lisbon on a tight budget.",
    history=[],                 # prior turns in assistant POV (user/assistant roles)
    manual=manual,              # PersonaManual or a raw steering string
)
print(user_msg)
```

`run_user_turn` flips roles so the simulator sees its own past messages as its own.

## CLI

The config interface is stable. The subcommands below are convenient direct
interfaces for scripts and debugging:

```bash
# Conversation → steering prompt
cue conditioned -c conversation.json
cue conditioned -t 'user: hi\nassistant: hello\nuser: more detail please'

# Sample synthetic users
cue sample -n 3 --seed 0

# One steered user turn against a base LM
export OPENAI_API_KEY=...
cue chat --scenario 'Book a cheap flight to Boston' -c conversation.json \
  --simulator openai --sim-model gpt-4o-mini

cue chat --scenario '…' --sample --simulator hf \
  --sim-model meta-llama/Llama-3.1-8B-Instruct
```

## What you get back

Config-driven runs write:

- `manuals.json`: structured general, user-specific, and style commands.
- `steering_prompts.txt`: prompts ready to inject into a simulator.
- `simulated_users.json`: generated user turns when a simulator is configured.
- `info.json`: resolved non-secret config and output paths.

`PersonaManual` fields:

| field | meaning |
|-------|---------|
| `steering_prompt` | Ready-to-inject text for a user-simulator system message |
| `commands` | Flat list of behavioral commands |
| `general` / `specific` / `style` | Per-head commands when dual decode is on |
| `examples` | Style samples (decoded or retrieved) |
| `embedding` | Raw CUE vector (`list[float]`, optional) |
| `source` | `"conditioned"` or `"sampled"` |

## Session preprocessing

Default is **`full`** (keep the whole dialogue). For document-heavy assistants (e.g. long
drafts), pass `session_preprocess="strip_document"` so the encoder is not dominated by
the document body. `user_only` keeps user turns only.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for Hatch commands, testing, type
checking, formatting, packaging, and CI. Model-download tests are marked
`model` and excluded from the default unit test run.

## License

Apache-2.0. Model weights on the Hub carry their own card/license.
