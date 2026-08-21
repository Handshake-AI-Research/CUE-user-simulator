---
library_name: transformers
license: apache-2.0
tags:
- cue
- user-simulation
- dialogue
- persona
- custom_code
base_model:
- nomic-ai/modernbert-embed-base
- Qwen/Qwen3-0.6B-Base
datasets:
- handshake-ai-research/cue-annotations
- handshake-ai-research/cue-example-pool
inference: false
---

# CUE

CUE reads a user's dialogue history and writes a **persona manual**: a short list of behavioral
commands ("keep requests terse", "ask about price before committing") that can be handed to any
assistant LM so it role-plays that user. A session is encoded into a single 1024-d CUE
embedding, and a decoder turns that embedding back into commands.

This is the general checkpoint from joint training. One decoder drives any simulator; no
per-simulator refined variants are published.

## Usage

The repo carries its own code, so no install is required:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("handshake-ai-research/cue", trust_remote_code=True)

session = [[
    {"role": "user", "content": "hey can you tighten this paragraph"},
    {"role": "assistant", "content": "Sure — here is a shorter version."},
    {"role": "user", "content": "shorter, and less formal"},
]]

embedding = model.encode(session)                    # (1, 1024) CUE embedding
manual = model.generate_manual(sessions=session)[0]  # {"commands": [...], "examples": [...]}
```

For the CLI, the parity tests, or export tooling, install the package instead:

```bash
pip install cue-simulator
```

### Sampling synthetic users

`sampler.pt` (647 MB) in this repo is a latent diffusion prior over CUE embeddings, so you can
draw users that no transcript describes:

```python
out = model.sample_user(n=4, seed=0)   # {"embeddings": (4, 1024), "manuals": [...]}
```

### Session preprocessing

`session_preprocess` defaults to `full` (turns as-is) for all inputs. Pass `strip_document`
yourself for document-heavy inputs such as document editing, where an assistant turn carries a
whole draft that would otherwise dominate the encoded session:

```python
embedding = model.encode(session, session_preprocess="strip_document")
```

### Example retrieval (optional)

`config.example_pool_id` points at
[`handshake-ai-research/cue-example-pool`](https://huggingface.co/datasets/handshake-ai-research/cue-example-pool),
which lets the decoder ground commands in real turns from similar users. It needs the retrieval
extra (`pip install "cue-simulator[retrieval]"`, i.e. `datasets` plus optional `faiss`):

```python
manual = model.generate_manual(sessions=session, example_retrieval=True)[0]
```

## Architecture

| part | detail |
|------|--------|
| turn / context encoder | `nomic-ai/modernbert-embed-base`, 256 tokens per turn, 64 turns max, `search_query: ` / `search_document: ` prefixes |
| session encoder | 4-layer, 8-head transformer over turn embeddings, 1024-d |
| CUE embedding | 1024-d bottleneck (LayerNorm'd) |
| decoder | `Qwen/Qwen3-0.6B-Base` with Flamingo-style gated cross-attention every 4 layers, 8 heads, 16 persona tokens, bfloat16 |
| command slots | 5 general + 5 user-specific + 5 style, 64-d slot embeddings, dual decode |

Both backbones were finetuned during joint training, so `model.safetensors` (3.03 GB, 757
tensors) holds every weight. The base repos above are used only for their architecture configs
and tokenizers.
