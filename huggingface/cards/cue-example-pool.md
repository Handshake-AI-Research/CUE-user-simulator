---
license: apache-2.0
language:
- en
size_categories:
- 100K<n<1M
tags:
- cue
- user-simulation
- retrieval
- embeddings
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: session_id
    dtype: string
  - name: cue_embedding
    list: float64
  - name: examples
    list:
    - name: text
      dtype: string
    - name: kind
      dtype: string
  splits:
  - name: train
    num_bytes: 1496250425
    num_examples: 160531
  download_size: 1494678142
  dataset_size: 1496250425
---

# CUE example pool

160,531 dialogue sessions, each stored as its 1024-d CUE embedding plus the real user turns
drawn from it. CUE uses this at decode time to ground a persona manual in things comparable
users actually said, rather than inventing illustrative examples.

| column | type | meaning |
|--------|------|---------|
| `session_id` | string | source session identifier |
| `cue_embedding` | list[float] | 1024-d CUE embedding of that session |
| `examples` | list[{text, kind}] | user turns, tagged `general` / `user_specific` / `style` |

## Usage

Retrieval is an optional extra (`datasets`, plus `faiss` for large pools):

```bash
pip install "cue-simulator[retrieval]"
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("handshake-ai-research/cue", trust_remote_code=True)
model.attach_example_pool("handshake-ai-research/cue-example-pool")
manual = model.generate_manual(sessions=session, example_retrieval=True)[0]
```

`config.example_pool_id` already names this dataset in the published CUE model, so
`generate_manual(..., example_retrieval=True)` attaches it without the explicit call.

## Compatibility

The embeddings live in the CUE space of
[`handshake-ai-research/cue`](https://huggingface.co/handshake-ai-research/cue) and must match
the model's 1024-d bottleneck; `attach_example_pool` raises on a mismatch. A pool built from a
different checkpoint is not interchangeable.
