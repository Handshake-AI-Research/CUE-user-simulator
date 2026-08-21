# Bundled Hugging Face adapter

`cue_hf` is the Transformers-compatible inference adapter shipped inside the
`cue-simulator` wheel. Hub repos still flatten this package into remote code
(`modeling_cue.py`, …). After changing files here, regenerate and push that
bundle so the Hub copy cannot drift from the packaged source:

```bash
uv run python -c "from cue_hf.remote_code import bundle_remote_code; bundle_remote_code('/tmp/cue-remote')"
# then upload the flattened files to handshake-ai-research/cue
```

`huggingface/tests/test_remote_code.py` regenerates the bundle from this
package and is the local drift guard. `tests/test_encode_parity.py` compares
encode() against `cue_training` when the research group is installed.

Originally vendored 2026-08-07 from CUE-clean, re-synced 2026-08-15, then
moved into this monorepo:

| cue_hf | cue_training source | Notes |
|--------|---------------------|-------|
| `cue_hf/schema.py` | `cue_training/data/schema.py` | inference subset |
| `cue_hf/render.py` | `cue_training/refinement/simulator.py` | `render_manual`, `render_dual_manual` only |
| `cue_hf/session_preprocess.py` | `cue_training/infer/session_preprocess.py` | verbatim |
| `cue_hf/encoder/*` | `cue_training/encoder/*` | `UnifiedEncoder` -> `CueEncoder` |
| `cue_hf/decoder/*` | `cue_training/decoder/*` | `UnifiedDecoder` -> `CueDecoder` |
| `cue_hf/modeling_cue.py` | `cue_training/model.py` | `UnifiedModel` -> `CueModel` |
| `cue_hf/sampler/*` | `cue_training/sampler/*` | training-only EMA dropped |
| `cue_hf/example_pool.py` | `cue_training/data/example_pool.py` | retrieve/load only |

Things to keep in sync, because no test fails without `cue_training`:

- `nn.TransformerEncoder(..., enable_nested_tensor=False)` in the session encoder.
- `clean_up_tokenization_spaces=False` on all three tokenizers.
