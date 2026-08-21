---
title: CUE - persona manuals for user simulation
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
short_description: Persona manuals that steer a user simulator
---

# CUE demo

Three tabs:

1. **Conversation to manual** — encode a dialogue history into a CUE embedding and decode one
   command per slot for each head (general / user-specific / style), plus the rendered
   steering prompt.
2. **Does the manual steer a simulator?** — the same task and the same assistant, run twice,
   where only one arm's user side receives the manual. Any behavioral difference is CUE's.
3. **Sample users from the prior** — draw synthetic users from the diffusion prior with no
   input conversation, which is how CUE populates an evaluation.

## Configuration

| variable | default | purpose |
|----------|---------|---------|
| `HF_TOKEN` | — | **required secret**: reads the private model repos and calls the simulator |
| `CUE_REPO` | `handshake-ai-research/cue` | CUE checkpoint; the general decoder drives any simulator |
| `SIM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | simulator and assistant, via Inference Providers |
| `POOL_REPO` | `handshake-ai-research/cue-example-pool` | example pool, downloaded only if retrieval is checked |
| `PAID_INFERENCE` | `0` | ticks tab 2's "spend inference credits" box by default |

Tabs 1 and 3 cost nothing beyond the Space's own GPU time. Tab 2 is the only billed path:
the simulated user and the assistant are both Inference Provider calls, 4 per turn, so it
stays opt-in per run unless `PAID_INFERENCE=1`.

`cue_hf/` is vendored into this Space because the source repo is private; redeploy with
`demo/deploy.py` to refresh it.

## Check a deploy

`demo/smoke_test.py` calls all three tabs once through the Gradio API:

```bash
HF_TOKEN=hf_... uv run --no-project --with gradio_client python demo/smoke_test.py
```

Gradio's `sdk_version` floats, so component kwargs are the usual breakage; this catches it
before you open the page.
