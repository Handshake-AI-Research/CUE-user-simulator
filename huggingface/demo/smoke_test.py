"""Call the deployed Space's endpoints once each and print what came back."""

from __future__ import annotations

import os

from gradio_client import Client

SPACE = os.environ.get("CUE_SPACE", "AnjaliRuban/cue-demo")
TRANSCRIPT = """user: can you tighten this paragraph
assistant: Sure - here is a shorter version.
user: still too long, cut it in half
assistant: Trimmed to two sentences.
user: fine. now make the tone less stiff"""

client = Client(SPACE, token=os.environ["HF_TOKEN"])
endpoints = [name for name, meta in client.view_api(return_format="dict")["named_endpoints"].items() for _ in [meta]]
print("endpoints:", endpoints, flush=True)

manual, steering = client.predict(TRANSCRIPT, "full", False, api_name="/extract_manual")
print("\n=== manual ===\n" + manual[:1200], flush=True)
print("\n=== steering prompt (first 600 chars) ===\n" + steering[:600], flush=True)

sampled = client.predict(2, 0, api_name="/sample_users")
print("\n=== sampled users ===\n" + sampled[:900], flush=True)

if os.environ.get("SMOKE_PAID") != "1":
    print("\nskipping /compare (billed Inference calls); set SMOKE_PAID=1 to include it")
    raise SystemExit(0)

steered, baseline = client.predict(
    "You want help planning a 3-day trip to Lisbon on a tight budget.",
    steering,
    1,
    True,
    api_name="/compare",
)
print("\n=== steered arm ===\n", steered, flush=True)
print("\n=== baseline arm ===\n", baseline, flush=True)
