"""Sample a synthetic user from the CUE prior (no dialogue required)."""

from __future__ import annotations

import os
import sys

from cue import Cue

CUE_REPO = os.environ.get("CUE_REPO", "handshake-ai-research/cue")

cue = Cue.from_pretrained(CUE_REPO, device=os.environ.get("CUE_DEVICE", "cpu"))
manual = cue.sample(seed=0)
sys.stdout.write(manual.steering_prompt + "\n")
