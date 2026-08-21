"""Create or refresh the private CUE demo Space.

The Space carries a copy of ``cue_hf`` because the source repo on GitHub is private and pip
cannot reach it from a Space build, so rerun this after changing the package.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

SPACE = os.environ.get("CUE_SPACE", "AnjaliRuban/cue-demo")
DEMO = Path(__file__).parent
PACKAGE = DEMO.parent / "cue_hf"


def main() -> None:
    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(DEMO),
        repo_id=SPACE,
        repo_type="space",
        allow_patterns=["app.py", "requirements.txt", "README.md"],
        commit_message="Update demo app",
    )
    api.upload_folder(
        folder_path=str(PACKAGE),
        path_in_repo="cue_hf",
        repo_id=SPACE,
        repo_type="space",
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        commit_message="Vendor cue_hf",
    )
    # Re-requesting hardware that is already assigned restarts the Space into a transitional
    # state with no GPU, so only ask when it is actually missing.
    runtime = api.get_space_runtime(SPACE)
    if "zero" not in str(runtime.requested_hardware or runtime.hardware or ""):
        api.request_space_hardware(SPACE, "zero-a10g")
        print("requested zero-a10g")

    secret = os.environ.get("CUE_SPACE_TOKEN")
    if secret:
        api.add_space_secret(SPACE, "HF_TOKEN", secret)
        print("set HF_TOKEN secret")
    else:
        print(
            "No CUE_SPACE_TOKEN in env, so HF_TOKEN was not set. The Space cannot read the "
            "private model repos until you add it under Settings > Variables and secrets. "
            "Use a fine-grained token with read + inference scope, not your write token."
        )
    print(f"https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
