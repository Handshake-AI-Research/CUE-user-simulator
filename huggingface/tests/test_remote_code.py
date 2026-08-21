"""The point of bundling is loading with cue-hf uninstalled, so prove it in a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from cue_hf.remote_code import (
    AUTO_MAP,
    bundle_remote_code,
    collect_modules,
    flat_module,
)

_LOADER = """
import json, sys


class _BlockCueHf:
    # Blocking the import is what makes this an outside-user test even when cue-hf is
    # pip-installed here: importing it registers CueModel for model_type 'cue', and that
    # registration outranks the repo's auto_map.
    def find_spec(self, name, path=None, target=None):
        if name == "cue_hf" or name.startswith("cue_hf."):
            raise ImportError("cue_hf is blocked for this test")
        return None


sys.meta_path.insert(0, _BlockCueHf())
try:
    import cue_hf
except ImportError:
    pass
else:
    raise SystemExit("cue_hf is importable here, so this proves nothing")

from transformers import AutoModel

directory, sessions = sys.argv[1], json.loads(sys.argv[2])
model = AutoModel.from_pretrained(directory, trust_remote_code=True)
if not type(model).__module__.startswith("transformers_modules"):
    raise SystemExit(f"loaded {type(model).__module__}, not the bundled copy")
print(json.dumps(model.encode(sessions).tolist()))
"""


def test_flat_modules_cover_the_graph_without_subpackages():
    root = Path(__file__).resolve().parents[1] / "cue_hf"
    sources = collect_modules(root)
    assert "cue_hf.encoder.model" in sources
    assert "cue_hf.cli" not in sources
    for module, text in sources.items():
        assert "from cue_hf" not in text, f"{module} kept an absolute import"
    assert all("." not in flat_module(module) for module in sources)


def test_bundle_writes_auto_map_and_modules(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    bundle_remote_code(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["auto_map"] == AUTO_MAP
    for reference in AUTO_MAP.values():
        assert (tmp_path / f"{reference.split('.')[0]}.py").is_file()


def test_bundling_into_a_pushed_repo_uploads_only_code(tiny_model, tmp_path, monkeypatch):
    """Cluster exports are already on the Hub, so the weights must not round-trip."""

    import huggingface_hub

    source = tmp_path / "exported"
    tiny_model.save_pretrained(source)
    uploaded: dict[str, Path] = {}

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: str(source / filename),
    )

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo_id, **kwargs):
            uploaded["repo"] = repo_id

        def upload_folder(self, repo_id, folder_path, **kwargs):
            uploaded["folder"] = Path(folder_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    from cue_hf.remote_code import bundle_remote_code_in_repo

    bundle_remote_code_in_repo("org/cue")
    assert uploaded["repo"] == "org/cue"
    names = {path.name for path in uploaded["folder"].iterdir()}
    assert "modeling_cue.py" in names
    assert not any(name.endswith((".safetensors", ".bin", ".pt")) for name in names)
    config = json.loads((uploaded["folder"] / "config.json").read_text(encoding="utf-8"))
    assert config["auto_map"] == AUTO_MAP


def test_bundle_requires_an_exported_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="export the model first"):
        bundle_remote_code(tmp_path)


def test_bundled_repo_loads_without_the_package_installed(tiny_model, sessions, tmp_path):
    repo = tmp_path / "repo"
    tiny_model.save_pretrained(repo)
    bundle_remote_code(repo)
    expected = tiny_model.encode(sessions)

    result = subprocess.run(
        [sys.executable, "-c", _LOADER, str(repo), json.dumps(sessions)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
        # Drop the repo from sys.path so the bundled copy is the only cue-hf available.
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "HF_MODULES_CACHE": str(tmp_path / "modules"),
        },
    )
    assert result.returncode == 0, result.stderr
    loaded = torch.tensor(json.loads(result.stdout.strip().splitlines()[-1]))
    assert torch.allclose(expected, loaded, atol=1e-5)
