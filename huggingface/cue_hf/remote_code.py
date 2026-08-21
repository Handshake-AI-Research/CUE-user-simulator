"""Bundle the inference code into a model repo so callers need no `pip install`.

Transformers' remote-code loader resolves a relative import by *filename*: for
``from .encoder.model import X`` it fetches a file literally named ``encoder.model.py``.
A Hub repo therefore cannot use subpackages, so this copies the module graph reachable
from the entry points into flat modules (``cue_hf.encoder.model`` -> ``encoder_model.py``)
and records an ``auto_map``. The installed package keeps its own layout.

Verifying a bundle requires an environment where ``cue_hf`` is *not* importable: importing
it registers ``CueModel`` for ``model_type: cue``, and that registration takes precedence
over ``auto_map``, so ``trust_remote_code=True`` silently loads the installed code instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cue_hf.log import log

PACKAGE = "cue_hf"
CONFIG_NAME = "config.json"
ENTRY_MODULES = (
    f"{PACKAGE}.configuration_cue",
    f"{PACKAGE}.modeling_cue",
    f"{PACKAGE}.processing_cue",
)
AUTO_MAP = {
    "AutoConfig": "configuration_cue.CueConfig",
    "AutoModel": "modeling_cue.CueModel",
}
# Leading whitespace is allowed because deferred imports sit inside functions, and the
# loader picks those up too.
_IMPORT = re.compile(rf"^([ \t]*)from {PACKAGE}(\.[A-Za-z0-9_.]+)? import ", re.MULTILINE)


def flat_module(module: str) -> str:
    """``cue_hf.encoder.model`` -> ``encoder_model``."""

    return module[len(PACKAGE) + 1 :].replace(".", "_")


def _source_file(module: str, root: Path) -> Path:
    relative = module[len(PACKAGE) + 1 :].replace(".", "/")
    direct = root / f"{relative}.py"
    return direct if direct.is_file() else root / relative / "__init__.py"


def _dependencies(text: str, module: str) -> list[str]:
    dependencies = []
    for match in _IMPORT.finditer(text):
        if match.group(2) is None:
            raise ValueError(
                f"{module} imports the {PACKAGE} package root, which has no flat "
                "equivalent; import the submodule directly instead"
            )
        dependencies.append(f"{PACKAGE}{match.group(2)}")
    return dependencies


def collect_modules(root: Path) -> dict[str, str]:
    """Map every module reachable from the entry points to its rewritten source."""

    sources: dict[str, str] = {}
    queue = list(ENTRY_MODULES)
    while queue:
        module = queue.pop()
        if module in sources:
            continue
        source = _source_file(module, root)
        if not source.is_file():
            raise FileNotFoundError(f"{module} has no source file at {source}")
        text = source.read_text(encoding="utf-8")
        queue.extend(_dependencies(text, module))
        sources[module] = _IMPORT.sub(lambda m: f"{m.group(1)}from .{flat_module(PACKAGE + m.group(2))} import ", text)
    flat = {}
    for module in sources:
        name = flat_module(module)
        if name in flat:
            raise ValueError(f"{module} and {flat[name]} both flatten to {name}.py")
        flat[name] = module
    return sources


def bundle_remote_code(
    directory: str | Path,
    *,
    push_to: str | None = None,
    private: bool = True,
    token: str | bool | None = None,
) -> Path:
    """Add flattened modules and an ``auto_map`` to an exported cue-hf model directory."""

    directory = Path(directory)
    config_path = directory / CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"{directory} has no {CONFIG_NAME}; export the model first")
    root = Path(__file__).resolve().parent
    for module, text in collect_modules(root).items():
        (directory / f"{flat_module(module)}.py").write_text(text, encoding="utf-8")
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = dict(AUTO_MAP)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    log("remote_code", f"bundled {len(ENTRY_MODULES)} entry points -> {directory}")
    if push_to:
        from huggingface_hub import HfApi

        api = HfApi(token=token if isinstance(token, str) else None)
        api.create_repo(push_to, private=private, exist_ok=True)
        api.upload_folder(repo_id=push_to, folder_path=str(directory))
        log("remote_code", f"pushed {directory} -> {push_to}")
    return directory


def bundle_remote_code_in_repo(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    private: bool = True,
) -> Path:
    """Add the code to a repo whose weights are already uploaded, without downloading them.

    For exports pushed straight off a cluster: only ``config.json`` comes down, and the
    upload adds the modules alongside the untouched weight files.
    """

    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    directory = Path(tempfile.mkdtemp(prefix="cue-hf-bundle-"))
    config = hf_hub_download(repo_id, CONFIG_NAME, revision=revision, token=token)
    shutil.copyfile(config, directory / CONFIG_NAME)
    return bundle_remote_code(directory, push_to=repo_id, private=private, token=token)
