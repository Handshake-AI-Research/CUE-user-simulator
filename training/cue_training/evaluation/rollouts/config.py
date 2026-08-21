"""Strict shared rollout config loading and path resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from cue_training.utils.config import expand_config_text, storage_root


class ConfigError(ValueError):
    """Raised when a rollout config is invalid or incomplete."""


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list")
    return value


def _as_path(value: Any, *, root: Path, path: str) -> Path:
    if value is None or value == "":
        raise ConfigError(f"{path} is required")
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _optional_path(value: Any, *, root: Path) -> Path | None:
    if value is None or value == "":
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _optional_checkpoint_path(value: Any, *, root: Path, path: str) -> Path | None:
    """Resolve local checkpoints or cache a published CUE Hugging Face export."""
    if value is None or value == "":
        return None
    raw = str(value)
    if raw.startswith("hf://"):
        repo_id = raw.removeprefix("hf://")
        if not repo_id:
            raise ConfigError(f"{path} has an empty Hugging Face repo id")
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(repo_id=repo_id))
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"{path}: failed to download {repo_id}: {exc}") from exc
    return _optional_path(raw, root=root)


def _unknown_keys(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ConfigError(f"{path} has unknown keys: {', '.join(extra)}")


def _parse_encode_session_preprocess(value: Any) -> str:
    mode = str(value or "auto").strip().lower() or "auto"
    allowed = {"auto", "full", "strip_document", "user_only"}
    if mode not in allowed:
        raise ConfigError(
            f"cue.encode_session_preprocess must be one of "
            f"{sorted(allowed)}, got {value!r}"
        )
    return mode


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048


@dataclass(frozen=True)
class SimulatorConfig:
    name: str
    kind: str  # api | vllm
    model: str
    served_name: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    gpus: tuple[int, ...] = ()
    port: int | None = None
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.92
    max_model_len: int | None = 16384
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass(frozen=True)
class AssistantConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    # When true, SimArena uses each episode's original assistant (mapped to a live LiteLLM
    # id); PRISM uses metadata.assistant_model when present else gpt-4-turbo. Tau2 keeps
    # ``model`` (its agent is not drawn from a multi-model human pool in the same way).
    match_dataset: bool = False


@dataclass(frozen=True)
class Tau2Benchmark:
    enabled: bool = True
    normalized: Path | None = None
    tau2_path: Path | None = None
    embeddings: Path | None = None
    cases_dir: Path | None = None


@dataclass(frozen=True)
class SimArenaDomain:
    enabled: bool = True
    annotations: Path | None = None
    embeddings: Path | None = None
    background: Path | None = None
    profiles_dir: Path | None = None


@dataclass(frozen=True)
class PrismBenchmark:
    enabled: bool = True
    normalized: Path | None = None
    embeddings: Path | None = None
    max_user_turns: int | None = None


@dataclass(frozen=True)
class BenchmarksConfig:
    tau2: Tau2Benchmark = field(default_factory=Tau2Benchmark)
    simulatorarena: dict[str, SimArenaDomain] = field(default_factory=dict)
    prism: PrismBenchmark = field(default_factory=PrismBenchmark)


@dataclass(frozen=True)
class ResourcesConfig:
    gpus: tuple[int, ...] = (0, 1)
    sidecar_port_start: int = 8713
    sidecar_port_end: int = 8799
    concurrency: int = 64
    output_root: Path = field(default_factory=lambda: storage_root() / "outputs" / "rollouts")
    cache_root: Path | None = None
    storage_root: Path | None = None
    vllm_venv: Path | None = None
    vllm_python: str = "3.12"
    start_vllm_script: Path | None = None
    # GPU for the CUE decoder sidecar. None -> auto-pick the first GPU not used by a sim/policy
    # vLLM this run starts (avoids OOM-colliding with the sim on GPU 0).
    sidecar_gpu: int | None = None


@dataclass(frozen=True)
class CueConfig:
    data_annotation_config: Path | None = None
    proposers: tuple[str, ...] = ()
    joint_training_checkpoint: Path | None = None
    refinement_checkpoints: dict[str, Path] = field(default_factory=dict)
    arms: tuple[str, ...] = ("paired", "sample_shuffled")
    decoder_dir: Path | None = None
    semantic_model: str = "intfloat/e5-base-v2"
    decode_temperature: float = 0.8
    decode_top_p: float = 0.9
    decode_num_candidates: int = 1
    decode_mode: str = "sample"
    decode_slot_dedup_jaccard: float = 0.5
    decode_noop_retries: int = 0
    decode_noop_temperature: float = 0.7
    decode_batch_size: int = 16
    example_retrieval_enabled: bool = False
    example_retrieval_k_sessions: int = 8
    example_retrieval_n_general: int = 2
    example_retrieval_n_specific: int = 2
    # Preprocess turns before live-encode / (optional) embedding export.
    # auto = strip_document for writing, full otherwise.
    encode_session_preprocess: str = "auto"
    # Diffusion sampler for sampled_uncond / sampled_pop arms.
    sampler_checkpoint: Path | None = None
    sampler_seed: int = 0
    sampler_guidance_w: float = 1.5
    sampler_ddim_steps: int | None = None
    sampler_device: str | None = None
    # Per-episode random condition-set size for sampled_pop (default 8 in sampler_arms).
    sampler_pop_max_k: int | None = None
    # Project sampled vectors onto the decoder LayerNorm manifold; None = use checkpoint config.
    sampler_project_manifold: bool | None = None

    def session_preprocess_for(self, domain: str) -> str:
        if self.encode_session_preprocess != "auto":
            return self.encode_session_preprocess
        return "strip_document" if domain == "writing" else "full"

    def decoder_dir_for(self, method: str, simulator: str) -> Path | None:
        """Resolve the decoder checkpoint for a CUE job: refinement (refined, per-sim) ->
        joint_training -> explicit decoder_dir."""

        if method == "refined":
            return self.refinement_checkpoints.get(simulator) or self.decoder_dir
        return self.joint_training_checkpoint or self.decoder_dir


@dataclass(frozen=True)
class BaselinesConfig:
    artifact_root: Path = field(default_factory=lambda: storage_root() / "baselines")
    train_data_path: Path | None = None
    train_jsonl_names: tuple[str, ...] = ("train.jsonl",)
    ppol_best_program: Path | None = None
    paper_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsConfig:
    enabled: tuple[str, ...] = ()
    oss_embed_url: str | None = None
    oss_gen_url: str | None = None


@dataclass(frozen=True)
class RolloutsConfig:
    root: Path
    benchmarks: BenchmarksConfig
    simulators: tuple[SimulatorConfig, ...]
    assistant: AssistantConfig
    resources: ResourcesConfig
    cue: CueConfig
    baselines: BaselinesConfig
    metrics: MetricsConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_resolved_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_resolved_dict(self) -> dict[str, Any]:
        return _dataclass_to_jsonable(self, skip={"raw"})

    def simulator(self, name: str) -> SimulatorConfig:
        for sim in self.simulators:
            if sim.name == name:
                return sim
        raise ConfigError(f"unknown simulator {name!r}")

    def enabled_benchmarks(self) -> list[tuple[str, str]]:
        """Return ``(benchmark, domain)`` pairs selected by config.

        Order is intentional for the rollout orchestrator: tau2 first, then
        SimulatorArena and Prism. ``pack_job_waves`` preserves this order across waves.
        """

        out: list[tuple[str, str]] = []
        if self.benchmarks.tau2.enabled:
            out.append(("tau2", "customer-service"))
        for domain, cfg in sorted(self.benchmarks.simulatorarena.items()):
            if cfg.enabled:
                out.append(("simulatorarena", domain))
        if self.benchmarks.prism.enabled:
            out.append(("prism", "diversity"))
        return out


def _dataclass_to_jsonable(obj: Any, *, skip: set[str] | None = None) -> Any:
    skip = skip or set()
    if is_dataclass(obj):
        data = {}
        for f in fields(obj):
            if f.name in skip:
                continue
            data[f.name] = _dataclass_to_jsonable(getattr(obj, f.name), skip=skip)
        return data
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_dataclass_to_jsonable(x, skip=skip) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_jsonable(v, skip=skip) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_jsonable(x, skip=skip) for x in obj]
    return obj


def _load_sampling(raw: Any, path: str) -> SamplingConfig:
    if raw is None:
        return SamplingConfig()
    data = _require_dict(raw, path)
    _unknown_keys(data, {"temperature", "top_p", "max_tokens"}, path)
    return SamplingConfig(
        temperature=float(data.get("temperature", 0.7)),
        top_p=float(data.get("top_p", 0.95)),
        max_tokens=int(data.get("max_tokens", 2048)),
    )


def _load_simulator(raw: Any, *, root: Path, path: str) -> SimulatorConfig:
    data = _require_dict(raw, path)
    _unknown_keys(
        data,
        {
            "name", "kind", "model", "served_name", "api_key_env", "api_base",
            "gpus", "port", "tensor_parallel", "gpu_memory_utilization",
            "max_model_len", "sampling",
        },
        path,
    )
    name = str(data.get("name") or "").strip()
    kind = str(data.get("kind") or "").strip().lower()
    model = str(data.get("model") or "").strip()
    if not name or not model:
        raise ConfigError(f"{path} requires name and model")
    if kind not in {"api", "vllm"}:
        raise ConfigError(f"{path}.kind must be 'api' or 'vllm'")
    gpus = tuple(int(x) for x in data.get("gpus") or ())
    port = data.get("port")
    if kind == "vllm":
        if port is None:
            raise ConfigError(f"{path}.port is required for vllm simulators")
        if not gpus:
            raise ConfigError(f"{path}.gpus is required for vllm simulators")
    return SimulatorConfig(
        name=name,
        kind=kind,
        model=model,
        served_name=(str(data["served_name"]) if data.get("served_name") else None),
        api_key_env=str(data.get("api_key_env") or ("HOSTED_VLLM_API_KEY" if kind == "vllm" else "OPENAI_API_KEY")),
        api_base=(str(data["api_base"]) if data.get("api_base") else None),
        gpus=gpus,
        port=(int(port) if port is not None else None),
        tensor_parallel=int(data.get("tensor_parallel") or max(1, len(gpus) or 1)),
        gpu_memory_utilization=float(data.get("gpu_memory_utilization") or 0.92),
        max_model_len=(int(data["max_model_len"]) if data.get("max_model_len") is not None else 16384),
        sampling=_load_sampling(data.get("sampling"), f"{path}.sampling"),
    )


def load_rollouts_config(path: str | Path, *, repo_root: Path | None = None) -> RolloutsConfig:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"rollout config not found: {cfg_path}")
    root = repo_root or (cfg_path.parents[1] if cfg_path.parent.name == "configs" else cfg_path.parent)
    root = Path(root).resolve()
    try:
        raw = json.loads(expand_config_text(cfg_path.read_text(encoding="utf-8"), cfg_path))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {cfg_path}: {exc}") from exc
    data = _require_dict(raw, "config")
    _unknown_keys(
        data,
        {"root", "benchmarks", "simulators", "assistant", "resources", "cue", "baselines", "metrics"},
        "config",
    )
    # Generated copies (e.g. per-seed configs under scratch) carry the repo root so their
    # relative paths keep resolving against the checkout rather than the copy's directory.
    if repo_root is None and data.get("root"):
        root = Path(str(data["root"])).expanduser().resolve()

    bench_raw = _require_dict(data.get("benchmarks") or {}, "benchmarks")
    _unknown_keys(bench_raw, {"tau2", "simulatorarena", "prism"}, "benchmarks")

    tau2_raw = _require_dict(bench_raw.get("tau2") or {}, "benchmarks.tau2")
    _unknown_keys(tau2_raw, {"enabled", "normalized", "tau2_path", "embeddings", "cases_dir"}, "benchmarks.tau2")
    tau2 = Tau2Benchmark(
        enabled=bool(tau2_raw.get("enabled", True)),
        normalized=_optional_path(tau2_raw.get("normalized"), root=root),
        tau2_path=_optional_path(tau2_raw.get("tau2_path"), root=root),
        embeddings=_optional_path(tau2_raw.get("embeddings"), root=root),
        cases_dir=_optional_path(tau2_raw.get("cases_dir"), root=root),
    )

    simarena_raw = _require_dict(bench_raw.get("simulatorarena") or {}, "benchmarks.simulatorarena")
    simarena: dict[str, SimArenaDomain] = {}
    for domain, domain_raw in simarena_raw.items():
        dpath = f"benchmarks.simulatorarena.{domain}"
        d = _require_dict(domain_raw, dpath)
        _unknown_keys(d, {"enabled", "annotations", "embeddings", "background", "profiles_dir"}, dpath)
        simarena[str(domain)] = SimArenaDomain(
            enabled=bool(d.get("enabled", True)),
            annotations=_optional_path(d.get("annotations"), root=root),
            embeddings=_optional_path(d.get("embeddings"), root=root),
            background=_optional_path(d.get("background"), root=root),
            profiles_dir=_optional_path(d.get("profiles_dir"), root=root),
        )

    prism_raw = _require_dict(bench_raw.get("prism") or {}, "benchmarks.prism")
    _unknown_keys(prism_raw, {"enabled", "normalized", "embeddings", "max_user_turns"}, "benchmarks.prism")
    prism = PrismBenchmark(
        enabled=bool(prism_raw.get("enabled", True)),
        normalized=_optional_path(prism_raw.get("normalized"), root=root),
        embeddings=_optional_path(prism_raw.get("embeddings"), root=root),
        max_user_turns=(int(prism_raw["max_user_turns"]) if prism_raw.get("max_user_turns") is not None else None),
    )

    sims_raw = _require_list(data.get("simulators") or [], "simulators")
    if not sims_raw:
        raise ConfigError("simulators must be a non-empty list")
    simulators = tuple(
        _load_simulator(item, root=root, path=f"simulators[{i}]") for i, item in enumerate(sims_raw)
    )
    names = [s.name for s in simulators]
    if len(names) != len(set(names)):
        raise ConfigError("simulator names must be unique")

    asst_raw = _require_dict(data.get("assistant") or {}, "assistant")
    _unknown_keys(asst_raw, {"model", "api_key_env", "api_base", "sampling", "match_dataset"}, "assistant")
    if not asst_raw.get("model"):
        raise ConfigError("assistant.model is required")
    assistant = AssistantConfig(
        model=str(asst_raw["model"]),
        api_key_env=str(asst_raw.get("api_key_env") or "OPENAI_API_KEY"),
        api_base=(str(asst_raw["api_base"]) if asst_raw.get("api_base") else None),
        sampling=_load_sampling(asst_raw.get("sampling"), "assistant.sampling"),
        match_dataset=bool(asst_raw.get("match_dataset") or False),
    )

    res_raw = _require_dict(data.get("resources") or {}, "resources")
    _unknown_keys(
        res_raw,
        {
            "gpus", "sidecar_port_start", "sidecar_port_end", "concurrency",
            "output_root", "cache_root", "storage_root", "vllm_venv",
            "vllm_python", "start_vllm_script", "sidecar_gpu",
        },
        "resources",
    )
    gpus = tuple(int(x) for x in (res_raw.get("gpus") or [0, 1]))
    if not gpus:
        raise ConfigError("resources.gpus must be non-empty")
    resources = ResourcesConfig(
        gpus=gpus,
        sidecar_port_start=int(res_raw.get("sidecar_port_start") or 8713),
        sidecar_port_end=int(res_raw.get("sidecar_port_end") or 8799),
        concurrency=int(res_raw.get("concurrency") or 64),
        output_root=_as_path(
            res_raw.get("output_root") or storage_root() / "outputs" / "rollouts",
            root=root,
            path="resources.output_root",
        ),
        cache_root=_optional_path(res_raw.get("cache_root"), root=root),
        storage_root=_optional_path(res_raw.get("storage_root"), root=root),
        vllm_venv=_optional_path(res_raw.get("vllm_venv"), root=root),
        vllm_python=str(res_raw.get("vllm_python") or "3.12"),
        start_vllm_script=_optional_path(res_raw.get("start_vllm_script"), root=root)
        or (root / "scripts" / "start_vllm.sh"),
        sidecar_gpu=(int(res_raw["sidecar_gpu"]) if res_raw.get("sidecar_gpu") is not None else None),
    )

    cue_raw = _require_dict(data.get("cue") or {}, "cue")
    _unknown_keys(
        cue_raw,
        {
            "data_annotation_config", "proposers", "joint_training_checkpoint", "refinement_checkpoints",
            "arms", "decoder_dir", "semantic_model", "decode_temperature", "decode_top_p",
            "decode_num_candidates", "decode_mode", "decode_slot_dedup_jaccard",
            "decode_noop_retries", "decode_noop_temperature",
            "decode_batch_size",
            "example_retrieval_enabled", "example_retrieval_k_sessions",
            "example_retrieval_n_general", "example_retrieval_n_specific",
            "encode_session_preprocess",
            "sampler_checkpoint", "sampler_seed", "sampler_guidance_w",
            "sampler_ddim_steps", "sampler_device", "sampler_pop_max_k",
            "sampler_project_manifold",
        },
        "cue",
    )
    refine_raw = _require_dict(cue_raw.get("refinement_checkpoints") or {}, "cue.refinement_checkpoints")
    cue = CueConfig(
        data_annotation_config=_optional_path(cue_raw.get("data_annotation_config"), root=root),
        proposers=tuple(str(x) for x in (cue_raw.get("proposers") or ())),
        joint_training_checkpoint=_optional_checkpoint_path(
            cue_raw.get("joint_training_checkpoint"), root=root,
            path="cue.joint_training_checkpoint",
        ),
        refinement_checkpoints={
            str(k): _optional_checkpoint_path(
                v, root=root, path=f"cue.refinement_checkpoints.{k}"
            )
            for k, v in refine_raw.items()
            if v
        },
        arms=tuple(str(x) for x in (cue_raw.get("arms") or ["paired", "sample_shuffled"])),
        decoder_dir=_optional_checkpoint_path(
            cue_raw.get("decoder_dir"), root=root, path="cue.decoder_dir"
        ),
        semantic_model=str(cue_raw.get("semantic_model") or "intfloat/e5-base-v2"),
        decode_temperature=float(cue_raw.get("decode_temperature", 0.8)),
        decode_top_p=float(cue_raw.get("decode_top_p", 0.9)),
        decode_num_candidates=int(cue_raw.get("decode_num_candidates") or 1),
        decode_mode=str(cue_raw.get("decode_mode") or "sample"),
        decode_slot_dedup_jaccard=float(cue_raw.get("decode_slot_dedup_jaccard") or 0.5),
        decode_noop_retries=int(cue_raw.get("decode_noop_retries") or 0),
        decode_noop_temperature=float(cue_raw.get("decode_noop_temperature") or 0.7),
        decode_batch_size=int(cue_raw.get("decode_batch_size") or 16),
        example_retrieval_enabled=bool(cue_raw.get("example_retrieval_enabled", False)),
        example_retrieval_k_sessions=int(cue_raw.get("example_retrieval_k_sessions") or 8),
        example_retrieval_n_general=int(cue_raw.get("example_retrieval_n_general") or 2),
        example_retrieval_n_specific=int(cue_raw.get("example_retrieval_n_specific") or 2),
        encode_session_preprocess=_parse_encode_session_preprocess(
            cue_raw.get("encode_session_preprocess")
        ),
        sampler_checkpoint=_optional_checkpoint_path(
            cue_raw.get("sampler_checkpoint"), root=root, path="cue.sampler_checkpoint"
        ),
        sampler_seed=int(cue_raw.get("sampler_seed") or 0),
        sampler_guidance_w=float(cue_raw.get("sampler_guidance_w") or 1.5),
        sampler_ddim_steps=(
            int(cue_raw["sampler_ddim_steps"]) if cue_raw.get("sampler_ddim_steps") is not None else None
        ),
        sampler_device=(str(cue_raw["sampler_device"]) if cue_raw.get("sampler_device") else None),
        sampler_pop_max_k=(
            int(cue_raw["sampler_pop_max_k"]) if cue_raw.get("sampler_pop_max_k") is not None else None
        ),
        sampler_project_manifold=(
            bool(cue_raw["sampler_project_manifold"])
            if cue_raw.get("sampler_project_manifold") is not None
            else None
        ),
    )

    base_raw = _require_dict(data.get("baselines") or {}, "baselines")
    _unknown_keys(
        base_raw,
        {
            "artifact_root", "train_data_path", "train_jsonl_names",
            "ppol_best_program", "paper_overrides",
        },
        "baselines",
    )
    baselines = BaselinesConfig(
        artifact_root=_as_path(
            base_raw.get("artifact_root") or storage_root() / "baselines",
            root=root,
            path="baselines.artifact_root",
        ),
        train_data_path=_optional_path(base_raw.get("train_data_path"), root=root),
        train_jsonl_names=tuple(str(x) for x in (base_raw.get("train_jsonl_names") or ["train.jsonl"])),
        ppol_best_program=_optional_path(base_raw.get("ppol_best_program"), root=root),
        paper_overrides=dict(base_raw.get("paper_overrides") or {}),
    )

    metrics_raw = _require_dict(data.get("metrics") or {}, "metrics")
    _unknown_keys(metrics_raw, {"enabled", "oss_embed_url", "oss_gen_url"}, "metrics")
    metrics = MetricsConfig(
        enabled=tuple(str(x) for x in (metrics_raw.get("enabled") or ())),
        oss_embed_url=(str(metrics_raw["oss_embed_url"]) if metrics_raw.get("oss_embed_url") else None),
        oss_gen_url=(str(metrics_raw["oss_gen_url"]) if metrics_raw.get("oss_gen_url") else None),
    )

    return RolloutsConfig(
        root=root,
        benchmarks=BenchmarksConfig(tau2=tau2, simulatorarena=simarena, prism=prism),
        simulators=simulators,
        assistant=assistant,
        resources=resources,
        cue=cue,
        baselines=baselines,
        metrics=metrics,
        raw=data,
    )


def write_resolved_config(cfg: RolloutsConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_resolved_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
