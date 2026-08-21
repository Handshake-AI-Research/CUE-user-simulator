"""Deterministic GPU/port planning for managed serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from cue_training.evaluation.rollouts.artifacts import JobSpec
from cue_training.evaluation.rollouts.config import ConfigError, RolloutsConfig, SimulatorConfig
from cue_training.evaluation.rollouts.recipes import MethodRecipe, get_recipe
from cue_training.runlog.log import warn


@dataclass(frozen=True)
class VllmRequest:
    name: str
    model: str
    served_name: str
    port: int
    gpus: tuple[int, ...]
    tensor_parallel: int
    gpu_memory_utilization: float
    max_model_len: int | None
    prompt_embeds: bool = False


@dataclass
class ServePlan:
    groups: list[list[VllmRequest]] = field(default_factory=list)
    sidecar_ports: dict[str, int] = field(default_factory=dict)
    # Resolved policy-vLLM request per method (GPU assigned to the first free device).
    policy_requests: dict[str, VllmRequest] = field(default_factory=dict)


def _gpu_set(gpus: tuple[int, ...]) -> frozenset[int]:
    return frozenset(gpus)


def _conflicts(a: VllmRequest, b: VllmRequest) -> bool:
    if a.port == b.port and (a.model != b.model or a.served_name != b.served_name):
        return True
    if a.port == b.port and a.gpus != b.gpus:
        return True
    return bool(_gpu_set(a.gpus) & _gpu_set(b.gpus)) and not (
        a.model == b.model and a.served_name == b.served_name and a.port == b.port
    )


def _pack_groups(requests: list[VllmRequest]) -> list[list[VllmRequest]]:
    """Greedy sequential schedule: maximize reuse, never oversubscribe GPUs."""

    unique: dict[tuple, VllmRequest] = {}
    for req in requests:
        key = (req.model, req.served_name, req.port, req.gpus)
        unique[key] = req
    ordered = list(unique.values())
    groups: list[list[VllmRequest]] = []
    remaining = ordered[:]
    while remaining:
        group: list[VllmRequest] = []
        used_gpus: set[int] = set()
        used_ports: set[int] = set()
        next_remaining: list[VllmRequest] = []
        for req in remaining:
            if req.port in used_ports:
                next_remaining.append(req)
                continue
            gset = _gpu_set(req.gpus)
            if used_gpus & gset:
                # Allow exact reuse of same model/port/gpus already in group.
                if any(
                    r.model == req.model
                    and r.served_name == req.served_name
                    and r.port == req.port
                    and r.gpus == req.gpus
                    for r in group
                ):
                    continue
                next_remaining.append(req)
                continue
            group.append(req)
            used_gpus |= set(gset)
            used_ports.add(req.port)
        if not group:
            # Impossible packing (single request exceeds declared GPUs).
            raise ConfigError(
                f"cannot schedule vLLM request on declared GPUs: {remaining[0]}"
            )
        groups.append(group)
        remaining = next_remaining
    return groups


def policy_vllm_for(
    method: str, cfg: RolloutsConfig, *, gpu: int | None = None, port: int | None = None
) -> VllmRequest | None:
    """Policy-vLLM request for a model-loading baseline.

    ``gpu``/``port`` let the planner assign the first free device; explicit
    ``baselines.paper_overrides[method].gpus``/``port`` always win. When neither is
    given, falls back to the legacy default (second declared GPU).
    """

    overrides = cfg.baselines.paper_overrides.get(method) or {}
    gpus = cfg.resources.gpus
    if gpu is not None:
        default_gpu = int(gpu)
    else:
        default_gpu = int(gpus[1]) if len(gpus) >= 2 else int(gpus[0])
    default_port = int(port) if port is not None else 8000 + default_gpu
    policy_gpu = tuple(overrides.get("gpus") or (default_gpu,))
    policy_port = int(overrides.get("port") or default_port)
    if method == "userlm":
        # UserLM-8b's config caps max_position_embeddings at 8192 (unlike Llama-3.1-8B's 131072),
        # so vLLM rejects a larger --max-model-len. Default to 8192; overridable.
        return VllmRequest(
            name="userlm",
            model=str(overrides.get("model") or "microsoft/UserLM-8b"),
            served_name=str(overrides.get("served_name") or "userlm"),
            port=policy_port,
            gpus=policy_gpu,
            tensor_parallel=1,
            gpu_memory_utilization=float(overrides.get("gpu_memory_utilization") or 0.92),
            max_model_len=int(overrides.get("max_model_len") or 8192),
        )
    if method == "usp":
        pretrained = (
            overrides.get("pretrained_model")
            or overrides.get("model")
            or "wangkevin02/USP"
        )
        return VllmRequest(
            name="usp",
            model=str(pretrained),
            served_name=str(overrides.get("served_name") or pretrained),
            port=policy_port,
            gpus=policy_gpu,
            tensor_parallel=1,
            gpu_memory_utilization=float(overrides.get("gpu_memory_utilization") or 0.92),
            max_model_len=int(overrides.get("max_model_len") or 4096),
        )
    return None


def policy_served_name(cfg: RolloutsConfig, method: str) -> str:
    """Return the name clients must send to a managed policy vLLM server."""

    overrides = cfg.baselines.paper_overrides.get(method) or {}
    if method == "usp":
        return str(
            overrides.get("served_name")
            or overrides.get("pretrained_model")
            or overrides.get("model")
            or "wangkevin02/USP"
        )
    if method == "userlm":
        return str(overrides.get("served_name") or "userlm")
    raise ValueError(f"no policy served name for {method!r}")


def uses_pretrained_usp(cfg: RolloutsConfig) -> bool:
    """USP always uses the published HF checkpoint in cue_training."""

    return True


def simulator_vllm(sim: SimulatorConfig) -> VllmRequest | None:
    if sim.kind != "vllm" or sim.port is None:
        return None
    declared = set(sim.gpus)
    return VllmRequest(
        name=f"sim:{sim.name}",
        model=sim.model,
        served_name=sim.served_name or sim.model,
        port=int(sim.port),
        gpus=sim.gpus,
        tensor_parallel=sim.tensor_parallel,
        gpu_memory_utilization=sim.gpu_memory_utilization,
        max_model_len=sim.max_model_len,
    )


def validate_gpu_budget(cfg: RolloutsConfig, requests: list[VllmRequest]) -> None:
    available = set(cfg.resources.gpus)
    for req in requests:
        missing = set(req.gpus) - available
        if missing:
            raise ConfigError(
                f"vLLM {req.name} requests GPUs {sorted(missing)} not in resources.gpus={sorted(available)}"
            )
        if len(req.gpus) > len(available):
            raise ConfigError(
                f"vLLM {req.name} needs {len(req.gpus)} GPUs but only {len(available)} declared"
            )


def sidecar_key(cfg: RolloutsConfig, job: JobSpec) -> str:
    """Key that groups jobs sharing one sidecar. For CUE, the decoder is what the sidecar loads,
    so key on the resolved decoder dir: ``general`` (one joint-training dir) shares across simulators,
    while ``refined`` (per-simulator decoder-refinement dirs) stays split. Baselines key per simulator."""

    if job.rollout_type == "cue":
        ddir = cfg.cue.decoder_dir_for(job.method, job.simulator)
        return f"cue:{job.method}:{ddir or ''}"
    return f"{job.method}:{job.simulator}"


def allocate_sidecar_port(cfg: RolloutsConfig, used: set[int]) -> int:
    for port in range(cfg.resources.sidecar_port_start, cfg.resources.sidecar_port_end + 1):
        if port not in used:
            used.add(port)
            return port
    raise ConfigError("no free sidecar ports in configured range")


def _probe_port(port: int, *, host: str = "127.0.0.1", timeout: float = 2.0) -> list[str] | None:
    """Served model names at ``/v1/models`` on ``port``, or None if nothing is listening."""

    import json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    data = payload.get("data") or []
    return [str(item.get("id") or "") for item in data if isinstance(item, dict)]


ProbeFn = "callable"


def assign_policy_requests(
    cfg: RolloutsConfig, jobs: list[JobSpec], *, probe=_probe_port
) -> dict[str, VllmRequest]:
    """Place each model-loading baseline's vLLM on the next genuinely-available GPU.

    A GPU is skipped when (1) a simulator vLLM this run starts uses it, (2) another policy
    model this run already took it, or (3) probing its conventional port finds a live,
    *incompatible* server (an orphan from a prior run). A live *compatible* server is reused
    in place (no restart). Explicit ``baselines.paper_overrides[method].gpus`` always win.
    ``probe=None`` disables probing (dry-run / tests).
    """

    # GPUs used by simulator vLLMs this run actually starts are reserved up front; their ports
    # (and any other configured sim ports) seed the used-port set so fallbacks avoid them.
    occupied: set[int] = set()
    used_ports: set[int] = set()
    for job in jobs:
        recipe = get_recipe(job.rollout_type, job.method)
        if recipe.needs_sim_vllm or job.needs_sim_vllm:
            sreq = simulator_vllm(cfg.simulator(job.simulator))
            if sreq is not None:
                occupied |= set(sreq.gpus)
    for sim in cfg.simulators:
        sreq = simulator_vllm(sim)
        if sreq is not None:
            used_ports.add(sreq.port)

    policy_methods: list[str] = []
    for job in jobs:
        recipe = get_recipe(job.rollout_type, job.method)
        if (recipe.needs_policy_vllm or job.needs_policy_vllm) and job.method not in policy_methods:
            policy_methods.append(job.method)

    resolved: dict[str, VllmRequest] = {}
    taken: set[int] = set()
    for method in policy_methods:
        overrides = cfg.baselines.paper_overrides.get(method) or {}
        if overrides.get("gpus"):
            req = policy_vllm_for(method, cfg)
            if req is not None:
                resolved[method] = req
                taken |= set(req.gpus)
                used_ports.add(req.port)
            continue

        want = policy_vllm_for(method, cfg)
        if want is None:
            continue
        want_names = {want.served_name, str(want.model), Path(str(want.model)).name}

        chosen_gpu: int | None = None
        chosen_port: int | None = None
        for gpu in cfg.resources.gpus:
            if gpu in occupied or gpu in taken:
                continue
            port = 8000 + gpu
            while _in_sidecar_range(cfg, port):
                port += 1
            served = probe(port) if probe else None
            if served is None:
                chosen_gpu, chosen_port = gpu, port  # free
                break
            if want_names & set(served):
                chosen_gpu, chosen_port = gpu, port  # compatible live server -> reuse in place
                break
            # incompatible live server (orphan) -> skip this GPU
        if chosen_gpu is None:
            # No GPU has a free/compatible conventional port (orphans on all of them). Co-locate
            # on the first non-reserved GPU but pick an unused port so we never clobber a live
            # server. (Co-location shares GPU memory -- kill orphans if this OOMs.)
            candidates = [g for g in cfg.resources.gpus if g not in occupied] or list(cfg.resources.gpus)
            chosen_gpu = int(candidates[0])
            chosen_port = _first_unused_port(cfg, probe, used_ports)
            warn(
                "rollouts.plan",
                f"{method}: no free GPU port; co-locating on gpu {chosen_gpu} at unused port "
                f"{chosen_port} (kill orphaned vLLM servers if this OOMs)",
            )
        req = policy_vllm_for(method, cfg, gpu=chosen_gpu, port=chosen_port)
        resolved[method] = req
        used_ports.add(req.port)
        taken |= set(req.gpus)
    return resolved


def _first_unused_port(cfg: RolloutsConfig, probe, reserved: set[int]) -> int:
    """First port >= 8000 that is not reserved, not in the sidecar range, and (if probing)
    has no live server."""

    for port in range(8000, 10000):
        if port in reserved or _in_sidecar_range(cfg, port):
            continue
        if probe is not None and probe(port) is not None:
            continue
        return port
    raise ConfigError("no unused vLLM port available in 8000-9999")


def _in_sidecar_range(cfg: RolloutsConfig, port: int) -> bool:
    return cfg.resources.sidecar_port_start <= port <= cfg.resources.sidecar_port_end


def plan_serving(cfg: RolloutsConfig, jobs: list[JobSpec], *, probe=_probe_port) -> ServePlan:
    policy_requests = assign_policy_requests(cfg, jobs, probe=probe)
    requests: list[VllmRequest] = []
    sidecar_ports: dict[str, int] = {}
    # Seed the sidecar allocator with every vLLM port (sim + policy) so a sidecar can never
    # be assigned a port already held by a model server.
    used_sidecar: set[int] = {r.port for r in policy_requests.values()}
    for sim in cfg.simulators:
        sreq = simulator_vllm(sim)
        if sreq is not None:
            used_sidecar.add(sreq.port)
    for job in jobs:
        recipe = get_recipe(job.rollout_type, job.method)
        sim = cfg.simulator(job.simulator)
        if recipe.needs_sim_vllm or job.needs_sim_vllm:
            req = simulator_vllm(sim)
            if req is not None:
                requests.append(req)
        if (recipe.needs_policy_vllm or job.needs_policy_vllm) and job.method in policy_requests:
            requests.append(policy_requests[job.method])
        if recipe.needs_sidecar or job.needs_sidecar:
            key = sidecar_key(cfg, job)
            if key not in sidecar_ports:
                sidecar_ports[key] = allocate_sidecar_port(cfg, used_sidecar)
    validate_gpu_budget(cfg, requests)
    # Fail early if a single simultaneous group would need more GPUs than available
    # when methods request overlapping incompatible models on all GPUs at once.
    for req in requests:
        if len(set(req.gpus)) > len(cfg.resources.gpus):
            raise ConfigError(f"request {req.name} exceeds declared GPU count")
    return ServePlan(
        groups=_pack_groups(requests), sidecar_ports=sidecar_ports, policy_requests=policy_requests
    )


def jobs_for_methods(
    cfg: RolloutsConfig,
    *,
    rollout_type: str,
    methods: list[str],
    train: bool,
    arms: Sequence[str] | None = None,
) -> list[JobSpec]:
    """Build the job DAG. ``arms`` overrides config/recipe arms when provided.

    Benchmarks are the outer loop (tau2 → simarena → prism via
    ``enabled_benchmarks``) so multi-method baseline runs finish every method's
    tau2 before starting SimulatorArena/Prism — matching cue-general's order.
    """

    from cue_training.evaluation.rollouts.recipes import harness_variant
    from cue_training.evaluation.common.sampler_arms import needs_precomputed_embeddings

    jobs: list[JobSpec] = []
    arms_override = tuple(str(a) for a in arms) if arms else None
    for benchmark, domain in cfg.enabled_benchmarks():
        for method in methods:
            recipe: MethodRecipe = get_recipe(rollout_type, method)
            # Model-loading baselines (userlm/usp) generate from their own policy
            # vLLM and ignore the simulator model, so the API (gpt) sim is a pointless dupe:
            # collapse to a single simulator (prefer a vLLM entry) for these methods.
            if recipe.needs_policy_vllm:
                sims = [next((s for s in cfg.simulators if s.kind == "vllm"), cfg.simulators[0])]
            else:
                sims = list(cfg.simulators)
            if rollout_type == "cue" and method == "refined" and cfg.cue.decoder_dir is None:
                sims = [sim for sim in sims if sim.name in cfg.cue.refinement_checkpoints]
            for sim in sims:
                if rollout_type == "cue" and method == "proposals":
                    # Proposals is RealUserSim on data-annotation personas only — never inherit
                    # cue.arms sampled_* (those need the diffusion sampler / CUE vectors).
                    allowed = set(recipe.default_arms)
                    if arms_override is not None:
                        job_arms = [a for a in arms_override if a in allowed]
                        dropped = [a for a in arms_override if a not in allowed]
                        if dropped:
                            from cue_training.runlog.log import warn

                            warn(
                                "rollouts",
                                f"proposals ignores non-persona arms {dropped}; "
                                f"using {list(recipe.default_arms)}",
                            )
                        if not job_arms:
                            job_arms = list(recipe.default_arms)
                    else:
                        job_arms = list(recipe.default_arms)
                elif arms_override is not None:
                    job_arms = list(arms_override)
                else:
                    job_arms = list(cfg.cue.arms if rollout_type == "cue" else recipe.default_arms)
                jobs.append(
                    JobSpec(
                        rollout_type=rollout_type,
                        method=method,
                        benchmark=benchmark,
                        domain=domain,
                        simulator=sim.name,
                        arms=job_arms,
                        variant=harness_variant(recipe, simulator_kind=sim.kind),
                        needs_cue_embeddings=needs_precomputed_embeddings(
                            job_arms, recipe_needs=recipe.needs_cue_embeddings
                        ),
                        needs_sidecar=recipe.needs_sidecar,
                        needs_policy_vllm=recipe.needs_policy_vllm,
                        needs_sim_vllm=recipe.needs_sim_vllm and sim.kind == "vllm",
                        train=bool(train and recipe.train_supported),
                    )
                )
    return jobs


def vllm_requests_for_job(
    cfg: RolloutsConfig, job: JobSpec, policy_requests: dict[str, VllmRequest] | None = None
) -> list[VllmRequest]:
    """vLLM servers this job would start/reuse (empty for pure API sims)."""

    recipe = get_recipe(job.rollout_type, job.method)
    out: list[VllmRequest] = []
    if recipe.needs_sim_vllm or job.needs_sim_vllm:
        req = simulator_vllm(cfg.simulator(job.simulator))
        if req is not None:
            out.append(req)
    if recipe.needs_policy_vllm or job.needs_policy_vllm:
        pref = (policy_requests or {}).get(job.method) or policy_vllm_for(job.method, cfg)
        if pref is not None:
            out.append(pref)
    return out


def jobs_conflict(
    cfg: RolloutsConfig, a: JobSpec, b: JobSpec, policy_requests: dict[str, VllmRequest] | None = None
) -> bool:
    """True when two jobs need incompatible vLLM placements (same GPU, different model)."""

    for ra in vllm_requests_for_job(cfg, a, policy_requests):
        for rb in vllm_requests_for_job(cfg, b, policy_requests):
            if _conflicts(ra, rb):
                return True
    return False


def pack_job_waves(
    cfg: RolloutsConfig, jobs: list[JobSpec], policy_requests: dict[str, VllmRequest] | None = None
) -> list[list[JobSpec]]:
    """Pack concurrent waves of jobs that do not need incompatible vLLM placements.

    Jobs are bucketed by ``(benchmark, domain)`` so tau2 finishes before
    SimulatorArena/Prism (job order from ``jobs_for_methods``). Within a bucket,
    different methods share a wave when their vLLM needs do not conflict — e.g.
    ``usp`` (policy GPU) with ``realusersim`` (API-only), then ``userlm`` with
    ``ppol`` in a later wave once the GPU is free.
    """

    from collections import OrderedDict

    buckets: OrderedDict[tuple[str, str], list[JobSpec]] = OrderedDict()
    for job in jobs:
        buckets.setdefault((job.benchmark, job.domain), []).append(job)

    waves: list[list[JobSpec]] = []
    for group in buckets.values():
        remaining = list(group)
        while remaining:
            wave: list[JobSpec] = []
            next_remaining: list[JobSpec] = []
            for job in remaining:
                if any(jobs_conflict(cfg, job, other, policy_requests) for other in wave):
                    next_remaining.append(job)
                else:
                    wave.append(job)
            if not wave:
                wave = [remaining[0]]
                next_remaining = remaining[1:]
            waves.append(wave)
            remaining = next_remaining
    return waves
