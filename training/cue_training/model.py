"""End-to-end unified model: encoder bottleneck → decoder."""

from __future__ import annotations

from collections import OrderedDict
from itertools import permutations
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cue_training.data.schema import (
    COMMAND_BLOCK_SENTINEL,
    COMMAND_SLOT_SENTINEL,
    GENERAL_COMMAND_SENTINEL,
    MAX_COMMANDS,
    STYLE_COMMAND_SENTINEL,
    USER_SPECIFIC_SENTINEL,
    canonicalize_profile,
    clean_generated_command,
    merge_dual_manual_blocks,
    parse_manual,
)
from cue_training.decoder.model import UnifiedDecoder
from cue_training.encoder.model import UnifiedEncoder


def vicreg_var_cov(z: torch.Tensor, *, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + off-diagonal covariance on unnormalized projections."""

    # z: [B, D]
    if z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(1.0 - std).mean()
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / max(1, z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off**2).sum() / z.size(1)
    return var_loss, cov_loss


def greedy_command_overlap(
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric greedy best-match cosine between command sets."""

    rows = []
    for i in range(anchors.size(0)):
        a = anchors[i, anchor_mask[i]]
        values = []
        for j in range(candidates.size(0)):
            b = candidates[j, candidate_mask[j]]
            if not len(a) or not len(b):
                values.append(anchors.new_tensor(0.0))
                continue
            similarity = a @ b.T
            values.append(
                0.5
                * (
                    similarity.max(dim=1).values.mean()
                    + similarity.max(dim=0).values.mean()
                )
            )
        rows.append(torch.stack(values))
    return torch.stack(rows)


def command_overlap_infonce(
    z_local: torch.Tensor,
    z_global: torch.Tensor,
    overlap: torch.Tensor,
    *,
    temperature: float,
    target_temperature: float,
) -> torch.Tensor:
    targets = F.softmax(overlap / max(target_temperature, 1e-6), dim=-1)
    logits = z_local @ z_global.T / max(temperature, 1e-6)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def command_token_jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity on casefolded whitespace splits."""

    ta = set(a.casefold().split())
    tb = set(b.casefold().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def select_command_threshold(
    candidates: list[str],
    kept: list[str],
    *,
    near_dup_jaccard: float = 0.5,
) -> str | None:
    """Pick the first candidate that is not a near-duplicate of already-kept commands.

    Decorrelates paraphrases without repulsive MMR (which pushed toward semantic
    opposites). Sample order is treated as relevance (earlier ≈ closer to the mode).
    A candidate is skipped when max token-Jaccard vs ``kept`` ≥ ``near_dup_jaccard``.
    """

    cleaned = [str(c).strip() for c in candidates if str(c).strip()]
    if not cleaned:
        return None
    thresh = min(1.0, max(0.0, float(near_dup_jaccard)))
    for command in cleaned:
        max_sim = (
            max(command_token_jaccard(command, prev) for prev in kept) if kept else 0.0
        )
        if kept and max_sim >= thresh:
            continue
        return command
    return None


def command_slot_assignment(cost: torch.Tensor) -> list[int]:
    """Minimum-cost unique slot for each command in a small [slots, commands] matrix."""

    n_slots, n_commands = cost.shape
    if n_commands == 0:
        return []
    if n_commands > n_slots:
        raise ValueError(f"{n_commands} commands exceed {n_slots} slots")
    detached = cost.detach().float().cpu()
    return list(
        min(
            permutations(range(n_slots), n_commands),
            key=lambda slots: sum(float(detached[slot, cmd]) for cmd, slot in enumerate(slots)),
        )
    )


class UnifiedModel(nn.Module):
    def __init__(
        self,
        encoder: UnifiedEncoder,
        decoder: UnifiedDecoder,
        *,
        target_max_tokens: int = 1024,
        command_cache_size: int = 20_000,
        dual_decode: bool = False,
        lambda_general_ce: float = 0.5,
        lambda_user_specific_ce: float = 0.5,
        lambda_style_ce: float = 0.0,
        command_slot_decode: bool = False,
        general_command_slots: int = 5,
        user_specific_command_slots: int = 5,
        style_command_slots: int = 0,
        command_noop_weight: float = 0.25,
        lambda_dual_separation: float = 0.0,
        dual_separation_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_max_tokens = target_max_tokens
        self.command_cache_size = max(0, int(command_cache_size))
        self.dual_decode = bool(dual_decode)
        self.lambda_general_ce = float(lambda_general_ce)
        self.lambda_user_specific_ce = float(lambda_user_specific_ce)
        self.lambda_style_ce = float(lambda_style_ce)
        self.command_slot_decode = bool(command_slot_decode)
        if self.command_slot_decode and not self.dual_decode:
            raise ValueError("command_slot_decode requires dual_decode=true")
        self.general_command_slots = max(1, int(general_command_slots))
        self.user_specific_command_slots = max(1, int(user_specific_command_slots))
        self.style_command_slots = max(0, int(style_command_slots))
        self.command_noop_weight = max(0.0, float(command_noop_weight))
        self.lambda_dual_separation = max(0.0, float(lambda_dual_separation))
        self.dual_separation_margin = max(0.0, float(dual_separation_margin))
        self._command_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @torch.no_grad()
    def _command_tensors(
        self,
        command_sets: list[list[str]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = [[text.strip() for text in commands[:MAX_COMMANDS] if text.strip()] for commands in command_sets]
        missing = []
        for commands in normalized:
            for text in commands:
                key = text.casefold()
                if key not in self._command_cache and key not in missing:
                    missing.append(key)
        if missing:
            embedded = self.encoder.embed_texts(missing, device=device).float().cpu()
            for key, vector in zip(missing, embedded, strict=True):
                self._command_cache[key] = vector
                self._command_cache.move_to_end(key)
                while len(self._command_cache) > self.command_cache_size:
                    self._command_cache.popitem(last=False)

        hidden = int(self.encoder.system_encoder.config.hidden_size)
        values = torch.zeros(
            len(normalized), MAX_COMMANDS, hidden, device=device, dtype=torch.float32
        )
        mask = torch.zeros(len(normalized), MAX_COMMANDS, device=device, dtype=torch.bool)
        for i, commands in enumerate(normalized):
            for j, text in enumerate(commands):
                key = text.casefold()
                vector = self._command_cache.get(key)
                if vector is None:
                    vector = self.encoder.embed_texts([text], device=device)[0].float().cpu()
                else:
                    self._command_cache.move_to_end(key)
                values[i, j] = vector.to(device)
                mask[i, j] = True
        return values, mask

    def _cue_infonce(
        self,
        bottleneck: torch.Tensor,
        command_sets: list[list[str]] | None,
        *,
        temperature: float,
        target_temperature: float,
        hard_session: bool = False,
        z_positive: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        z_local = F.normalize(bottleneck, dim=-1)

        if hard_session:
            # Anchor = original session; positive keys = perturbed view of the same
            # session (when provided). Other sessions' perturbed views are negatives.
            z_keys_local = F.normalize(z_positive, dim=-1) if z_positive is not None else z_local
            z_global = z_keys_local
            offset = 0
            if self.training and dist.is_available() and dist.is_initialized():
                from torch.distributed.nn.functional import all_gather

                z_global = torch.cat(all_gather(z_keys_local), dim=0)
                offset = dist.get_rank() * z_local.size(0)

            logits = z_local @ z_global.T / max(temperature, 1e-6)
            labels = torch.arange(z_local.size(0), device=z_local.device) + offset
            loss = F.cross_entropy(logits, labels)
            return loss, None, z_local @ z_global.T

        z_global = z_local
        if self.training and dist.is_available() and dist.is_initialized():
            from torch.distributed.nn.functional import all_gather

            z_global = torch.cat(all_gather(z_local), dim=0)

        commands_local, masks_local = self._command_tensors(
            command_sets, device=bottleneck.device
        )
        commands_global = commands_local
        masks_global = masks_local
        if self.training and dist.is_available() and dist.is_initialized():
            gathered_commands = [torch.empty_like(commands_local) for _ in range(dist.get_world_size())]
            gathered_masks = [torch.empty_like(masks_local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_commands, commands_local)
            dist.all_gather(gathered_masks, masks_local)
            commands_global = torch.cat(gathered_commands, dim=0)
            masks_global = torch.cat(gathered_masks, dim=0)

        overlap = greedy_command_overlap(
            commands_local, masks_local, commands_global, masks_global
        )
        loss = command_overlap_infonce(
            z_local,
            z_global,
            overlap,
            temperature=temperature,
            target_temperature=target_temperature,
        )
        return loss, overlap, z_local @ z_global.T

    def _cue_style_loss(
        self,
        bottleneck: torch.Tensor,
        style_targets: torch.Tensor,
        *,
        temperature: float,
        target_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_local = F.normalize(bottleneck, dim=-1)
        z_global = z_local
        if self.training and dist.is_available() and dist.is_initialized():
            from torch.distributed.nn.functional import all_gather

            z_global = torch.cat(all_gather(z_local), dim=0)
        loss = command_overlap_infonce(
            z_local,
            z_global,
            style_targets,
            temperature=temperature,
            target_temperature=target_temperature,
        )
        return loss, z_local @ z_global.T

    def encode(self, sessions: list[list[dict[str, str]]]) -> torch.Tensor:
        return self.encoder.encode_sessions(sessions)["bottleneck"]

    def _encode_targets(
        self, target_texts: list[str], *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.decoder.tokenizer
        encoded = tok(
            [t + (tok.eos_token or "") for t in target_texts],
            padding=True,
            truncation=True,
            max_length=self.target_max_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def _sentinel_ids(self, sentinel: str, *, device: torch.device) -> torch.Tensor:
        return self.decoder.tokenizer(
            sentinel, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(device)

    def _slot_cost_matrices(
        self,
        bottleneck: torch.Tensor,
        command_sets: list[list[str]],
        *,
        head: str,
        num_slots: int,
    ) -> tuple[list[torch.Tensor], list[list[str]]]:
        """Teacher-forced CE for every slot/command pair, plus one empty target."""

        normalized = [
            [str(command).strip() for command in commands if str(command).strip()]
            for commands in command_sets
        ]
        normalized = [
            (
                [
                    commands[index]
                    for index in torch.randperm(len(commands))[:num_slots].tolist()
                ]
                if self.training and len(commands) > num_slots
                else commands[:num_slots]
            )
            for commands in normalized
        ]
        flat_targets: list[str] = []
        owners: list[int] = []
        offsets: list[tuple[int, int]] = []
        for sample, commands in enumerate(normalized):
            start = len(flat_targets)
            targets = commands + [""]  # Empty + EOS teaches unmatched slots to stop.
            flat_targets.extend(targets)
            owners.extend([sample] * len(targets))
            offsets.append((start, len(targets)))
        owner_tensor = torch.tensor(owners, device=bottleneck.device, dtype=torch.long)
        expanded = bottleneck.index_select(0, owner_tensor)
        target_ids, target_mask = self._encode_targets(
            flat_targets, device=bottleneck.device
        )
        sentinel = self._sentinel_ids(COMMAND_SLOT_SENTINEL, device=bottleneck.device)
        targets_per_slot = expanded.size(0)
        slot_ids = torch.cat(
            [
                torch.full(
                    (targets_per_slot,),
                    self.decoder.command_slot_id(head, slot),
                    device=bottleneck.device,
                    dtype=torch.long,
                )
                for slot in range(num_slots)
            ]
        )
        slot_losses = self.decoder(
            bottleneck=expanded.repeat(num_slots, 1),
            target_ids=target_ids.repeat(num_slots, 1),
            target_mask=target_mask.repeat(num_slots, 1),
            sentinel_ids=sentinel,
            slot_ids=slot_ids,
            reduction="none",
        ).view(num_slots, targets_per_slot)
        matrices = [
            slot_losses[:, start : start + width]
            for start, width in offsets
        ]
        return matrices, normalized

    def _command_set_loss(
        self,
        bottleneck: torch.Tensor,
        command_sets: list[list[str]],
        *,
        head: str,
        num_slots: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, str, torch.Tensor]]]:
        matrices, normalized = self._slot_cost_matrices(
            bottleneck, command_sets, head=head, num_slots=num_slots
        )
        sample_losses: list[torch.Tensor] = []
        matches: list[tuple[int, int, str, torch.Tensor]] = []
        for sample, (matrix, commands) in enumerate(zip(matrices, normalized, strict=True)):
            assignment = command_slot_assignment(matrix[:, : len(commands)])
            used = set(assignment)
            terms: list[torch.Tensor] = []
            weights: list[float] = []
            for command_index, slot in enumerate(assignment):
                own = matrix[slot, command_index]
                terms.append(own)
                weights.append(1.0)
                matches.append((sample, slot, commands[command_index], own))
            for slot in range(num_slots):
                if slot not in used and self.command_noop_weight > 0:
                    terms.append(matrix[slot, -1])
                    weights.append(self.command_noop_weight)
            if not terms:
                sample_losses.append(matrix.sum() * 0.0)
                continue
            weight_tensor = matrix.new_tensor(weights)
            sample_losses.append(
                (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum().clamp(min=1e-6)
            )
        return torch.stack(sample_losses).mean(), matches

    def _dual_separation_loss(
        self,
        bottleneck: torch.Tensor,
        general_matches: list[tuple[int, int, str, torch.Tensor]],
        specific_matches: list[tuple[int, int, str, torch.Tensor]],
    ) -> torch.Tensor:
        """Margin requiring each matched command to score worse under the other head.

        "dual" is exact: only the general and user-specific heads are cross-scored. Style is
        the third decode head but stays out of this, because style commands are meant to
        co-occur with both of the others.
        """

        terms: list[torch.Tensor] = []
        for wrong_head, matches in (
            ("specific", general_matches),
            ("general", specific_matches),
        ):
            if not matches:
                continue
            owners = torch.tensor(
                [row[0] for row in matches],
                device=bottleneck.device,
                dtype=torch.long,
            )
            target_ids, target_mask = self._encode_targets(
                [row[2] for row in matches], device=bottleneck.device
            )
            wrong = self.decoder(
                bottleneck=bottleneck.index_select(0, owners),
                target_ids=target_ids,
                target_mask=target_mask,
                sentinel_ids=self._sentinel_ids(
                    COMMAND_SLOT_SENTINEL, device=bottleneck.device
                ),
                slot_ids=torch.tensor(
                    [
                        self.decoder.command_slot_id(wrong_head, row[1])
                        for row in matches
                    ],
                    device=bottleneck.device,
                    dtype=torch.long,
                ),
                reduction="none",
            )
            own = torch.stack([row[3] for row in matches])
            terms.append(F.relu(self.dual_separation_margin + own - wrong))
        if not terms:
            return bottleneck.sum() * 0.0
        return torch.cat(terms).mean()

    def forward(
        self,
        *,
        sessions: list[list[dict[str, str]]],
        target_texts: list[str],
        lambda_consistency: float = 0.1,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_cue_infonce: float = 0.0,
        lambda_cue_style: float = 0.0,
        cue_infonce_temp: float = 0.1,
        cue_overlap_target_temp: float = 0.2,
        cue_hard_session: bool = False,
        style_targets: torch.Tensor | None = None,
        command_sets: list[list[str]] | None = None,
        perturbed_sessions: list[list[dict[str, str]]] | None = None,
        measure_shuffle: bool = False,
        general_targets: list[str] | None = None,
        user_specific_targets: list[str] | None = None,
        general_command_sets: list[list[str]] | None = None,
        user_specific_command_sets: list[list[str]] | None = None,
        style_manual_targets: list[str] | None = None,
        style_command_sets: list[list[str]] | None = None,
        lambda_general_ce: float | None = None,
        lambda_user_specific_ce: float | None = None,
        lambda_style_ce: float | None = None,
        lambda_dual_separation: float | None = None,
        dual_decode: bool | None = None,
    ) -> dict[str, Any]:
        enc = self.encoder.encode_sessions(sessions)
        bottleneck = enc["bottleneck"]
        device = bottleneck.device
        use_dual = self.dual_decode if dual_decode is None else bool(dual_decode)
        w_general = self.lambda_general_ce if lambda_general_ce is None else float(lambda_general_ce)
        w_specific = (
            self.lambda_user_specific_ce
            if lambda_user_specific_ce is None
            else float(lambda_user_specific_ce)
        )
        w_style = self.lambda_style_ce if lambda_style_ce is None else float(lambda_style_ce)
        w_separation = (
            self.lambda_dual_separation
            if lambda_dual_separation is None
            else float(lambda_dual_separation)
        )

        if use_dual:
            if general_targets is None or user_specific_targets is None:
                raise ValueError("dual_decode requires general_targets and user_specific_targets")
            general_ids, general_mask = self._encode_targets(general_targets, device=device)
            general_sentinel = self._sentinel_ids(GENERAL_COMMAND_SENTINEL, device=device)
            if self.command_slot_decode:
                if general_command_sets is None or user_specific_command_sets is None:
                    raise ValueError(
                        "command_slot_decode requires general and user-specific command sets"
                    )
                ce_general, general_matches = self._command_set_loss(
                    bottleneck,
                    general_command_sets,
                    head="general",
                    num_slots=self.general_command_slots,
                )
                ce_specific, specific_matches = self._command_set_loss(
                    bottleneck,
                    user_specific_command_sets,
                    head="specific",
                    num_slots=self.user_specific_command_slots,
                )
                ce = w_general * ce_general + w_specific * ce_specific
                metrics: dict[str, float] = {
                    "ce": float(ce.detach()),
                    "ce_general": float(ce_general.detach()),
                    "ce_user_specific": float(ce_specific.detach()),
                    "bn_finite": float(torch.isfinite(bottleneck).all().detach()),
                }
                if self.style_command_slots > 0 and w_style > 0:
                    ce_style, _ = self._command_set_loss(
                        bottleneck,
                        style_command_sets or [[] for _ in range(bottleneck.size(0))],
                        head="style",
                        num_slots=self.style_command_slots,
                    )
                    ce = ce + w_style * ce_style
                    metrics["ce"] = float(ce.detach())
                    metrics["ce_style"] = float(ce_style.detach())
            else:
                specific_ids, specific_mask = self._encode_targets(
                    user_specific_targets, device=device
                )
                specific_sentinel = self._sentinel_ids(
                    USER_SPECIFIC_SENTINEL, device=device
                )
                ce_general = self.decoder(
                    bottleneck=bottleneck,
                    target_ids=general_ids,
                    target_mask=general_mask,
                    sentinel_ids=general_sentinel,
                )
                ce_specific = self.decoder(
                    bottleneck=bottleneck,
                    target_ids=specific_ids,
                    target_mask=specific_mask,
                    sentinel_ids=specific_sentinel,
                )
                ce = w_general * ce_general + w_specific * ce_specific
                metrics = {
                    "ce": float(ce.detach()),
                    "ce_general": float(ce_general.detach()),
                    "ce_user_specific": float(ce_specific.detach()),
                    "bn_finite": float(torch.isfinite(bottleneck).all().detach()),
                }
                if w_style > 0 and style_manual_targets is not None:
                    style_ids, style_mask = self._encode_targets(
                        style_manual_targets, device=device
                    )
                    style_sentinel = self._sentinel_ids(
                        STYLE_COMMAND_SENTINEL, device=device
                    )
                    ce_style = self.decoder(
                        bottleneck=bottleneck,
                        target_ids=style_ids,
                        target_mask=style_mask,
                        sentinel_ids=style_sentinel,
                    )
                    ce = ce + w_style * ce_style
                    metrics["ce"] = float(ce.detach())
                    metrics["ce_style"] = float(ce_style.detach())
            if self.command_slot_decode and w_separation > 0:
                separation = self._dual_separation_loss(
                    bottleneck, general_matches, specific_matches
                )
                ce = ce + w_separation * separation
                metrics["dual_separation"] = float(separation.detach())
            sentinel_ids = general_sentinel
            target_ids = general_ids
            target_mask = general_mask
        else:
            target_ids, target_mask = self._encode_targets(target_texts, device=device)
            sentinel_ids = self._sentinel_ids(COMMAND_BLOCK_SENTINEL, device=device)
            ce = self.decoder(
                bottleneck=bottleneck,
                target_ids=target_ids,
                target_mask=target_mask,
                sentinel_ids=sentinel_ids,
            )
            metrics = {
                "ce": float(ce.detach()),
                "bn_finite": float(torch.isfinite(bottleneck).all().detach()),
            }
        loss = ce

        need_perturbed = perturbed_sessions is not None and (
            lambda_consistency > 0
            or (lambda_cue_infonce > 0 and cue_hard_session)
        )
        enc_p = (
            self.encoder.encode_sessions(perturbed_sessions)
            if need_perturbed
            else None
        )

        if enc_p is not None and lambda_consistency > 0:
            a = F.normalize(bottleneck, dim=-1, eps=1e-6)
            b = F.normalize(enc_p["bottleneck"], dim=-1, eps=1e-6)
            cons = (1.0 - (a * b).sum(dim=-1)).mean()
            loss = loss + lambda_consistency * cons
            metrics["consistency"] = float(cons.detach())

        if lambda_var > 0 or lambda_cov > 0:
            var_l, cov_l = vicreg_var_cov(enc["vicreg"])
            loss = loss + lambda_var * var_l + lambda_cov * cov_l
            metrics["var"] = float(var_l.detach())
            metrics["cov"] = float(cov_l.detach())

        if lambda_cue_infonce > 0 and (
            cue_hard_session
            or (command_sets is not None and len(command_sets) == bottleneck.size(0))
        ):
            cue_nce, command_overlap, cue_cosine = self._cue_infonce(
                bottleneck,
                command_sets,
                temperature=cue_infonce_temp,
                target_temperature=cue_overlap_target_temp,
                hard_session=cue_hard_session,
                z_positive=enc_p["bottleneck"] if enc_p is not None else None,
            )
            if cue_cosine.size(1) > 1:
                loss = loss + lambda_cue_infonce * cue_nce
                metrics["cue_infonce"] = float(cue_nce.detach())
                offset = 0
                if self.training and dist.is_available() and dist.is_initialized():
                    offset = dist.get_rank() * bottleneck.size(0)
                offdiag = torch.ones_like(cue_cosine, dtype=torch.bool)
                rows = torch.arange(bottleneck.size(0), device=bottleneck.device)
                offdiag[rows, offset + rows] = False
                if command_overlap is not None:
                    metrics["command_overlap_mean"] = float(
                        command_overlap.detach()[offdiag].mean()
                    )
                metrics["cue_cosine_mean"] = float(cue_cosine.detach()[offdiag].mean())

        if (
            lambda_cue_style > 0
            and style_targets is not None
            and style_targets.shape[0] == bottleneck.size(0)
        ):
            style_nce, cue_cosine = self._cue_style_loss(
                bottleneck,
                style_targets,
                temperature=cue_infonce_temp,
                target_temperature=cue_overlap_target_temp,
            )
            if cue_cosine.size(1) > 1:
                loss = loss + lambda_cue_style * style_nce
                metrics["cue_style"] = float(style_nce.detach())
                offset = 0
                if self.training and dist.is_available() and dist.is_initialized():
                    offset = dist.get_rank() * bottleneck.size(0)
                offdiag = torch.ones_like(cue_cosine, dtype=torch.bool)
                rows = torch.arange(bottleneck.size(0), device=bottleneck.device)
                offdiag[rows, offset + rows] = False
                metrics["style_target_mean"] = float(style_targets.detach()[offdiag].mean())
                metrics["cue_style_cosine_mean"] = float(cue_cosine.detach()[offdiag].mean())

        if measure_shuffle and bottleneck.size(0) > 1 and not self.command_slot_decode:
            shuffled = bottleneck.roll(1, dims=0)
            with torch.no_grad():
                shuffle_ce = self.decoder(
                    bottleneck=shuffled,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    sentinel_ids=sentinel_ids,
                )
            metrics["shuffle_ce"] = float(shuffle_ce.detach())
            metrics["shuffle_gap"] = float((shuffle_ce - ce).detach())

        metrics.update(self.decoder.gate_magnitudes())
        return {"loss": loss, "metrics": metrics, "bottleneck": bottleneck}

    @torch.no_grad()
    def generate_manuals(
        self,
        sessions: list[list[dict[str, str]]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        dual_decode: bool | None = None,
    ) -> list[dict[str, Any] | None]:
        bottleneck = self.encode(sessions)
        use_dual = self.dual_decode if dual_decode is None else bool(dual_decode)
        if use_dual:
            return self._generate_dual_manuals(
                bottleneck=bottleneck,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        sentinel_ids = self._sentinel_ids(COMMAND_BLOCK_SENTINEL, device=bottleneck.device)
        texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=sentinel_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return [parse_manual(t) for t in texts]

    @torch.no_grad()
    def generate_command_slot_blocks(
        self,
        bottleneck: torch.Tensor,
        *,
        head: str,
        num_slots: int,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_candidates_per_slot: int = 1,
        near_dup_jaccard: float = 0.8,
        noop_retries: int = 0,
        noop_temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Decode one short command per slot; optional per-slot threshold dedup.

        With ``num_candidates_per_slot=1`` (default), greedy/sample once and drop
        near-duplicates (Jaccard ≥ ``near_dup_jaccard``). With N>1 (``diverse_slots``),
        sample N candidates per slot and keep the first that clears the threshold
        vs already-accepted commands in this head (decorrelation, not repulsive MMR).

        Greedy decode is deterministic, so a slot whose mode is ``<NO_COMMAND>`` stays
        empty forever and the manual comes back nearly blank. ``noop_retries`` resamples
        just those rows at ``noop_temperature`` instead of dropping the slot.
        """

        per_sample: list[list[str]] = [[] for _ in range(bottleneck.size(0))]
        sentinel = self._sentinel_ids(COMMAND_SLOT_SENTINEL, device=bottleneck.device)
        n_per = max(1, int(num_candidates_per_slot))
        for slot in range(num_slots):
            slot_ids = torch.full(
                (bottleneck.size(0),),
                self.decoder.command_slot_id(head, slot),
                device=bottleneck.device,
                dtype=torch.long,
            )
            if n_per == 1:
                texts = self.decoder.generate(
                    bottleneck=bottleneck,
                    sentinel_ids=sentinel,
                    slot_ids=slot_ids,
                    max_new_tokens=min(max_new_tokens, 128),
                    temperature=temperature,
                    top_p=top_p,
                )
                pending: list[int] = []
                for row, (commands, raw) in enumerate(
                    zip(per_sample, texts, strict=True)
                ):
                    chosen = select_command_threshold(
                        [clean_generated_command(raw)],
                        commands,
                        near_dup_jaccard=near_dup_jaccard,
                    )
                    if chosen:
                        commands.append(chosen)
                    else:
                        pending.append(row)
                for _ in range(max(0, int(noop_retries))):
                    if not pending:
                        break
                    index = torch.tensor(
                        pending, device=bottleneck.device, dtype=torch.long
                    )
                    retry = self.decoder.generate(
                        bottleneck=bottleneck[index],
                        sentinel_ids=sentinel,
                        slot_ids=slot_ids[index],
                        max_new_tokens=min(max_new_tokens, 128),
                        temperature=max(1e-3, float(noop_temperature)),
                        top_p=top_p,
                    )
                    still: list[int] = []
                    for row, raw in zip(pending, retry, strict=True):
                        chosen = select_command_threshold(
                            [clean_generated_command(raw)],
                            per_sample[row],
                            near_dup_jaccard=near_dup_jaccard,
                        )
                        if chosen:
                            per_sample[row].append(chosen)
                        else:
                            still.append(row)
                    pending = still
                continue
            flat_texts, _, _ = self.decoder.sample_manuals(
                bottleneck=bottleneck,
                sentinel_ids=sentinel,
                slot_ids=slot_ids,
                num_candidates=n_per,
                max_new_tokens=min(max_new_tokens, 128),
                temperature=temperature,
                top_p=top_p,
            )
            for row, commands in enumerate(per_sample):
                cands = [
                    clean_generated_command(flat_texts[row * n_per + i])
                    for i in range(n_per)
                ]
                chosen = select_command_threshold(
                    cands,
                    commands,
                    near_dup_jaccard=near_dup_jaccard,
                )
                if chosen:
                    commands.append(chosen)
        return [{"commands": commands, "examples": []} for commands in per_sample]

    @torch.no_grad()
    def sample_command_slot_manuals(
        self,
        bottleneck: torch.Tensor,
        *,
        num_candidates: int,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[
        list[str],
        list[dict[str, Any] | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, int, int]],
    ]:
        """Sample whole-manual candidates from independent command slots."""

        candidate_count = bottleneck.size(0) * max(1, int(num_candidates))
        sentinel = self._sentinel_ids(COMMAND_SLOT_SENTINEL, device=bottleneck.device)
        commands = {
            head: [[] for _ in range(candidate_count)]
            for head in ("general", "specific", "style")
        }
        token_blocks = []
        mask_blocks = []
        segments: list[tuple[int, int, int]] = []
        offset = 0
        head_slots = (
            ("general", self.general_command_slots),
            ("specific", self.user_specific_command_slots),
            ("style", self.style_command_slots),
        )
        for head, num_slots in head_slots:
            for slot in range(num_slots):
                slot_id = self.decoder.command_slot_id(head, slot)
                texts, token_ids, mask = self.decoder.sample_slot_commands(
                    bottleneck=bottleneck,
                    sentinel_ids=sentinel,
                    slot_ids=torch.full(
                        (bottleneck.size(0),),
                        slot_id,
                        device=bottleneck.device,
                        dtype=torch.long,
                    ),
                    num_candidates=num_candidates,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                kept_mask = mask.clone()
                for row, raw in enumerate(texts):
                    command = clean_generated_command(raw)
                    tokens = set(command.casefold().split())
                    if not command or any(
                        len(tokens & set(existing.casefold().split()))
                        / max(1, len(tokens | set(existing.casefold().split())))
                        >= 0.8
                        for existing in commands[head][row]
                    ):
                        kept_mask[row].zero_()
                        continue
                    commands[head][row].append(command)
                width = token_ids.size(1)
                segments.append((slot_id, offset, width))
                offset += width
                token_blocks.append(token_ids)
                mask_blocks.append(kept_mask)

        manuals = [
            merge_dual_manual_blocks(
                {"commands": commands["general"][row], "examples": []},
                {"commands": commands["specific"][row], "examples": []},
                {"commands": commands["style"][row], "examples": []},
            )
            for row in range(candidate_count)
        ]
        texts = ["\n".join(manual["commands"]) for manual in manuals]
        return (
            texts,
            [manual if manual["commands"] else None for manual in manuals],
            bottleneck,
            torch.cat(token_blocks, dim=1),
            torch.cat(mask_blocks, dim=1),
            segments,
        )

    @torch.no_grad()
    def _generate_dual_manuals(
        self,
        *,
        bottleneck: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> list[dict[str, Any] | None]:
        device = bottleneck.device
        if self.command_slot_decode:
            general_blocks = self.generate_command_slot_blocks(
                bottleneck,
                head="general",
                num_slots=self.general_command_slots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            specific_blocks = self.generate_command_slot_blocks(
                bottleneck,
                head="specific",
                num_slots=self.user_specific_command_slots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            style_blocks = (
                self.generate_command_slot_blocks(
                    bottleneck,
                    head="style",
                    num_slots=self.style_command_slots,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if self.style_command_slots > 0
                else [{"commands": [], "examples": []}] * bottleneck.size(0)
            )
            manuals = [
                merge_dual_manual_blocks(general, specific, style)
                for general, specific, style in zip(
                    general_blocks, specific_blocks, style_blocks, strict=True
                )
            ]
            return [manual if manual.get("commands") else None for manual in manuals]
        general_sentinel = self._sentinel_ids(GENERAL_COMMAND_SENTINEL, device=device)
        specific_sentinel = self._sentinel_ids(USER_SPECIFIC_SENTINEL, device=device)
        general_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=general_sentinel,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        specific_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=specific_sentinel,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        style_texts = None
        if self.style_command_slots > 0 or self.lambda_style_ce > 0:
            style_sentinel = self._sentinel_ids(STYLE_COMMAND_SENTINEL, device=device)
            style_texts = self.decoder.generate(
                bottleneck=bottleneck,
                sentinel_ids=style_sentinel,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        manuals: list[dict[str, Any] | None] = []
        for idx, (general_raw, specific_raw) in enumerate(
            zip(general_texts, specific_texts, strict=True)
        ):
            general = parse_manual(general_raw) or {"commands": [], "examples": []}
            specific = parse_manual(specific_raw) or {"commands": [], "examples": []}
            style = (
                (parse_manual(style_texts[idx]) or {"commands": [], "examples": []})
                if style_texts is not None
                else {"commands": [], "examples": []}
            )
            merged = merge_dual_manual_blocks(general, specific, style)
            manuals.append(merged if merged.get("commands") else None)
        return manuals

    @torch.no_grad()
    def generate_dual_manual_texts(
        self,
        sessions: list[list[dict[str, str]]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]]:
        """Decode general + user-specific + style blocks and return merged render text."""

        from cue_training.refinement.simulator import render_dual_manual

        bottleneck = self.encode(sessions)
        device = bottleneck.device
        empty = {"commands": [], "examples": []}
        if self.command_slot_decode:
            general_blocks = self.generate_command_slot_blocks(
                bottleneck,
                head="general",
                num_slots=self.general_command_slots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            specific_blocks = self.generate_command_slot_blocks(
                bottleneck,
                head="specific",
                num_slots=self.user_specific_command_slots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            style_blocks = (
                self.generate_command_slot_blocks(
                    bottleneck,
                    head="style",
                    num_slots=self.style_command_slots,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if self.style_command_slots > 0
                else [dict(empty) for _ in range(bottleneck.size(0))]
            )
            return [
                (
                    general,
                    specific,
                    style,
                    render_dual_manual(general, specific, style),
                )
                for general, specific, style in zip(
                    general_blocks, specific_blocks, style_blocks, strict=True
                )
            ]
        general_sentinel = self._sentinel_ids(GENERAL_COMMAND_SENTINEL, device=device)
        specific_sentinel = self._sentinel_ids(USER_SPECIFIC_SENTINEL, device=device)
        general_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=general_sentinel,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        specific_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=specific_sentinel,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        style_texts = None
        if self.style_command_slots > 0 or self.lambda_style_ce > 0:
            style_sentinel = self._sentinel_ids(STYLE_COMMAND_SENTINEL, device=device)
            style_texts = self.decoder.generate(
                bottleneck=bottleneck,
                sentinel_ids=style_sentinel,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]] = []
        for idx, (general_raw, specific_raw) in enumerate(
            zip(general_texts, specific_texts, strict=True)
        ):
            general = parse_manual(general_raw) or dict(empty)
            specific = parse_manual(specific_raw) or dict(empty)
            style = (
                (parse_manual(style_texts[idx]) or dict(empty))
                if style_texts is not None
                else dict(empty)
            )
            rows.append(
                (general, specific, style, render_dual_manual(general, specific, style))
            )
        return rows

    def config_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.config_dict(),
            "decoder": self.decoder.config,
            "dual_decode": self.dual_decode,
            "lambda_general_ce": self.lambda_general_ce,
            "lambda_user_specific_ce": self.lambda_user_specific_ce,
            "lambda_style_ce": self.lambda_style_ce,
            "command_slot_decode": self.command_slot_decode,
            "general_command_slots": self.general_command_slots,
            "user_specific_command_slots": self.user_specific_command_slots,
            "style_command_slots": self.style_command_slots,
            "slot_embedding_dim": int(
                self.decoder.config.get("slot_embedding_dim", 64)
            ),
            "command_noop_weight": self.command_noop_weight,
            "lambda_dual_separation": self.lambda_dual_separation,
            "dual_separation_margin": self.dual_separation_margin,
        }


def build_unified_model(
    cfg: dict[str, Any], *, device: str = "cpu", allow_meta_to_empty: bool = False
) -> UnifiedModel:
    enc_cfg = cfg.get("encoder", {})
    dec_cfg = cfg.get("decoder", {})
    encoder = UnifiedEncoder(
        user_model_name=enc_cfg.get("user_model_name", "intfloat/e5-base-v2"),
        system_model_name=enc_cfg.get("system_model_name", "intfloat/e5-base-v2"),
        bottleneck_dim=int(enc_cfg.get("bottleneck_dim", 1024)),
        session_dim=int(enc_cfg.get("session_dim", 1024)),
        session_layers=int(enc_cfg.get("session_layers", 4)),
        session_heads=int(enc_cfg.get("session_heads", 8)),
        max_turns=int(enc_cfg.get("max_turns", 64)),
        max_length=int(enc_cfg.get("max_length", 256)),
        query_prefix=str(enc_cfg.get("query_prefix", "search_query: ")),
        passage_prefix=str(enc_cfg.get("passage_prefix", "search_document: ")),
    )
    dtype_name = dec_cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    use_command_slots = bool(cfg.get("command_slot_decode", False))
    general_slots = int(cfg.get("general_command_slots", 5))
    specific_slots = int(cfg.get("user_specific_command_slots", 5))
    style_slots = int(cfg.get("style_command_slots", 0))
    decoder = UnifiedDecoder.build(
        model_name=dec_cfg.get("model_name", "Qwen/Qwen3-0.6B-Base"),
        bottleneck_dim=int(enc_cfg.get("bottleneck_dim", 1024)),
        num_persona_tokens=int(dec_cfg.get("num_persona_tokens", 16)),
        insert_every=int(dec_cfg.get("insert_every", 1)),
        num_heads=int(dec_cfg.get("num_heads", 8)),
        general_command_slots=general_slots if use_command_slots else 0,
        user_specific_command_slots=specific_slots if use_command_slots else 0,
        style_command_slots=style_slots if use_command_slots else 0,
        slot_embedding_dim=int(
            cfg.get("slot_embedding_dim", dec_cfg.get("slot_embedding_dim", 64))
        ),
        dtype=dtype,
        freeze_lm=bool(dec_cfg.get("freeze_lm", False)),
    )
    model = UnifiedModel(
        encoder,
        decoder,
        target_max_tokens=int(cfg.get("target_max_tokens", 1024)),
        command_cache_size=int(cfg.get("command_cache_size", 20_000)),
        dual_decode=bool(cfg.get("dual_decode", False)),
        lambda_general_ce=float(cfg.get("lambda_general_ce", 0.5)),
        lambda_user_specific_ce=float(cfg.get("lambda_user_specific_ce", 0.5)),
        lambda_style_ce=float(cfg.get("lambda_style_ce", 0.0)),
        command_slot_decode=use_command_slots,
        general_command_slots=general_slots,
        user_specific_command_slots=specific_slots,
        style_command_slots=style_slots,
        command_noop_weight=float(cfg.get("command_noop_weight", 0.25)),
        lambda_dual_separation=float(cfg.get("lambda_dual_separation", 0.0)),
        dual_separation_margin=float(cfg.get("dual_separation_margin", 0.2)),
    )
    try:
        return model.to(device)
    except NotImplementedError:
        # Some submodule was meta-initialized (transformers low_cpu_mem_usage), so a plain
        # .to(device) fails ("Cannot copy out of meta tensor"). For inference the caller reloads
        # a COMPLETE checkpoint right after (n_missing=0), so materializing empties on-device and
        # letting load_state_dict fill real weights is correct. Guarded so training never zeroes
        # its from_pretrained base weights.
        if allow_meta_to_empty:
            return model.to_empty(device=device)
        raise


def manuals_to_targets(manuals: list[dict[str, Any]]) -> list[str]:
    return [canonicalize_profile(m) for m in manuals]
