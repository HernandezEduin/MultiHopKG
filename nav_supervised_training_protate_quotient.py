"""Supervised pRotatE navigation on the model's pi-periodic quotient space."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

import nav_supervised_training as base
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.utils.convenience import get_embeddings_from_indices
from temporary_patches.protate_navigation import enable_protate_navigation_patches
from temporary_patches.protate_policy import enable_protate_policy_patch
from temporary_patches.protate_quotient import (
    enable_protate_quotient_navigation_patch,
    relation_to_quotient_action,
)
from temporary_patches.protate_supervision import _adapter_loss, _expand_rollouts


def _sigma_loss(sigma, sigma_scalar, expected_sigma, mask=None):
    target = torch.full_like(sigma, expected_sigma)
    if mask is None:
        return sigma_scalar * F.mse_loss(sigma, target)
    while mask.ndim < sigma.ndim:
        mask = mask.unsqueeze(-1)
    loss = (sigma - target).pow(2) * mask
    denom = mask.expand_as(loss).sum().clamp_min(1.0)
    return sigma_scalar * loss.sum() / denom


def _masked_mse(prediction, target, mask):
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    loss = (prediction - target).pow(2) * mask
    denom = mask.expand_as(loss).sum().clamp_min(1.0)
    return loss.sum() / denom


def _pad_paths(paths, reasoning_hops, device):
    """Pad ragged QA paths to the fixed reasoning budget.

    Each real step stores (head, relation, tail). Padding rows are zeros but are
    never read as gold relations because step masks are derived from each row's
    question hop count.
    """
    batch = len(paths)
    padded = torch.zeros((batch, reasoning_hops, 3), dtype=torch.long, device=device)
    for i, path in enumerate(paths):
        if len(path) > reasoning_hops:
            raise ValueError(
                f"Question path has {len(path)} hops but reasoning budget is {reasoning_hops}"
            )
        if len(path):
            padded[i, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    return padded


def _adapter_loss_mixed(env, adapter_out, source_ent, paths_t, question_hops):
    """Adapter ablation for mixed-hop batches using only real relation steps."""
    head_emb = get_embeddings_from_indices(
        env.knowledge_graph.entity_embedding,
        torch.tensor(source_ent, dtype=torch.long, device=paths_t.device),
    )
    rel_sum = torch.zeros_like(head_emb)
    for step in range(paths_t.shape[1]):
        relation = get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding,
            paths_t[:, step, 1],
        )
        active = (question_hops > step).to(relation.dtype).unsqueeze(-1)
        rel_sum = rel_sum + relation * active
    rel_mean = rel_sum / question_hops.clamp_min(1).to(rel_sum.dtype).unsqueeze(-1)
    query_emb = torch.cat([head_emb, rel_mean], dim=-1)
    query_emb = _expand_rollouts(env, query_emb)
    return F.mse_loss(adapter_out, query_emb)


def supervise_models_protate_quotient(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    steps_in_episode: int,
    hops: List[int] = None,
    paths=None,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03,
    noise_scale: float = 0.5,
):
    """Supervise mixed question-hop lengths under one fixed reasoning budget.

    ``hops`` is the per-question semantic/path length. ``steps_in_episode`` is
    the reasoning budget used by the policy/environment and normally equals the
    maximum supported hop count. For a question that terminates before the
    reasoning budget, remaining supervised steps use the zero quotient action,
    i.e. continuous NO-OP, so the policy learns to remain at the reached answer.
    """
    del noise_scale
    assert env.knowledge_graph.model_name == "pRotatE"
    if hops is None or paths is None:
        raise ValueError("Mixed-hop quotient supervision requires Hops and Paths")

    reasoning_hops = int(steps_in_episode)
    if reasoning_hops <= 0:
        raise ValueError("steps_in_episode must be positive")

    device = question_embeddings.device
    question_hops = torch.tensor(hops, dtype=torch.long, device=device)
    if torch.any(question_hops < 1):
        raise ValueError("Question Hops values must be >= 1")
    if torch.any(question_hops > reasoning_hops):
        raise ValueError(
            "Question Hops values cannot exceed the reasoning hop budget "
            f"({reasoning_hops})"
        )

    paths_t = _pad_paths(paths, reasoning_hops, device)
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    state = obs.state
    position = obs.kge_cur_pos
    adapter_out = env.q_projected
    loss = torch.tensor(0.0, device=device)

    for step in range(reasoning_hops):
        active = question_hops > step
        relation = get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding, paths_t[:, step, 1]
        )
        gold_target = relation_to_quotient_action(env, relation)
        target = torch.where(active.unsqueeze(-1), gold_target, torch.zeros_like(gold_target))
        target = _expand_rollouts(env, target)

        _, _, _, mu, sigma = nav_agent(state)
        # Every example is supervised at every reasoning step: real relation
        # before the question ends, zero/NO-OP afterwards.
        all_examples = torch.ones(
            mu.shape[:-1], dtype=mu.dtype, device=mu.device
        )
        loss = loss + _masked_mse(mu, target, all_examples)
        loss = loss + _sigma_loss(
            sigma, sigma_scalar, expected_sigma, mask=all_examples
        )

        previous = position
        position = env.knowledge_graph.flexible_forward(position, target)
        if env.add_transition_state:
            state = torch.cat([env.q_projected, previous, target, position], dim=-1)
        else:
            state = torch.cat([env.q_projected, position], dim=-1)

    if adapter_scalar == 0.0:
        return loss
    adapter_loss = _adapter_loss_mixed(
        env, adapter_out, source_ent, paths_t, question_hops
    )
    return loss + adapter_scalar * adapter_loss


def install_temporary_patches() -> None:
    enable_protate_navigation_patches()
    enable_protate_quotient_navigation_patch()
    enable_protate_policy_patch()
    base.supervise_models = supervise_models_protate_quotient


if __name__ == "__main__":
    install_temporary_patches()
    base.main()
