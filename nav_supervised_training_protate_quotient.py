"""Supervised pRotatE navigation on the model's pi-periodic quotient space."""

from __future__ import annotations

from typing import List

import numpy as np
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


def _sigma_loss(sigma, sigma_scalar, expected_sigma):
    return sigma_scalar * F.mse_loss(
        sigma, torch.full_like(sigma, expected_sigma)
    )


def single_hop_quotient(
    nav_agent,
    env,
    question_embeddings,
    source_ent,
    answer_id,
    paths,
    adapter_scalar=0.5,
    sigma_scalar=0.1,
    expected_sigma=0.03,
    noise_scale=0.5,
):
    del noise_scale
    device = question_embeddings.device
    paths_t = torch.tensor(np.asarray(paths), dtype=torch.long, device=device)
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    adapter_out = env.q_projected

    relation = get_embeddings_from_indices(
        env.knowledge_graph.relation_embedding, paths_t[:, 0, 1]
    )
    target = _expand_rollouts(env, relation_to_quotient_action(env, relation))
    _, _, _, mu, sigma = nav_agent(obs.state)
    loss = F.mse_loss(mu, target) + _sigma_loss(
        sigma, sigma_scalar, expected_sigma
    )
    return loss + adapter_scalar * _adapter_loss(
        env, adapter_out, source_ent, paths_t
    )


def multihop_quotient(
    nav_agent,
    env,
    question_embeddings,
    source_ent,
    answer_id,
    hops,
    paths,
    adapter_scalar=0.5,
    sigma_scalar=0.1,
    expected_sigma=0.03,
    noise_scale=0.5,
):
    del noise_scale
    device = question_embeddings.device
    paths_t = torch.tensor(np.asarray(paths), dtype=torch.long, device=device)
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    state = obs.state
    position = obs.kge_cur_pos
    adapter_out = env.q_projected
    loss = torch.tensor(0.0, device=device)

    for step in range(hops):
        relation = get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding, paths_t[:, step, 1]
        )
        target = _expand_rollouts(env, relation_to_quotient_action(env, relation))
        _, _, _, mu, sigma = nav_agent(state)
        loss = loss + F.mse_loss(mu, target)
        loss = loss + _sigma_loss(sigma, sigma_scalar, expected_sigma)

        previous = position
        position = env.knowledge_graph.flexible_forward(position, target)
        if env.add_transition_state:
            state = torch.cat(
                [env.q_projected, previous, target, position], dim=-1
            )
        else:
            state = torch.cat([env.q_projected, position], dim=-1)

    return loss + adapter_scalar * _adapter_loss(
        env, adapter_out, source_ent, paths_t
    )


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
    assert env.knowledge_graph.model_name == "pRotatE"
    if hops is not None:
        assert all(steps_in_episode >= h for h in hops)
        assert all(h == hops[0] for h in hops)
        assert paths is not None

    kwargs = dict(
        nav_agent=nav_agent,
        env=env,
        question_embeddings=question_embeddings,
        source_ent=source_ent,
        answer_id=answer_id,
        paths=paths,
        adapter_scalar=adapter_scalar,
        sigma_scalar=sigma_scalar,
        expected_sigma=expected_sigma,
        noise_scale=noise_scale,
    )
    if hops is None or all(h == 1 for h in hops):
        return single_hop_quotient(**kwargs)
    return multihop_quotient(hops=hops[0], **kwargs)


def install_temporary_patches() -> None:
    enable_protate_navigation_patches()
    enable_protate_quotient_navigation_patch()
    enable_protate_policy_patch()
    base.supervise_models = supervise_models_protate_quotient


if __name__ == "__main__":
    install_temporary_patches()
    base.main()
