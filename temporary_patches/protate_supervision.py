"""pRotatE-specific supervised navigation targets.

The original supervised warm-up derives an action label independently for every
edge as ``difference(current_entity, gold_tail)``. That is well matched to
TransE, but pRotatE is pi-periodic under ``abs(sin(.))``: phase-equivalent
solutions can be far apart in ordinary MSE coordinates. On real Kinship data,
this makes endpoint-derived action labels highly variable even for the same
relation.

For pRotatE, use the pretrained relation phase itself as the semantic action
label. During multi-hop warm-up, teacher-force the annotated gold tail into the
next policy state. This separates two questions:

* can the question-conditioned policy predict the correct relation rotation?
* can pretrained pRotatE relation rotations compose perfectly in embedding
  space?

The latter is evaluated separately; it should not inject approximation error
into the supervised labels/states.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.utils.convenience import get_embeddings_from_indices


def relation_to_navigation_action(
    env: ITLGraphEnvironment,
    relation_embedding: torch.Tensor,
) -> torch.Tensor:
    """Convert raw pRotatE relation embeddings to normalized policy actions."""

    relation_rad = env.knowledge_graph.denormalize_embedding(relation_embedding)
    relation_rad = torch.atan2(torch.sin(relation_rad), torch.cos(relation_rad))
    return relation_rad / math.pi


def _expand_rollouts(env: ITLGraphEnvironment, tensor: torch.Tensor) -> torch.Tensor:
    if env.num_rollouts > 0:
        return tensor.unsqueeze(1).expand(-1, env.num_rollouts, -1)
    return tensor


def _teacher_forced_state(
    env: ITLGraphEnvironment,
    prev_position: torch.Tensor,
    target_action: torch.Tensor,
    gold_tail: torch.Tensor,
) -> torch.Tensor:
    """Build the next policy state using the annotated gold entity position."""

    if env.add_transition_state:
        return torch.cat(
            [env.q_projected, prev_position, target_action, gold_tail], dim=-1
        )
    return torch.cat([env.q_projected, gold_tail], dim=-1)


def _adapter_loss(
    env: ITLGraphEnvironment,
    adapter_out: torch.Tensor,
    source_ent: List[int],
    paths: torch.Tensor,
) -> torch.Tensor:
    """Retain the base adapter objective for a controlled target ablation."""

    head_emb = get_embeddings_from_indices(
        env.knowledge_graph.entity_embedding,
        torch.tensor(source_ent, dtype=torch.int, device=paths.device),
    )

    rel_embs = []
    for step in range(paths.shape[1]):
        rel_embs.append(
            get_embeddings_from_indices(
                env.knowledge_graph.relation_embedding,
                paths[:, step, 1],
            )
        )
    rel_emb = torch.mean(torch.stack(rel_embs, dim=0), dim=0)

    query_emb = torch.cat([head_emb, rel_emb], dim=-1)
    query_emb = _expand_rollouts(env, query_emb)
    return F.mse_loss(adapter_out, query_emb)


def single_hop_supervision_relation_target(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    paths,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03,
    noise_scale: float = 0.5,
):
    """Supervise one-hop pRotatE navigation with the pretrained relation phase."""

    del noise_scale  # Endpoint noise is intentionally not part of relation labels.
    device = question_embeddings.device
    paths_t = torch.tensor(paths, dtype=torch.int, device=device)

    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    adapter_out = env.q_projected

    relation_embedding = get_embeddings_from_indices(
        env.knowledge_graph.relation_embedding,
        paths_t[:, 0, 1],
    )
    target_action = relation_to_navigation_action(env, relation_embedding)
    target_action = _expand_rollouts(env, target_action)

    _, _, _, mu, sigma = nav_agent(obs.state)
    policy_loss = F.mse_loss(mu, target_action)
    policy_loss = policy_loss + sigma_scalar * F.mse_loss(
        sigma, torch.full_like(sigma, expected_sigma)
    )

    adapter_loss = _adapter_loss(env, adapter_out, source_ent, paths_t)
    return policy_loss + adapter_scalar * adapter_loss


def multihop_supervision_relation_target(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    hops: int,
    paths,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03,
    noise_scale: float = 0.5,
):
    """Supervise pRotatE relation actions while teacher-forcing gold path states."""

    del noise_scale  # Relation labels are fixed by the pretrained KGE.
    device = question_embeddings.device
    paths_t = torch.tensor(paths, dtype=torch.int, device=device)

    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    state = obs.state
    current_position = obs.kge_cur_pos
    adapter_out = env.q_projected

    policy_loss = torch.tensor(0.0, device=device)

    for step in range(hops):
        path_ids = paths_t[:, step, :]
        relation_embedding = get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding,
            path_ids[:, 1],
        )
        target_action = relation_to_navigation_action(env, relation_embedding)

        gold_tail = get_embeddings_from_indices(
            env.knowledge_graph.entity_embedding,
            path_ids[:, 2],
        )

        target_action = _expand_rollouts(env, target_action)
        gold_tail = _expand_rollouts(env, gold_tail)

        _, _, _, mu, sigma = nav_agent(state)
        policy_loss = policy_loss + F.mse_loss(mu, target_action)
        policy_loss = policy_loss + sigma_scalar * F.mse_loss(
            sigma, torch.full_like(sigma, expected_sigma)
        )

        # Follow the annotated path exactly during supervised warm-up. The
        # policy is trained to predict relation rotations, while the next state
        # remains the true intermediate entity supplied by the dataset.
        state = _teacher_forced_state(
            env,
            prev_position=current_position,
            target_action=target_action,
            gold_tail=gold_tail,
        )
        current_position = gold_tail

    adapter_loss = _adapter_loss(env, adapter_out, source_ent, paths_t)
    return policy_loss + adapter_scalar * adapter_loss
