"""Temporary pRotatE supervised entry point with continuous gold-relation states.

This variant keeps the pretrained relation phase as the supervised action label,
but for multi-hop warm-up the next policy state is produced by applying that
gold relation action continuously through pRotatE space. It therefore matches
inference-state geometry more closely than gold-entity teacher forcing while
avoiding compounding the policy's own prediction error during supervision.
"""

from __future__ import annotations

from typing import List

import torch

import nav_supervised_training as base
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from temporary_patches.protate_navigation import enable_protate_navigation_patches
from temporary_patches.protate_policy import enable_protate_policy_patch
from temporary_patches.protate_supervision import (
    multihop_supervision_relation_target_continuous,
    single_hop_supervision_relation_target,
)


def supervise_models_protate_continuous(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    steps_in_episode: int,
    hops: List[int] = None,
    paths: List[List[int]] = None,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03,
    noise_scale: float = 0.5,
):
    if hops is not None:
        assert all(steps_in_episode >= h for h in hops)
        assert all(h == hops[0] for h in hops)
        assert paths is not None

    assert env.knowledge_graph.model_name == "pRotatE"

    if hops is None or all(h == 1 for h in hops):
        return single_hop_supervision_relation_target(
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

    return multihop_supervision_relation_target_continuous(
        nav_agent=nav_agent,
        env=env,
        question_embeddings=question_embeddings,
        source_ent=source_ent,
        answer_id=answer_id,
        hops=hops[0],
        paths=paths,
        adapter_scalar=adapter_scalar,
        sigma_scalar=sigma_scalar,
        expected_sigma=expected_sigma,
        noise_scale=noise_scale,
    )


def install_temporary_patches() -> None:
    enable_protate_navigation_patches()
    enable_protate_policy_patch()
    base.supervise_models = supervise_models_protate_continuous


if __name__ == "__main__":
    install_temporary_patches()
    base.main()
