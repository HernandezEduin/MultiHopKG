"""Temporary entry point for supervised pRotatE navigation experiments.

This file intentionally keeps ``nav_supervised_training.py`` unchanged. It
installs the pRotatE navigation compatibility layer before any KGE instances
are created, then relaxes the supervised-training dispatch so pRotatE can use
the same single-hop and multi-hop supervision routines as TransE.

Run this file with the same arguments/configuration normally supplied to
``nav_supervised_training.py``.
"""

from __future__ import annotations

from typing import List

import torch

import nav_supervised_training as base
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from temporary_patches.protate_navigation import enable_protate_navigation_patches


def supervise_models_protate_compatible(
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
    """Dispatch supervised navigation for TransE and temporary pRotatE support."""

    if hops is not None:
        assert all(
            steps_in_episode >= h for h in hops
        ), "All Hops values must be smaller than or equal to steps_in_episode"
        assert all(
            h == hops[0] for h in hops
        ), "All Hops values must be equal to the first value in hops"
        assert paths is not None, "Paths should be provided when hops are specified."

    supported_models = {"TransE", "pRotatE"}
    assert env.knowledge_graph.model_name in supported_models, (
        f"Unsupported KGE model: {env.knowledge_graph.model_name}. "
        f"Temporary supervised navigation supports: {sorted(supported_models)}"
    )

    if hops is None or all(h == 1 for h in hops):
        return base.single_hop_supervision(
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

    return base.multihop_supervision(
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
    """Install the pRotatE geometry patches and supervised dispatcher."""

    enable_protate_navigation_patches()
    base.supervise_models = supervise_models_protate_compatible


if __name__ == "__main__":
    install_temporary_patches()
    base.main()
