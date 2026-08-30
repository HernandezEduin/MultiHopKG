"""Run checkpoint diagnostics consistently in pRotatE quotient space.

For mixed-hop QA checkpoints, ``--question_hops N`` filters each split to one
question hop count and diagnoses exactly N reasoning steps. This avoids padding
shorter annotated paths with artificial relations while keeping the checkpoint's
policy weights unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diagnostics.protate_policy_checkpoint as base
from temporary_patches.protate_navigation import enable_protate_navigation_patches
from temporary_patches.protate_policy import enable_protate_policy_patch
from temporary_patches.protate_quotient import (
    canonicalize_raw_state_pi,
    enable_protate_quotient_navigation_patch,
    relation_to_quotient_action,
)


def _teacher_forced_state_quotient(
    env,
    prev_position: torch.Tensor,
    target_action: torch.Tensor,
    gold_tail: torch.Tensor,
) -> torch.Tensor:
    gold_tail = canonicalize_raw_state_pi(env.knowledge_graph, gold_tail)
    if env.add_transition_state:
        return torch.cat(
            [env.q_projected, prev_position, target_action, gold_tail], dim=-1
        )
    return torch.cat([env.q_projected, gold_tail], dim=-1)


def _extract_question_hops() -> int | None:
    """Consume the quotient-wrapper-only hop filter before base argparse runs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--question_hops",
        type=int,
        default=None,
        help="For mixed-hop checkpoints, diagnose only questions with this Hops value.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known.question_hops


def _install_hop_filter(question_hops: int) -> None:
    if question_hops <= 0:
        raise ValueError("--question_hops must be positive")

    original_build_environment = base.build_environment

    def build_environment_filtered(config, device):
        # The checkpoint policy itself is independent of episode length. For
        # diagnostics, reason only for the semantic hop count being evaluated.
        config["num_rollout_steps"] = question_hops
        env, nav_agent, splits = original_build_environment(config, device)

        filtered = {}
        for split_name, df in splits.items():
            if "Hops" not in df.columns:
                raise ValueError(
                    "--question_hops requires the cached QA data to contain a Hops column"
                )
            hops_numeric = df["Hops"].astype(int)
            filtered[split_name] = df[hops_numeric == question_hops].reset_index(
                drop=True
            )

        return env, nav_agent, filtered

    base.build_environment = build_environment_filtered


if __name__ == "__main__":
    question_hops = _extract_question_hops()

    enable_protate_navigation_patches()
    enable_protate_quotient_navigation_patch()
    enable_protate_policy_patch()

    base.relation_to_navigation_action = relation_to_quotient_action
    base._teacher_forced_state = _teacher_forced_state_quotient

    if question_hops is not None:
        _install_hop_filter(question_hops)

    base.main()
