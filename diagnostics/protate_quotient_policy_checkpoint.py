"""Run checkpoint diagnostics consistently in pRotatE quotient space."""

from __future__ import annotations

import torch

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


if __name__ == "__main__":
    enable_protate_navigation_patches()
    enable_protate_quotient_navigation_patch()
    enable_protate_policy_patch()
    base.relation_to_navigation_action = relation_to_quotient_action
    base._teacher_forced_state = _teacher_forced_state_quotient
    base.main()
