"""Temporary quotient-space navigation helpers for pRotatE.

pRotatE scores residuals through ``abs(sin(delta))``. Therefore phases separated
by pi are equivalent to the KGE even though they are numerically different. The
continuous policy should not have to learn that equivalence from data.

This patch chooses a unique representative for each pi-equivalence class:

    theta_q = 0.5 * atan2(sin(2 theta), cos(2 theta))

so ``theta_q`` lies in [-pi/2, pi/2]. In normalized policy coordinates this is
[-0.5, 0.5]. The KGE's scoring and ANN distance remain unchanged.
"""

from __future__ import annotations

from typing import List

import torch

from multihopkg.environments import Observation
from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment


_ORIGINAL_ENV_RESET = ITLGraphEnvironment.reset
_INSTALLED = False


def canonicalize_phase_pi(phase: torch.Tensor) -> torch.Tensor:
    """Return the canonical representative modulo pi in [-pi/2, pi/2]."""

    return 0.5 * torch.atan2(torch.sin(2.0 * phase), torch.cos(2.0 * phase))


def canonicalize_action_pi(action: torch.Tensor) -> torch.Tensor:
    """Canonicalize normalized pRotatE actions into [-0.5, 0.5]."""

    return canonicalize_phase_pi(action * torch.pi) / torch.pi


def canonicalize_raw_state_pi(
    model: KGEModel,
    raw_state: torch.Tensor,
) -> torch.Tensor:
    """Canonicalize a stored pRotatE entity/state embedding modulo pi."""

    phase = model.denormalize_embedding(raw_state)
    return model.normalize_embedding(canonicalize_phase_pi(phase))


def difference_phase_quotient(
    self: KGEModel,
    head: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    """Return the pRotatE displacement on its pi-periodic quotient space."""

    head_phase = self.denormalize_embedding(head)
    tail_phase = self.denormalize_embedding(tail)
    return canonicalize_phase_pi(tail_phase - head_phase) / torch.pi


def flexible_forward_protate_quotient(
    self: KGEModel,
    cur_states: torch.Tensor,
    cur_actions: torch.Tensor,
) -> torch.Tensor:
    """Apply an action and return a canonical pi-equivalent pRotatE state."""

    head_phase = self.denormalize_embedding(cur_states)
    tail_phase = canonicalize_phase_pi(head_phase + cur_actions * torch.pi)
    return self.normalize_embedding(tail_phase)


def relation_to_quotient_action(
    env: ITLGraphEnvironment,
    relation_embedding: torch.Tensor,
) -> torch.Tensor:
    """Convert raw pretrained pRotatE relations to quotient policy actions."""

    relation_phase = env.knowledge_graph.denormalize_embedding(relation_embedding)
    return canonicalize_phase_pi(relation_phase) / torch.pi


def _reset_protate_quotient(
    self: ITLGraphEnvironment,
    initial_states_info: torch.Tensor,
    answer_ent: List[int],
    source_ent: List[int] = None,
    warmup: bool = True,
) -> Observation:
    """Run the normal reset then canonicalize the pRotatE policy position."""

    observation = _ORIGINAL_ENV_RESET(
        self,
        initial_states_info,
        answer_ent,
        source_ent=source_ent,
        warmup=warmup,
    )

    if self.knowledge_graph.model_name != "pRotatE":
        return observation

    self.current_position = canonicalize_raw_state_pi(
        self.knowledge_graph, self.current_position
    )

    if self.add_transition_state:
        zero_action = torch.zeros(
            *self.current_position.shape[:-1],
            self.action_dim,
            dtype=self.current_position.dtype,
            device=self.current_position.device,
        )
        projected_state = torch.cat(
            [
                self.q_projected,
                self.current_position,
                zero_action,
                self.current_position,
            ],
            dim=-1,
        )
    else:
        projected_state = torch.cat(
            [self.q_projected, self.current_position], dim=-1
        )

    return Observation(
        state=projected_state,
        kge_cur_pos=self.current_position,
        kge_prev_pos=torch.zeros_like(self.current_position.detach()),
        kge_action=torch.zeros(
            self.action_dim,
            dtype=self.current_position.dtype,
            device=self.current_position.device,
        ),
    )


def enable_protate_quotient_navigation_patch() -> None:
    """Install quotient geometry before constructing pRotatE model instances."""

    global _INSTALLED
    if _INSTALLED:
        return

    # KGEModel caches these bound methods in __init__, so patch before model
    # construction. The base pRotatE navigation patch should be installed first
    # so ANN/absolute-distance semantics already match abs(sin(.)) geometry.
    KGEModel.difference_phase = difference_phase_quotient
    KGEModel.flexible_forward_protate = flexible_forward_protate_quotient
    ITLGraphEnvironment.reset = _reset_protate_quotient
    _INSTALLED = True
