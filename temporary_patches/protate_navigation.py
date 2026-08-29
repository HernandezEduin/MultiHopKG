"""Temporary pRotatE navigation compatibility patches.

These patches intentionally leave the pretrained pRotatE link-prediction
implementation untouched. They only adapt the continuous-navigation interface
so that the policy, environment, and nearest-neighbor evaluation agree on the
same action coordinates and cyclic geometry.

Action convention
-----------------
The navigation policy emits values in [-1, 1] because both its mean and sampled
outputs are passed through tanh. For pRotatE we interpret one action component
as

    action = shortest_phase_displacement / pi

so -1 and +1 correspond to -pi and +pi radians respectively.

This avoids the previous mismatch where supervised targets were expressed in
radians ([-pi, pi]) while the policy could only represent [-1, 1], and where
those radian targets were subsequently denormalized a second time by the KGE
transition function.
"""

from __future__ import annotations

from typing import Tuple

import torch

from multihopkg.emb.operations import angular_difference, normalize_angle_smooth
from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.vector_search import ANN_IndexMan_pRotatE


def difference_phase_navigation(
    self: KGEModel,
    head: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    """Return the shortest pRotatE phase displacement in policy coordinates.

    ``head`` and ``tail`` are stored KGE embeddings. They are first converted to
    radians, the shortest 2*pi-periodic displacement is computed, and the result
    is divided by pi so the target lies in [-1, 1].
    """

    head_rad = self.denormalize_embedding(head)
    tail_rad = self.denormalize_embedding(tail)
    delta_rad = angular_difference(
        head_rad,
        tail_rad,
        smooth=torch.is_grad_enabled(),
    )
    return delta_rad / torch.pi


def flexible_forward_protate_navigation(
    self: KGEModel,
    cur_states: torch.Tensor,
    cur_actions: torch.Tensor,
) -> torch.Tensor:
    """Apply normalized policy actions directly as pRotatE phase rotations.

    ``cur_actions`` are navigation actions in [-1, 1], not raw pRotatE relation
    embeddings. Converting them with ``action * pi`` therefore gives the phase
    displacement exactly once; calling ``denormalize_embedding`` on the action
    here would incorrectly apply the KGE embedding scale a second time.
    """

    head_rad = self.denormalize_embedding(cur_states)
    rotation_rad = cur_actions * torch.pi
    tail_rad = normalize_angle_smooth(head_rad + rotation_rad)
    return self.normalize_embedding(tail_rad)


def protate_navigation_distance(
    target_rad: torch.Tensor,
    entity_rad: torch.Tensor,
) -> torch.Tensor:
    """Distance matching the periodicity of the pRotatE scoring function.

    pRotatE ranks triples using abs(sin(h + r - t)); consequently a residual and
    that residual plus pi are equivalent for scoring. This distance uses the
    same abs(sin(.)) geometry instead of ordinary 2*pi angular distance.
    """

    return torch.abs(torch.sin(target_rad - entity_rad)).sum(dim=-1)


def protate_ann_search(
    self: ANN_IndexMan_pRotatE,
    target_embeddings: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pRotatE nearest-neighbor search with rollout support and three outputs.

    Supports [D], [B, D], and [B, R, D] navigation tensors. The final return
    shape mirrors the input leading dimensions and is compatible with the
    environment's expected ``(embeddings, indices, distances)`` interface.
    """

    if not isinstance(target_embeddings, torch.Tensor):
        raise TypeError("Target embeddings must be a torch.Tensor")
    if topk < 1:
        raise ValueError("topk must be at least 1")

    original_device = target_embeddings.device
    original_shape = target_embeddings.shape

    if target_embeddings.dim() == 1:
        leading_shape = ()
        flat_targets = target_embeddings.reshape(1, -1)
    elif target_embeddings.dim() == 2:
        leading_shape = (target_embeddings.shape[0],)
        flat_targets = target_embeddings
    elif target_embeddings.dim() == 3:
        leading_shape = target_embeddings.shape[:2]
        flat_targets = target_embeddings.reshape(-1, target_embeddings.shape[-1])
    else:
        raise ValueError(
            "Target embeddings must have shape [D], [B, D], or [B, R, D]"
        )

    flat_targets = flat_targets.detach().cpu().float()
    target_rad = flat_targets / (self.embedding_range / torch.pi)
    entity_rad = self.embedding_vectors.squeeze(0)

    distances = protate_navigation_distance(
        target_rad.unsqueeze(1),
        entity_rad.unsqueeze(0),
    )
    distances, indices = torch.topk(
        distances,
        k=min(topk, distances.shape[-1]),
        dim=-1,
        largest=False,
        sorted=True,
    )

    resulting_embeddings = entity_rad[indices] * (self.embedding_range / torch.pi)

    if leading_shape:
        resulting_embeddings = resulting_embeddings.reshape(
            *leading_shape, resulting_embeddings.shape[-2], resulting_embeddings.shape[-1]
        )
        indices = indices.reshape(*leading_shape, indices.shape[-1])
        distances = distances.reshape(*leading_shape, distances.shape[-1])
    else:
        resulting_embeddings = resulting_embeddings.squeeze(0)
        indices = indices.squeeze(0)
        distances = distances.squeeze(0)

    return (
        resulting_embeddings.to(original_device),
        indices.to(original_device),
        distances.to(original_device),
    )


def enable_protate_navigation_patches() -> None:
    """Install the temporary pRotatE navigation patches before model creation."""

    # Patch methods before KGEModel instances are created. KGEModel.__init__ stores
    # bound methods in difference_func/flexible_func, so install these first.
    KGEModel.difference_phase = difference_phase_navigation
    KGEModel.flexible_forward_protate = flexible_forward_protate_navigation

    # The pRotatE ANN implementation previously returned two values while the
    # environment expected three and did not accept rollout-shaped tensors.
    ANN_IndexMan_pRotatE.search = protate_ann_search
