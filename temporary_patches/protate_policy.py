"""Temporary pRotatE policy compatibility patch.

The base ContinuousPolicyGradient currently applies tanh to the Gaussian mean
and then applies tanh again to a sample from that Gaussian. During supervised
training the returned mean is matched directly to the target action, so with a
small sigma the action actually executed at rollout is approximately
``tanh(target)`` rather than ``target``.

For pRotatE, where normalized actions map linearly to phase rotations in
[-pi, pi], this systematic shrinkage can be substantial. This patch uses the
standard squashed-Gaussian parameterization: an unconstrained latent Gaussian
mean and a single tanh mapping into action space. The returned ``mu`` remains in
action space so the existing supervised-training loss can remain unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient


def sample_action_single_tanh(
    self: ContinuousPolicyGradient,
    observations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a bounded action while applying tanh exactly once.

    The Normal distribution lives in an unconstrained latent space. ``mu``
    returned to callers is the deterministic action-space mean ``tanh(mu_raw)``
    so existing supervised code can continue to optimize MSE(mu, target_action).
    """

    projections = F.relu(self.hidden1(observations))
    projections = F.relu(self.hidden2(projections))

    mu_raw = self.mu_layer(projections)
    mu = torch.tanh(mu_raw)

    log_sigma_control = torch.tanh(self.sigma_layer(projections))
    log_sigma = self.log_std_min + 0.5 * (
        self.log_std_max - self.log_std_min
    ) * (log_sigma_control + 1)
    sigma = torch.exp(log_sigma)

    dist = torch.distributions.Normal(mu_raw, sigma)
    entropy = dist.entropy().sum(dim=-1)

    z = dist.rsample()
    actions = torch.tanh(z)
    log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-7)
    log_probs = log_probs.sum(dim=-1)

    return actions, log_probs, entropy, mu, sigma


def enable_protate_policy_patch() -> None:
    """Install the pRotatE-only single-tanh policy sampler."""

    ContinuousPolicyGradient._sample_action = sample_action_single_tanh
