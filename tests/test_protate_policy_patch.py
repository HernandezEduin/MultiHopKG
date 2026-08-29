"""Tests for the temporary pRotatE policy compatibility patch."""

import unittest

import torch

from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from temporary_patches.protate_policy import enable_protate_policy_patch


class TestTemporaryPRotatEPolicyPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_protate_policy_patch()

    def test_supervised_mean_matches_single_squashed_execution(self):
        torch.manual_seed(0)
        policy = ContinuousPolicyGradient(
            beta=0.0,
            gamma=1.0,
            dim_action=2,
            dim_hidden=4,
            dim_observation=3,
            log_std_min=-20,
            log_std_max=-20,
        )

        # Make the network produce a known latent mean independent of input.
        with torch.no_grad():
            policy.hidden1.weight.zero_()
            policy.hidden1.bias.fill_(1.0)
            policy.hidden2.weight.zero_()
            policy.hidden2.bias.fill_(1.0)
            policy.mu_layer.weight.zero_()
            policy.mu_layer.bias.copy_(torch.tensor([1.0, -1.0]))
            policy.sigma_layer.weight.zero_()
            policy.sigma_layer.bias.zero_()

        observations = torch.zeros(8, 3)
        actions, _, _, mu, _ = policy(observations)

        expected = torch.tanh(torch.tensor([1.0, -1.0])).expand_as(mu)
        self.assertTrue(torch.allclose(mu, expected, atol=1e-6))
        self.assertTrue(torch.allclose(actions, expected, atol=1e-4))

        # Guard against the previous behavior: tanh(tanh(raw_mean)).
        double_squashed = torch.tanh(expected)
        self.assertFalse(torch.allclose(actions, double_squashed, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
