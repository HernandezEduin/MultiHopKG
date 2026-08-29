"""Tests for the temporary pRotatE continuous-navigation compatibility layer."""

import math
import unittest

import torch

from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.vector_search import ANN_IndexMan_pRotatE
from temporary_patches.protate_navigation import (
    enable_protate_navigation_patches,
    protate_navigation_distance,
)


class TestTemporaryPRotatENavigationPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Must happen before constructing KGEModel instances because KGEModel
        # caches bound transition/difference methods in __init__.
        enable_protate_navigation_patches()

    def make_model(self, hidden_dim=4):
        return KGEModel(
            model_name="pRotatE",
            nentity=4,
            nrelation=2,
            hidden_dim=hidden_dim,
            gamma=6.0,
        )

    def raw_from_radians(self, model, radians):
        return model.normalize_embedding(torch.tensor(radians, dtype=torch.float32))

    def assert_same_phase(self, model, actual, expected, atol=1e-5):
        actual_rad = model.denormalize_embedding(actual)
        expected_rad = model.denormalize_embedding(expected)
        wrapped_error = torch.atan2(
            torch.sin(actual_rad - expected_rad),
            torch.cos(actual_rad - expected_rad),
        )
        self.assertTrue(
            torch.allclose(wrapped_error, torch.zeros_like(wrapped_error), atol=atol),
            msg=f"phase error was {wrapped_error}",
        )

    def test_difference_is_within_policy_action_range(self):
        model = self.make_model(hidden_dim=4)

        head = self.raw_from_radians(
            model,
            [[-math.pi + 0.01, -2.4, 0.0, math.pi - 0.01]],
        )
        tail = self.raw_from_radians(
            model,
            [[math.pi - 0.01, 2.4, math.pi - 0.02, -math.pi + 0.01]],
        )

        action = model.difference(head, tail)

        self.assertLessEqual(float(action.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(action.min()), -1.0 - 1e-6)

    def test_oracle_difference_round_trip_recovers_tail(self):
        """The most important navigation invariant for supervised targets."""
        model = self.make_model(hidden_dim=4)

        # Include two boundary-crossing dimensions so this also catches the
        # original double-scaling/wrapping failure.
        head = self.raw_from_radians(
            model,
            [[math.pi - 0.10, -math.pi + 0.20, -1.0, 0.75]],
        )
        tail = self.raw_from_radians(
            model,
            [[-math.pi + 0.10, math.pi - 0.20, 1.3, -2.2]],
        )

        action = model.difference(head, tail)
        reconstructed_tail = model.flexible_forward(head, action)

        self.assert_same_phase(model, reconstructed_tail, tail)

    def test_known_quarter_turn_action(self):
        model = self.make_model(hidden_dim=2)
        head = self.raw_from_radians(model, [[0.0, -math.pi / 2]])

        # Policy coordinates: 0.5 means +pi/2 and -0.5 means -pi/2.
        action = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        actual = model.flexible_forward(head, action)
        expected = self.raw_from_radians(model, [[math.pi / 2, -math.pi]])

        self.assert_same_phase(model, actual, expected)

    def test_round_trip_gradients_are_finite_near_wrap_boundary(self):
        model = self.make_model(hidden_dim=2)

        head = self.raw_from_radians(
            model,
            [[math.pi - 1e-3, -math.pi + 2e-3]],
        ).detach().requires_grad_(True)
        tail = self.raw_from_radians(
            model,
            [[-math.pi + 3e-3, math.pi - 4e-3]],
        )

        action = model.difference(head, tail)
        reconstructed_tail = model.flexible_forward(head, action)
        loss = reconstructed_tail.square().mean()
        loss.backward()

        self.assertIsNotNone(head.grad)
        self.assertTrue(torch.isfinite(head.grad).all())
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(reconstructed_tail).all())

    def test_navigation_distance_matches_protate_pi_periodicity(self):
        target = torch.tensor([[0.0]])
        candidates = torch.tensor([[0.0], [math.pi], [math.pi / 2]])

        distance = protate_navigation_distance(
            target.unsqueeze(1), candidates.unsqueeze(0)
        ).squeeze(0)

        self.assertAlmostEqual(float(distance[0]), 0.0, places=6)
        self.assertAlmostEqual(float(distance[1]), 0.0, places=5)
        self.assertAlmostEqual(float(distance[2]), 1.0, places=6)

    def test_absolute_difference_matches_protate_pi_periodicity(self):
        model = self.make_model(hidden_dim=1)
        head = self.raw_from_radians(model, [[0.0]])
        pi_equivalent = self.raw_from_radians(model, [[math.pi]])
        quarter_turn = self.raw_from_radians(model, [[math.pi / 2]])

        equivalent_distance = model.absolute_difference(head, pi_equivalent)
        quarter_turn_distance = model.absolute_difference(head, quarter_turn)

        self.assertTrue(
            torch.allclose(
                equivalent_distance,
                torch.zeros_like(equivalent_distance),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                quarter_turn_distance,
                torch.ones_like(quarter_turn_distance),
                atol=1e-6,
            )
        )

    def test_ann_search_returns_three_values_and_supports_rollouts(self):
        embedding_range = 2.0
        radians = torch.tensor(
            [
                [0.0, 0.0],
                [math.pi, math.pi],
                [math.pi / 2, math.pi / 2],
            ],
            dtype=torch.float32,
        )
        raw_embeddings = radians * (embedding_range / math.pi)
        ann = ANN_IndexMan_pRotatE(
            raw_embeddings,
            embedding_range=embedding_range,
        )

        # [batch=2, rollouts=3, dim=2]
        query = raw_embeddings[0].reshape(1, 1, 2).expand(2, 3, 2).clone()
        resulting_embeddings, indices, distances = ann.search(query, topk=2)

        self.assertEqual(tuple(resulting_embeddings.shape), (2, 3, 2, 2))
        self.assertEqual(tuple(indices.shape), (2, 3, 2))
        self.assertEqual(tuple(distances.shape), (2, 3, 2))

        # 0 and pi are equivalent under abs(sin(delta)), so both should be the
        # two nearest candidates with approximately zero distance.
        self.assertTrue(torch.allclose(distances, torch.zeros_like(distances), atol=1e-5))
        self.assertTrue(
            torch.all(
                torch.sort(indices, dim=-1).values
                == torch.tensor([0, 1]).reshape(1, 1, 2)
            )
        )


if __name__ == "__main__":
    unittest.main()
