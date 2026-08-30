"""Exposure-bias experiments for supervised pRotatE quotient navigation.

This entry point leaves ``nav_supervised_training_protate_quotient.py`` as the
pure gold-state baseline and adds controlled alternatives for the state fed to
the *next* supervised reasoning step:

* ``gold``: always advance with the gold relation/NO-OP target (baseline);
* ``fixed``: independently use the policy mean with a fixed probability;
* ``scheduled``: linearly increase that probability over training;
* ``predicted``: always advance with the policy mean.

The supervised action target itself always remains the annotated relation (or
zero quotient action after a shorter question has terminated). Policy actions
used to construct the next state are detached, so these experiments change the
training-state distribution without introducing an additional through-time
loss path into earlier actions.

Experiment-only arguments are consumed here before the normal MultiHopKG parser
runs, so existing YAML files do not need new keys.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List

import torch

import nav_supervised_training as base
import nav_supervised_training_protate_quotient as quotient
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.utils.convenience import get_embeddings_from_indices
from temporary_patches.protate_quotient import relation_to_quotient_action
from temporary_patches.protate_supervision import _expand_rollouts


_STATE_MODE = "gold"
_FIXED_PREDICTED_PROB = 0.5
_SCHEDULE_START = 0.0
_SCHEDULE_END = 1.0
_SUPERVISION_CALL = 0
_TOTAL_TRAINING_CALLS = 1


def _parse_exposure_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--supervised_state_mode",
        choices=("gold", "fixed", "scheduled", "predicted"),
        default="gold",
        help="State source for the next supervised hop.",
    )
    parser.add_argument(
        "--supervised_predicted_state_probability",
        type=float,
        default=0.5,
        help="Predicted-state probability for --supervised_state_mode=fixed.",
    )
    parser.add_argument(
        "--supervised_schedule_start",
        type=float,
        default=0.0,
        help="Initial predicted-state probability for scheduled sampling.",
    )
    parser.add_argument(
        "--supervised_schedule_end",
        type=float,
        default=1.0,
        help="Final predicted-state probability for scheduled sampling.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    for name in (
        "supervised_predicted_state_probability",
        "supervised_schedule_start",
        "supervised_schedule_end",
    ):
        value = float(getattr(known, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    return known


def _predicted_state_probability() -> float:
    if _STATE_MODE == "gold":
        return 0.0
    if _STATE_MODE == "predicted":
        return 1.0
    if _STATE_MODE == "fixed":
        return _FIXED_PREDICTED_PROB
    if _STATE_MODE == "scheduled":
        if _TOTAL_TRAINING_CALLS <= 1:
            progress = 1.0
        else:
            progress = min(
                max(_SUPERVISION_CALL / float(_TOTAL_TRAINING_CALLS - 1), 0.0),
                1.0,
            )
        return _SCHEDULE_START + progress * (_SCHEDULE_END - _SCHEDULE_START)
    raise RuntimeError(f"Unknown supervised state mode: {_STATE_MODE}")


def _choose_transition_action(
    gold_target: torch.Tensor,
    predicted_mu: torch.Tensor,
    predicted_probability: float,
) -> torch.Tensor:
    """Choose gold vs detached predicted transition independently per rollout."""
    if predicted_probability <= 0.0:
        return gold_target
    predicted = predicted_mu.detach()
    if predicted_probability >= 1.0:
        return predicted

    choose_predicted = torch.rand(
        predicted.shape[:-1], device=predicted.device
    ) < predicted_probability
    return torch.where(choose_predicted.unsqueeze(-1), predicted, gold_target)


def supervise_models_protate_quotient_exposure(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    steps_in_episode: int,
    hops: List[int] = None,
    paths=None,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03,
    noise_scale: float = 0.5,
):
    """Supervise gold actions while varying the next-state source."""
    global _SUPERVISION_CALL
    del noise_scale

    assert env.knowledge_graph.model_name == "pRotatE"
    if hops is None or paths is None:
        raise ValueError("Exposure supervision requires Hops and Paths")

    reasoning_hops = int(steps_in_episode)
    device = question_embeddings.device
    question_hops = torch.tensor(hops, dtype=torch.long, device=device)
    if torch.any(question_hops < 1) or torch.any(question_hops > reasoning_hops):
        raise ValueError(
            f"Question Hops must be in [1, {reasoning_hops}] for this reasoning budget"
        )

    paths_t = quotient._pad_paths(paths, reasoning_hops, device)
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    state = obs.state
    position = obs.kge_cur_pos
    adapter_out = env.q_projected
    loss = torch.tensor(0.0, device=device)
    predicted_probability = _predicted_state_probability()

    for step in range(reasoning_hops):
        active = question_hops > step
        relation = get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding, paths_t[:, step, 1]
        )
        gold_relation_target = relation_to_quotient_action(env, relation)
        target = torch.where(
            active.unsqueeze(-1),
            gold_relation_target,
            torch.zeros_like(gold_relation_target),
        )
        target = _expand_rollouts(env, target)

        _, _, _, mu, sigma = nav_agent(state)
        all_examples = torch.ones(mu.shape[:-1], dtype=mu.dtype, device=mu.device)
        loss = loss + quotient._masked_mse(mu, target, all_examples)
        loss = loss + quotient._sigma_loss(
            sigma, sigma_scalar, expected_sigma, mask=all_examples
        )

        transition_action = _choose_transition_action(
            target, mu, predicted_probability
        )
        previous = position
        position = env.knowledge_graph.flexible_forward(position, transition_action)
        if env.add_transition_state:
            state = torch.cat(
                [env.q_projected, previous, transition_action, position], dim=-1
            )
        else:
            state = torch.cat([env.q_projected, position], dim=-1)

    _SUPERVISION_CALL += 1

    if adapter_scalar == 0.0:
        return loss
    adapter_loss = quotient._adapter_loss_mixed(
        env, adapter_out, source_ent, paths_t, question_hops
    )
    return loss + adapter_scalar * adapter_loss


def _install_training_progress_wrapper() -> None:
    """Infer total batch calls so scheduled sampling follows training progress."""
    original_train = base.train_nav_multihopkg

    def train_with_progress(*args, **kwargs):
        global _SUPERVISION_CALL, _TOTAL_TRAINING_CALLS

        if kwargs:
            epochs = int(kwargs["epochs"])
            batch_size = int(kwargs["batch_size"])
            train_data = kwargs["train_data"]
        else:
            # Positional order follows base.train_nav_multihopkg.
            batch_size = int(args[0])
            epochs = int(args[2])
            train_data = args[7]

        batches_per_epoch = max(1, math.ceil(len(train_data) / batch_size))
        _SUPERVISION_CALL = 0
        _TOTAL_TRAINING_CALLS = max(1, epochs * batches_per_epoch)

        print(
            "Exposure supervision: "
            f"mode={_STATE_MODE}, "
            f"fixed_p={_FIXED_PREDICTED_PROB:.3f}, "
            f"schedule={_SCHEDULE_START:.3f}->{_SCHEDULE_END:.3f}, "
            f"training_batches={_TOTAL_TRAINING_CALLS}"
        )
        return original_train(*args, **kwargs)

    base.train_nav_multihopkg = train_with_progress


def install_exposure_experiment(args: argparse.Namespace) -> None:
    global _STATE_MODE, _FIXED_PREDICTED_PROB, _SCHEDULE_START, _SCHEDULE_END

    _STATE_MODE = args.supervised_state_mode
    _FIXED_PREDICTED_PROB = float(args.supervised_predicted_state_probability)
    _SCHEDULE_START = float(args.supervised_schedule_start)
    _SCHEDULE_END = float(args.supervised_schedule_end)

    quotient.install_temporary_patches()
    base.supervise_models = supervise_models_protate_quotient_exposure
    _install_training_progress_wrapper()


if __name__ == "__main__":
    exposure_args = _parse_exposure_args()
    install_exposure_experiment(exposure_args)
    base.main()
