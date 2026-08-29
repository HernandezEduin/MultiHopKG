"""Real-data diagnostics for temporary pRotatE continuous-navigation patches.

This script loads a trained pRotatE checkpoint plus a QA CSV containing path
triples and answers questions about the geometry of the learned embedding space:

1. Oracle reachability: if we compute the exact head->tail action with the
   patched ``difference`` operator, does applying that action recover the gold
   tail and rank it highly among entity embeddings?
2. Relation reachability: if we use the pretrained pRotatE relation embedding
   itself as a phase rotation, where does the resulting state rank the gold
   tail?
3. Policy squash reachability: if the supervised policy mean exactly matched a
   target, what happens after the policy's second tanh is applied at rollout?
4. Pi aliases: how many entity embeddings are effectively indistinguishable
   from each other under pRotatE's abs(sin(delta)) geometry?

The script does not train or modify model weights.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import multihopkg.data_utils as data_utils
from multihopkg.exogenous.sun_models import KGEModel
from temporary_patches.protate_navigation import (
    enable_protate_navigation_patches,
    protate_navigation_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--trained_model_path", required=True)
    parser.add_argument("--qa_path", required=True)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--alias_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> KGEModel:
    entity_embeddings = np.load(
        os.path.join(args.trained_model_path, "entity_embedding.npy")
    )
    relation_embeddings = np.load(
        os.path.join(args.trained_model_path, "relation_embedding.npy")
    )
    checkpoint = torch.load(
        os.path.join(args.trained_model_path, "checkpoint"),
        map_location="cpu",
    )

    gamma = args.gamma
    if gamma is None:
        if "gamma" in checkpoint:
            gamma_value = checkpoint["gamma"]
            gamma = float(gamma_value.item() if hasattr(gamma_value, "item") else gamma_value)
        else:
            model_state = checkpoint.get("model_state_dict", {})
            state_gamma = model_state.get("gamma")
            if state_gamma is None:
                raise ValueError(
                    "Could not infer gamma from checkpoint. Pass --gamma explicitly."
                )
            gamma = float(state_gamma.item())

    return KGEModel.from_pretrained(
        model_name="pRotatE",
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=gamma,
        state_dict=checkpoint["model_state_dict"],
    ).eval()


def _resolve_id(value, mapping, component_name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)

    value_str = str(value)
    if value_str.lstrip("-").isdigit():
        return int(value_str)
    if value_str not in mapping:
        raise KeyError(f"Unknown {component_name} in path triple: {value_str!r}")
    return int(mapping[value_str])


def parse_paths(value, ent2id, rel2id) -> List[Tuple[int, int, int]]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Unsupported Paths value: {type(value)}")

    triples = []
    for triple in value:
        if isinstance(triple, np.ndarray):
            triple = triple.tolist()
        if len(triple) != 3:
            raise ValueError(f"Expected path triple [h, r, t], got: {triple}")
        head, relation, tail = triple
        triples.append(
            (
                _resolve_id(head, ent2id, "head entity"),
                _resolve_id(relation, rel2id, "relation"),
                _resolve_id(tail, ent2id, "tail entity"),
            )
        )
    return triples


def relation_to_navigation_action(model: KGEModel, relation_raw: torch.Tensor) -> torch.Tensor:
    """Convert a raw learned pRotatE relation embedding to policy coordinates."""
    relation_rad = model.denormalize_embedding(relation_raw)
    relation_rad = torch.atan2(torch.sin(relation_rad), torch.cos(relation_rad))
    return relation_rad / math.pi


def entity_ranks(
    model: KGEModel,
    query_raw: torch.Tensor,
    gold_tail_ids: torch.Tensor,
) -> torch.Tensor:
    """Rank gold entities under pRotatE-compatible navigation geometry."""
    entity_raw = model.get_all_entity_embeddings_wo_dropout().detach().cpu().float()
    entity_rad = model.denormalize_embedding(entity_raw)
    query_rad = model.denormalize_embedding(query_raw.detach().cpu().float())

    distances = protate_navigation_distance(
        query_rad.unsqueeze(1), entity_rad.unsqueeze(0)
    )
    gold_distances = distances.gather(1, gold_tail_ids.reshape(-1, 1)).squeeze(1)
    ranks = (distances < (gold_distances.unsqueeze(1) - 1e-8)).sum(dim=1) + 1
    return ranks


def summarize_ranks(name: str, ranks: Sequence[int]) -> dict:
    ranks_t = torch.tensor(list(ranks), dtype=torch.float32)
    return {
        f"{name}_count": int(ranks_t.numel()),
        f"{name}_hits1": float((ranks_t <= 1).float().mean()),
        f"{name}_hits3": float((ranks_t <= 3).float().mean()),
        f"{name}_hits10": float((ranks_t <= 10).float().mean()),
        f"{name}_mean_rank": float(ranks_t.mean()),
        f"{name}_mrr": float((1.0 / ranks_t).mean()),
    }


def summarize_action_magnitudes(name: str, actions: Sequence[torch.Tensor]) -> dict:
    values = torch.cat([action.reshape(-1) for action in actions]).abs().float()
    deterministic_limit = math.tanh(1.0)
    return {
        f"{name}_mean_abs_component": float(values.mean()),
        f"{name}_max_abs_component": float(values.max()),
        f"{name}_fraction_abs_gt_0_5": float((values > 0.5).float().mean()),
        f"{name}_fraction_abs_gt_tanh1": float(
            (values > deterministic_limit).float().mean()
        ),
        "policy_double_tanh_max_deterministic_abs_action": deterministic_limit,
    }


def compute_alias_stats(model: KGEModel, tolerance: float) -> dict:
    entity_raw = model.get_all_entity_embeddings_wo_dropout().detach().cpu().float()
    entity_rad = model.denormalize_embedding(entity_raw)
    distances = protate_navigation_distance(
        entity_rad.unsqueeze(1), entity_rad.unsqueeze(0)
    )

    alias_counts = (distances <= tolerance).sum(dim=1)
    return {
        "alias_tolerance": tolerance,
        "entities": int(entity_raw.shape[0]),
        "entities_with_alias": int((alias_counts > 1).sum()),
        "fraction_entities_with_alias": float((alias_counts > 1).float().mean()),
        "mean_equivalent_entities": float(alias_counts.float().mean()),
        "max_equivalent_entities": int(alias_counts.max()),
        "alias_count_histogram": {
            str(k): int(v) for k, v in sorted(Counter(alias_counts.tolist()).items())
        },
    }


def main() -> None:
    args = parse_args()
    enable_protate_navigation_patches()
    model = load_model(args)

    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(args.data_dir)
    del id2ent, id2rel

    qa_df = pd.read_csv(args.qa_path)
    if args.max_questions is not None:
        qa_df = qa_df.iloc[: args.max_questions]

    oracle_ranks = []
    oracle_after_policy_squash_ranks = []
    relation_ranks = []
    relation_after_policy_squash_ranks = []
    oracle_actions = []
    relation_actions = []
    relation_vs_oracle_phase_error = []
    oracle_roundtrip_error = []
    hops_seen = 0

    entity_embeddings = model.get_all_entity_embeddings_wo_dropout()
    relation_embeddings = model.get_all_relations_embeddings_wo_dropout()

    for _, row in qa_df.iterrows():
        paths = parse_paths(row["Paths"], ent2id, rel2id)
        for head_id, relation_id, tail_id in paths:
            head = entity_embeddings[head_id].unsqueeze(0)
            tail = entity_embeddings[tail_id].unsqueeze(0)
            relation = relation_embeddings[relation_id].unsqueeze(0)
            gold_tail_id = torch.tensor([tail_id], dtype=torch.long)

            oracle_action = model.difference(head, tail)
            oracle_actions.append(oracle_action.detach().cpu())
            oracle_tail = model.flexible_forward(head, oracle_action)
            oracle_rank = entity_ranks(model, oracle_tail, gold_tail_id).item()
            oracle_ranks.append(int(oracle_rank))

            # Current policy behavior with near-zero sigma: supervised loss makes
            # mu approach target_action, then rollout applies actions=tanh(z),
            # z approximately mu. Thus even perfect supervision executes
            # tanh(target_action), not target_action.
            oracle_after_policy_squash = torch.tanh(oracle_action)
            squashed_oracle_tail = model.flexible_forward(
                head, oracle_after_policy_squash
            )
            oracle_after_policy_squash_ranks.append(
                int(entity_ranks(model, squashed_oracle_tail, gold_tail_id).item())
            )

            oracle_tail_rad = model.denormalize_embedding(oracle_tail)
            tail_rad = model.denormalize_embedding(tail)
            roundtrip_residual = torch.abs(
                torch.atan2(
                    torch.sin(oracle_tail_rad - tail_rad),
                    torch.cos(oracle_tail_rad - tail_rad),
                )
            )
            oracle_roundtrip_error.append(float(roundtrip_residual.max()))

            relation_action = relation_to_navigation_action(model, relation)
            relation_actions.append(relation_action.detach().cpu())
            relation_tail = model.flexible_forward(head, relation_action)
            relation_rank = entity_ranks(model, relation_tail, gold_tail_id).item()
            relation_ranks.append(int(relation_rank))

            relation_after_policy_squash = torch.tanh(relation_action)
            squashed_relation_tail = model.flexible_forward(
                head, relation_after_policy_squash
            )
            relation_after_policy_squash_ranks.append(
                int(entity_ranks(model, squashed_relation_tail, gold_tail_id).item())
            )

            relation_rad = relation_action * math.pi
            oracle_rad = oracle_action * math.pi
            relation_error = torch.abs(torch.sin(relation_rad - oracle_rad)).mean()
            relation_vs_oracle_phase_error.append(float(relation_error))
            hops_seen += 1

    result = {
        "questions": int(len(qa_df)),
        "hops_evaluated": hops_seen,
        "embedding_range": float(model.embedding_range.item()),
        **summarize_ranks("oracle", oracle_ranks),
        **summarize_ranks(
            "oracle_after_policy_double_tanh", oracle_after_policy_squash_ranks
        ),
        **summarize_action_magnitudes("oracle_action", oracle_actions),
        **summarize_ranks("relation", relation_ranks),
        **summarize_ranks(
            "relation_after_policy_double_tanh", relation_after_policy_squash_ranks
        ),
        **summarize_action_magnitudes("relation_action", relation_actions),
        "oracle_roundtrip_max_phase_error": float(max(oracle_roundtrip_error)),
        "oracle_roundtrip_mean_max_phase_error": float(
            np.mean(oracle_roundtrip_error)
        ),
        "relation_vs_oracle_mean_abs_sin_phase_error": float(
            np.mean(relation_vs_oracle_phase_error)
        ),
        **compute_alias_stats(model, args.alias_tolerance),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
