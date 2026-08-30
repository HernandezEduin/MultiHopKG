"""Validate pRotatE quotient-space action/state canonicalization on real data.

pRotatE scores with ``abs(sin(delta))``, so phases that differ by pi are
indistinguishable to the KGE. This diagnostic compares standard and canonical
relation actions and can evaluate the full QA CSV or an exact SplitLabel subset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import multihopkg.data_utils as data_utils
from diagnostics.protate_navigation_realdata import (
    entity_ranks,
    load_model,
    parse_paths,
    relation_to_navigation_action,
    summarize_ranks,
)
from temporary_patches.protate_navigation import enable_protate_navigation_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--trained_model_path", required=True)
    parser.add_argument("--qa_path", required=True)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("all", "train", "dev", "test"),
        default="all",
        help="Evaluate only this SplitLabel subset. Default: all rows.",
    )
    return parser.parse_args()


def canonicalize_action_pi(action: torch.Tensor) -> torch.Tensor:
    """Canonicalize normalized phase actions modulo pi into [-0.5, 0.5]."""
    phase = action * math.pi
    canonical_phase = 0.5 * torch.atan2(
        torch.sin(2.0 * phase), torch.cos(2.0 * phase)
    )
    return canonical_phase / math.pi


def canonicalize_entity_state_pi(model, raw_state: torch.Tensor) -> torch.Tensor:
    """Map raw entity coordinates to a unique pi-equivalent representative."""
    phase = model.denormalize_embedding(raw_state)
    canonical_phase = 0.5 * torch.atan2(
        torch.sin(2.0 * phase), torch.cos(2.0 * phase)
    )
    return model.normalize_embedding(canonical_phase)


def summarize_magnitudes(prefix: str, values) -> dict:
    flat = torch.cat([x.reshape(-1) for x in values]).abs().float()
    return {
        f"{prefix}_mean_abs_component": float(flat.mean()),
        f"{prefix}_max_abs_component": float(flat.max()),
        f"{prefix}_fraction_abs_gt_0_5": float((flat > 0.5).float().mean()),
    }


def main() -> None:
    args = parse_args()
    enable_protate_navigation_patches()
    model = load_model(args)

    _, ent2id, _, rel2id = data_utils.load_dictionaries(args.data_dir)
    qa_df = pd.read_csv(args.qa_path)

    if args.split != "all":
        if "SplitLabel" not in qa_df.columns:
            raise ValueError(
                f"--split={args.split} requested, but {args.qa_path} has no SplitLabel column"
            )
        qa_df = qa_df[qa_df["SplitLabel"].astype(str) == args.split].reset_index(
            drop=True
        )

    if args.max_questions is not None:
        qa_df = qa_df.iloc[: args.max_questions]

    if len(qa_df) == 0:
        raise ValueError(f"No QA rows selected for split={args.split!r}")

    entity_embeddings = model.get_all_entity_embeddings_wo_dropout()
    relation_embeddings = model.get_all_relations_embeddings_wo_dropout()

    standard_actions = []
    canonical_actions = []
    standard_hop_ranks = []
    canonical_hop_ranks = []
    standard_path_ranks = []
    canonical_path_ranks = []
    canonical_state_disagreement = []
    processed_questions = 0

    for _, row in qa_df.iterrows():
        paths = parse_paths(row["Paths"], ent2id, rel2id)
        if not paths:
            continue
        processed_questions += 1

        standard_state = entity_embeddings[paths[0][0]].unsqueeze(0)
        canonical_state = standard_state.clone()

        for head_id, relation_id, tail_id in paths:
            head = entity_embeddings[head_id].unsqueeze(0)
            relation = relation_embeddings[relation_id].unsqueeze(0)
            gold_tail_id = torch.tensor([tail_id], dtype=torch.long)

            standard_action = relation_to_navigation_action(model, relation)
            canonical_action = canonicalize_action_pi(standard_action)
            standard_actions.append(standard_action.detach().cpu())
            canonical_actions.append(canonical_action.detach().cpu())

            standard_tail = model.flexible_forward(head, standard_action)
            canonical_tail = model.flexible_forward(head, canonical_action)
            standard_hop_ranks.append(
                int(entity_ranks(model, standard_tail, gold_tail_id).item())
            )
            canonical_hop_ranks.append(
                int(entity_ranks(model, canonical_tail, gold_tail_id).item())
            )

            standard_canon = canonicalize_entity_state_pi(model, standard_tail)
            canonical_canon = canonicalize_entity_state_pi(model, canonical_tail)
            canonical_state_disagreement.append(
                float((standard_canon - canonical_canon).abs().max())
            )

            standard_state = model.flexible_forward(standard_state, standard_action)
            canonical_state = model.flexible_forward(canonical_state, canonical_action)

        final_tail_id = torch.tensor([paths[-1][2]], dtype=torch.long)
        standard_path_ranks.append(
            int(entity_ranks(model, standard_state, final_tail_id).item())
        )
        canonical_path_ranks.append(
            int(entity_ranks(model, canonical_state, final_tail_id).item())
        )

    if processed_questions == 0:
        raise ValueError("No valid paths were processed")

    result = {
        "split": args.split,
        "selected_rows": int(len(qa_df)),
        "questions": int(processed_questions),
        **summarize_magnitudes("standard_relation_action", standard_actions),
        **summarize_magnitudes("canonical_relation_action", canonical_actions),
        **summarize_ranks("standard_relation_hop", standard_hop_ranks),
        **summarize_ranks("canonical_relation_hop", canonical_hop_ranks),
        **summarize_ranks("standard_relation_path", standard_path_ranks),
        **summarize_ranks("canonical_relation_path", canonical_path_ranks),
        "canonical_state_max_abs_disagreement": float(
            max(canonical_state_disagreement)
        ),
        "canonical_state_mean_max_abs_disagreement": float(
            sum(canonical_state_disagreement) / len(canonical_state_disagreement)
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
