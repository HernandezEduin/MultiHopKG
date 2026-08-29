"""Diagnose a trained supervised pRotatE navigation checkpoint by hop.

The script reconstructs the environment from ``<checkpoint_path>/config.json``
and loads ``nav_supervised_model.pth``. It then evaluates the deterministic
policy mean in two state regimes:

* gold state: the next state is teacher-forced to the annotated tail entity,
  matching the current pRotatE supervised warm-up;
* free state: the policy mean is actually composed through pRotatE embedding
  space, matching deterministic inference dynamics.

For each hop it reports relation rank, action error, sigma, and the rank of the
annotated tail entity after the free transition. It also reports deterministic
final-answer ranks. This is intended to localize whether remaining failures are
relation prediction errors or state/composition errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import multihopkg.data_utils as data_utils
from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.utils.convenience import get_embeddings_from_indices
from multihopkg.utils.saving import load_nav_supervised_checkpoint
from multihopkg.vector_search import ANN_IndexMan_pRotatE
from temporary_patches.protate_navigation import (
    enable_protate_navigation_patches,
    protate_navigation_distance,
)
from temporary_patches.protate_policy import enable_protate_policy_patch
from temporary_patches.protate_supervision import (
    _teacher_forced_state,
    relation_to_navigation_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="test",
        help="QA split to diagnose.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Override the device stored in the checkpoint config.",
    )
    return parser.parse_args()


def _config_value(config: dict, key: str, default=None):
    value = config.get(key, default)
    return default if value is None else value


def _checkpoint_gamma(checkpoint: dict, config: dict) -> float:
    if config.get("gamma") is not None:
        return float(config["gamma"])
    if checkpoint.get("gamma") is not None:
        gamma = checkpoint["gamma"]
        return float(gamma.item() if hasattr(gamma, "item") else gamma)
    state_gamma = checkpoint.get("model_state_dict", {}).get("gamma")
    if state_gamma is None:
        raise ValueError("Could not infer pRotatE gamma from config/checkpoint")
    return float(state_gamma.item())


def build_environment(config: dict, device: torch.device):
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(config["data_dir"])

    logger = logging.getLogger("protate_policy_checkpoint")
    train_df, dev_df, test_df, _ = data_utils.load_qa_data(
        cached_metadata_path=config["cached_QAMetaData_path"],
        raw_QAData_path=config["raw_QAData_path"],
        question_tokenizer_name=config["question_tokenizer_name"],
        answer_tokenizer_name=config["answer_tokenizer_name"],
        entity2id=ent2id,
        relation2id=rel2id,
        logger=logger,
        force_recompute=False,
    )

    kge_path = config["trained_model_path"]
    entity_embeddings = np.load(os.path.join(kge_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(kge_path, "relation_embedding.npy"))
    kge_checkpoint = torch.load(
        os.path.join(kge_path, "checkpoint"), map_location="cpu"
    )
    kge_model = KGEModel.from_pretrained(
        model_name="pRotatE",
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=_checkpoint_gamma(kge_checkpoint, config),
        state_dict=kge_checkpoint["model_state_dict"],
    )

    ann_ent = ANN_IndexMan_pRotatE(
        kge_model.get_all_entity_embeddings_wo_dropout(),
        embedding_range=kge_model.embedding_range.item(),
    )
    ann_rel = ANN_IndexMan_pRotatE(
        kge_model.get_all_relations_embeddings_wo_dropout(),
        embedding_range=kge_model.embedding_range.item(),
    )

    question_model = AutoModel.from_pretrained(config["question_embedding_model"]).to(device)
    question_model.eval()

    env = ITLGraphEnvironment(
        question_embedding_module=question_model,
        question_embedding_module_trainable=_config_value(
            config, "question_embedding_module_trainable", False
        ),
        entity_dim=kge_model.get_entity_dim(),
        ff_dropout_rate=float(_config_value(config, "ff_dropout_rate", 0.0)),
        history_dim=int(_config_value(config, "history_dim", 200)),
        history_num_layers=int(_config_value(config, "history_num_layers", 1)),
        knowledge_graph=kge_model,
        relation_dim=kge_model.get_relation_dim(),
        node_data=_config_value(config, "node_data_path", ""),
        node_data_key=_config_value(config, "node_data_key", ""),
        rel_data=_config_value(config, "relationship_data_path", ""),
        rel_data_key=_config_value(config, "relationship_data_key", ""),
        id2entity=id2ent,
        entity2id=ent2id,
        id2relation=id2rel,
        relation2id=rel2id,
        ann_index_manager_ent=ann_ent,
        ann_index_manager_rel=ann_rel,
        num_rollouts=0,
        num_rollouts_test=0,
        steps_in_episode=int(config["num_rollout_steps"]),
        trained_pca=None,
        graph_pca=None,
        graph_annotation=None,
        nav_start_emb_type=config["nav_start_emb_type"],
        epsilon=float(config["nav_epsilon_error"]),
        use_ann_reward=bool(config["use_ann_reward"]),
        use_kge_question_embedding=bool(
            _config_value(config, "use_kge_question_embedding", False)
        ),
        add_transition_state=bool(_config_value(config, "add_transition_state", False)),
    ).to(device)
    env.eval()

    entity_dim = kge_model.get_entity_dim()
    relation_dim = kge_model.get_relation_dim()
    add_transition_state = bool(_config_value(config, "add_transition_state", False))
    observation_dim = (
        3 * entity_dim + 2 * relation_dim
        if add_transition_state
        else 2 * entity_dim + relation_dim
    )
    nav_agent = ContinuousPolicyGradient(
        beta=float(config["beta"]),
        gamma=float(config["rl_gamma"]),
        dim_action=relation_dim,
        dim_hidden=int(_config_value(config, "rnn_hidden", 200)),
        dim_observation=observation_dim,
    ).to(device)
    load_nav_supervised_checkpoint(nav_agent, env, config["_checkpoint_path"], logger)
    nav_agent.eval()

    return env, nav_agent, {"train": train_df, "dev": dev_df, "test": test_df}


def _rank_targets(distances: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    target_distance = distances.gather(1, target_ids.reshape(-1, 1)).squeeze(1)
    return (distances < (target_distance.unsqueeze(1) - 1e-8)).sum(dim=1) + 1


def relation_ranks(
    predicted_action: torch.Tensor,
    all_relation_actions: torch.Tensor,
    gold_relation_ids: torch.Tensor,
) -> torch.Tensor:
    distances = torch.abs(
        torch.sin(
            math.pi
            * (predicted_action.unsqueeze(1) - all_relation_actions.unsqueeze(0))
        )
    ).sum(dim=-1)
    return _rank_targets(distances, gold_relation_ids)


def entity_ranks(
    env: ITLGraphEnvironment,
    positions: torch.Tensor,
    gold_entity_ids: torch.Tensor,
) -> torch.Tensor:
    entity_raw = env.knowledge_graph.get_all_entity_embeddings_wo_dropout()
    entity_rad = env.knowledge_graph.denormalize_embedding(entity_raw)
    position_rad = env.knowledge_graph.denormalize_embedding(positions)
    distances = protate_navigation_distance(
        position_rad.unsqueeze(1), entity_rad.unsqueeze(0)
    )
    return _rank_targets(distances, gold_entity_ids)


def _append(store: Dict[str, List[torch.Tensor]], key: str, value: torch.Tensor):
    store.setdefault(key, []).append(value.detach().cpu().reshape(-1))


def _summarize_ranks(result: dict, prefix: str, ranks: torch.Tensor):
    ranks = ranks.float()
    result[f"{prefix}_count"] = int(ranks.numel())
    result[f"{prefix}_hits1"] = float((ranks <= 1).float().mean())
    result[f"{prefix}_hits3"] = float((ranks <= 3).float().mean())
    result[f"{prefix}_mean_rank"] = float(ranks.mean())
    result[f"{prefix}_mrr"] = float((1.0 / ranks).mean())


def main() -> None:
    args = parse_args()
    enable_protate_navigation_patches()
    enable_protate_policy_patch()

    config_path = os.path.join(args.checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing saved config: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    config["_checkpoint_path"] = args.checkpoint_path

    requested_device = args.device or _config_value(config, "device", "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    env, nav_agent, splits = build_environment(config, device)
    df = splits[args.split]
    if args.max_questions is not None:
        df = df.iloc[: args.max_questions]

    all_relation_raw = env.knowledge_graph.get_all_relations_embeddings_wo_dropout()
    all_relation_actions = relation_to_navigation_action(env, all_relation_raw)

    metrics: Dict[str, List[torch.Tensor]] = {}
    steps = int(config["num_rollout_steps"])

    with torch.no_grad():
        for offset in range(0, len(df), args.batch_size):
            batch = df.iloc[offset : offset + args.batch_size]
            questions = batch["Question"].tolist()
            source_ent = batch["Source-Entity"].tolist()
            answer_ids = batch["Answer-Entity"].tolist()
            paths = torch.tensor(
                np.asarray(batch["Paths"].tolist()),
                dtype=torch.long,
                device=device,
            )

            question_embeddings = env.get_llm_embeddings(questions, device)
            obs = env.reset(
                question_embeddings,
                answer_ent=answer_ids,
                source_ent=source_ent,
                warmup=True,
            )

            gold_state = obs.state
            free_state = obs.state.clone()
            gold_position = obs.kge_cur_pos
            free_position = obs.kge_cur_pos.clone()

            for step in range(steps):
                relation_ids = paths[:, step, 1]
                gold_tail_ids = paths[:, step, 2]
                relation_raw = get_embeddings_from_indices(
                    env.knowledge_graph.relation_embedding, relation_ids
                )
                target_action = relation_to_navigation_action(env, relation_raw)
                gold_tail = get_embeddings_from_indices(
                    env.knowledge_graph.entity_embedding, gold_tail_ids
                )

                _, _, _, gold_mu, gold_sigma = nav_agent(gold_state)
                _, _, _, free_mu, free_sigma = nav_agent(free_state)

                gold_rel_rank = relation_ranks(
                    gold_mu, all_relation_actions, relation_ids
                )
                free_rel_rank = relation_ranks(
                    free_mu, all_relation_actions, relation_ids
                )
                _append(metrics, f"hop{step+1}_gold_relation_rank", gold_rel_rank)
                _append(metrics, f"hop{step+1}_free_relation_rank", free_rel_rank)

                gold_delta = gold_mu - target_action
                free_delta = free_mu - target_action
                _append(
                    metrics,
                    f"hop{step+1}_gold_periodic_action_error",
                    torch.abs(torch.sin(math.pi * gold_delta)).mean(dim=-1),
                )
                _append(
                    metrics,
                    f"hop{step+1}_free_periodic_action_error",
                    torch.abs(torch.sin(math.pi * free_delta)).mean(dim=-1),
                )
                _append(
                    metrics,
                    f"hop{step+1}_gold_linear_action_mae",
                    gold_delta.abs().mean(dim=-1),
                )
                _append(
                    metrics,
                    f"hop{step+1}_free_linear_action_mae",
                    free_delta.abs().mean(dim=-1),
                )
                _append(
                    metrics,
                    f"hop{step+1}_gold_sigma_mean",
                    gold_sigma.mean(dim=-1),
                )
                _append(
                    metrics,
                    f"hop{step+1}_free_sigma_mean",
                    free_sigma.mean(dim=-1),
                )

                next_free_position = env.knowledge_graph.flexible_forward(
                    free_position, free_mu
                )
                free_tail_rank = entity_ranks(
                    env, next_free_position, gold_tail_ids
                )
                _append(metrics, f"hop{step+1}_free_gold_tail_rank", free_tail_rank)

                gold_state = _teacher_forced_state(
                    env,
                    prev_position=gold_position,
                    target_action=target_action,
                    gold_tail=gold_tail,
                )
                gold_position = gold_tail

                if env.add_transition_state:
                    free_state = torch.cat(
                        [
                            env.q_projected,
                            free_position,
                            free_mu,
                            next_free_position,
                        ],
                        dim=-1,
                    )
                else:
                    free_state = torch.cat(
                        [env.q_projected, next_free_position], dim=-1
                    )
                free_position = next_free_position

            final_answer_ids = torch.tensor(answer_ids, dtype=torch.long, device=device)
            final_ranks = entity_ranks(env, free_position, final_answer_ids)
            _append(metrics, "deterministic_final_answer_rank", final_ranks)

    result = {
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "questions": int(len(df)),
        "steps": steps,
        "supervised_adapter_scalar": float(
            _config_value(config, "supervised_adapter_scalar", 0.0)
        ),
        "epochs_configured": int(config["epochs"]),
    }

    for step in range(1, steps + 1):
        for state_kind in ("gold", "free"):
            key = f"hop{step}_{state_kind}_relation_rank"
            ranks = torch.cat(metrics[key])
            _summarize_ranks(result, f"hop{step}_{state_kind}_relation", ranks)

        tail_ranks = torch.cat(metrics[f"hop{step}_free_gold_tail_rank"])
        _summarize_ranks(result, f"hop{step}_free_gold_tail", tail_ranks)

        for metric_name in (
            "gold_periodic_action_error",
            "free_periodic_action_error",
            "gold_linear_action_mae",
            "free_linear_action_mae",
            "gold_sigma_mean",
            "free_sigma_mean",
        ):
            values = torch.cat(metrics[f"hop{step}_{metric_name}"]).float()
            result[f"hop{step}_{metric_name}_mean"] = float(values.mean())

    final_ranks = torch.cat(metrics["deterministic_final_answer_rank"])
    _summarize_ranks(result, "deterministic_final_answer", final_ranks)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
