"""
This script serves as a counterpart to the `mlm_training.py` file.

While `mlm_training.py` uses masked language modeling with three different embedding spaces 
(Question Textual Embedding with BERT, KG Embedding with a KGE model, and Answer Embedding with BART) 
to answer questions, this script focuses solely on using the KGE model for question answering.

Optionally, it may also incorporate Question Textual Embedding with BERT alongside the KG Embedding 
from the KGE model.
"""
# TODO: Make the question embeddings optional
# TODO: Improve dump_eval_metrics to support both `mlm_training.py` and `nav_training.py`

#-------------------------------------------------------------------------------
############################# Start of Imports #################################
#-------------------------------------------------------------------------------

# Standard Library Imports
import argparse
import os
import sys
import time
import random

# Debugging and Development Tools
import debugpy
import logging
import wandb
from rich import traceback

# Data Manipulation and Analysis
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter 
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
)

# Typing
from typing import List, Tuple, Dict, Any, DefaultDict

# Personal Package Imports (multihopkg)
# Utilities
import multihopkg.data_utils as data_utils
import multihopkg.utils_debug.distribution_tracker as dist_tracker
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.wandb import histogram_all_modules
from multihopkg.utils_debug.dump_evals import dump_evaluation_metrics

# Logging
from multihopkg.logging import setup_logger
from multihopkg.logs import torch_module_logging

# Reinforcement Learning
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment

# Knowledge Graph Embeddings
from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices

# Vector Search
from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_pRotatE

# Configuration
from multihopkg.run_configs import alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml

#-------------------------------------------------------------------------------
############################## End of Imports ##################################
#-------------------------------------------------------------------------------

traceback.install()
wandb_run = None


def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    set_seeds(args.seed)
    logger = setup_logger("__NAV_SV__")

    # Get Tokenizer
    question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)

    # TODO: Remove from Navigation code
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_tokenizer_name)

    assert isinstance(args, argparse.Namespace)

    return args, question_tokenizer, answer_tokenizer, logger

def single_hop_supervision(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    paths: List[List[int]],
    steps_in_episode: int,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03, # best gueess so far 0.01
):
    # === Get batch entities and relations ===
    device = question_embeddings.device
    paths = torch.tensor(paths, dtype=torch.int, device=device) # Shape: (batch, hops, path_length)
    head_ids = source_ent
    rel_ids = paths[:, 0, 1] # [batch,]

    head_emb = get_embeddings_from_indices(
        env.knowledge_graph.entity_embedding,
        torch.tensor(head_ids, dtype=torch.int),
    ) # Shape: (batch, embedding_dim)

    rel_emb  = get_embeddings_from_indices(
        env.knowledge_graph.relation_embedding,
        torch.tensor(rel_ids, dtype=torch.int),
    ) # Shape: (batch, embedding_dim)

    tail_emb = get_embeddings_from_indices(
        env.knowledge_graph.entity_embedding,
        torch.tensor(answer_id, dtype=torch.int),
    ) # Shape: (batch, embedding_dim)

    target_action = env.knowledge_graph.difference(head_emb, tail_emb) # Ideal translation vector
    # Note: target action is in radians if using pRotatE

    # === Reset env to update current position and projected question ===
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)

    # === Compute forward pass ===
    adapter_out = env.q_projected
    state = obs.state  # shape: (batch, state_dim)
    _, _, _, mu, sigma = nav_agent(state)

    policy_loss = nn.functional.mse_loss(mu, target_action)

    # === Losses ===
    query_emb = torch.cat(
        [head_emb, rel_emb], dim=-1
    )
    adapter_loss = nn.functional.mse_loss(adapter_out, query_emb)
    policy_loss += sigma_scalar * nn.functional.mse_loss(sigma, torch.full_like(sigma, expected_sigma))

    #Note: Sigma might be unstabilizing the training, but it is not clear yet.

    return policy_loss + adapter_scalar * adapter_loss

def multihop_supervision(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    steps_in_episode: int,
    hops: int,
    paths: List[List[int]],
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03, # best gueess so far 0.01
):

    # TODO: Improve paths in dataloader
    device = question_embeddings.device
    paths = torch.tensor(paths, dtype=torch.int, device=device) # Shape: (batch, hops, path_length)

    # === Reset env to update current position and projected question ===
    obs = env.reset(question_embeddings, answer_id, source_ent=source_ent, warmup=True)
    state = obs.state  # shape: (batch, state_dim)

    # === Compute forward pass ===
    adapter_out = env.q_projected  # shape: (batch, adapter_dim)
    
    # Initialize policy_loss on the right device
    policy_loss = torch.tensor(0.0, device=device)
    for step in range(hops):
        # Get the path embeddings
        path_ids = paths[:, step, :] # Shape: (batch, path_length)

        tail_ids = path_ids[:,2]  # (batch,) - 3rd column is tail entity
        tail_emb_step = get_embeddings_from_indices(
            env.knowledge_graph.entity_embedding,
            tail_ids,
        ) # Shape: (batch, embedding_dim)

        # Target action: vector from current to next entity
        target_action = env.knowledge_graph.difference(obs.kge_cur_pos, tail_emb_step) # Ideal translation vector

        # Forward policy
        _, _, _, mu, sigma = nav_agent(state)
        
         # Step environment with target_action
        obs, _, _ = env.step(target_action)
        state = obs.state

        # Accumulate loss
        policy_loss += nn.functional.mse_loss(mu, target_action)
        policy_loss += sigma_scalar * nn.functional.mse_loss(sigma, torch.full_like(sigma, expected_sigma))


    # === Losses: Adapter Loss ===
    # Head entity embeddings (source_ent: list of ints, batch size)
    head_emb = get_embeddings_from_indices(
        env.knowledge_graph.entity_embedding,
        torch.tensor(source_ent, dtype=torch.int),
    ) # Shape: (batch, embedding_dim)

    rel_emb = get_embeddings_from_indices(
        env.knowledge_graph.relation_embedding,
        path_ids[:,1],
    ) # Shape: (batch, embedding_dim)

    query_emb = torch.cat(
        [head_emb, rel_emb], dim=-1
    )

    adapter_loss = nn.functional.mse_loss(adapter_out, query_emb)

    return policy_loss + adapter_scalar * adapter_loss

def supervise_models(
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    question_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    steps_in_episode: int,
    hops: List[int] = None,
    paths: List[List[int]] = None,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03, # best gueess so far 0.01
):
    """
    Warmup step to train Residual Adapter and Stochastic Policy using supervised targets:
    - Adapter learns to map questions to relation embeddings.
    - Policy learns to predict head→tail vector as continuous action.
    """
    if hops is not None:
        assert all(steps_in_episode >= h for h in hops), "All Hops values must be smaller than or equal to steps_in_episode"
        assert all(h == hops[0] for h in hops), "All Hops values must be equal to the first value in hops"
        assert paths is not None, "Paths should be provided when hops are specified."

    assert env.knowledge_graph.model_name == "TransE", f"Unsupported KGE model: {env.knowledge_graph.model_name}. The currently supported model is TransE."

    if all(h == 1 for h in hops) or hops is None:
        # Single hop supervision
        return single_hop_supervision(
            nav_agent=nav_agent,
            env=env,
            question_embeddings=question_embeddings,
            source_ent=source_ent,
            answer_id=answer_id,
            paths=paths,
            steps_in_episode=steps_in_episode,
            adapter_scalar=adapter_scalar,
            sigma_scalar=sigma_scalar,
            expected_sigma=expected_sigma,
        )
    
    else:
        # multi hop supervision
        return multihop_supervision(
            nav_agent=nav_agent,
            env=env,
            question_embeddings=question_embeddings,
            source_ent=source_ent,
            answer_id=answer_id,
            steps_in_episode=steps_in_episode,
            hops=hops[0],  # Assuming hops is a list of equal values
            paths=paths,
            adapter_scalar=adapter_scalar,
            sigma_scalar=sigma_scalar,
            expected_sigma=expected_sigma,
        )

def rollout(
    # TODO: self.mdl should point to (policy network)
    steps_in_episode: int,
    nav_agent: ContinuousPolicyGradient,
    env: ITLGraphEnvironment,
    questions_embeddings: torch.Tensor,
    source_ent: List[int],
    answer_id: List[int],
    dev_mode: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Executes reinforcement learning (RL) episode rollouts in parallel for a given number of steps.
    This function is the core of the training process, used by both `batch_loop` and `batch_loop_dev`.
    
    During the rollout:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions.
    - Rewards are computed from the knowledge graph environment (KGE).
    - Evaluation metrics are optionally collected in development mode (`dev_mode`).

    Args:
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        questions_embeddings (torch.Tensor): 
            Pre-embedded representations of the questions to be answered. Shape: (batch_size, embedding_dim).
        source_entity (List[int]): 
            A list of relevant entities for each question, represented as lists of entity IDs.
        answer_id (List[int]): 
            A list of IDs corresponding to the correct answer entities.
        dev_mode (bool, optional): 
            If `True`, additional evaluation metrics are collected for debugging or analysis. Defaults to `False`.

    Returns:
        - log_action_probs (List[torch.Tensor]): 
            A list of log probabilities of the actions taken by the navigation agent at each step.
        - kg_rewards (List[torch.Tensor]): 
            A list of rewards computed by the knowledge graph environment for each step.
        - eval_metrics (Dict[str, Any]): 
            A dictionary of evaluation metrics collected during the rollout (only populated if `dev_mode=True`).
    """

    assert steps_in_episode > 0

    ########################################
    # Prepare lists to be returned
    ########################################
    log_action_probs = []
    entropies = []
    kg_rewards = []
    eval_metrics = DefaultDict(list)

    # Get initial observation. A concatenation of centroid and question atm. Passed through the path encoder
    observations = env.reset(
        questions_embeddings,
        answer_ent = answer_id,
        source_ent = source_ent
    )

    num_rollouts = env.num_rollouts

    answer_tensor = get_embeddings_from_indices(
            env.knowledge_graph.entity_embedding,
            torch.tensor(answer_id, dtype=torch.int),
    ).unsqueeze(1) # Shape: (batch, 1, embedding_dim)

    if num_rollouts > 0: answer_tensor = answer_tensor.expand(-1, num_rollouts, -1) # Shape: (batch, num_rollouts, embedding_dim)

    cur_state = observations.state
    # Should be of shape (batch_size, 1, hidden_dim)

    # pn.initialize_path(kg) # TOREM: Unecessasry to ask pn to form it for us.
    states_so_far = []
    for t in range(steps_in_episode):

        # Ask the navigator to navigate, agent is presented state, not position
        # State is meant to summrized path history.
        sampled_actions, log_probs, entropy, _, _ = nav_agent(cur_state)

        # TODO: Make sure we are gettign rewards from the environment.
        observations, kg_extrinsic_rewards, kg_dones = env.step(sampled_actions)

        cur_state = observations.state
        # VISITED EMBEDDINGS IS THE ENCODER

        ########################################
        # Calculate the Reward
        ########################################
        
        # Calculate how close we are
        kg_intrinsic_reward = env.knowledge_graph.absolute_difference(
            observations.kge_cur_pos,
            answer_tensor,
        ).norm(dim=-1, keepdim=True)

        reward = kg_dones*kg_extrinsic_rewards - torch.logical_not(kg_dones)*kg_intrinsic_reward  # Merging positive environment rewards with negative intrinsic ones

        if num_rollouts > 0: 
            kg_rewards.append(reward.mean(axis=1))  # (batch_size, num_rollouts, 1) -> (batch_size, 1)
            
            log_probs = log_probs.mean(axis=1)      # (batch_size, num_rollouts) -> (batch_size,)
            entropy = entropy.mean(axis=1)          # (batch_size, num_rollouts) -> (batch_size,)
        else: 
            kg_rewards.append(reward)               # (batch_size, 1)

        ########################################
        # Log Stuff for across batch
        ########################################
        # For now, we use states given by the path encoder and positions mostly for debugging
        states_so_far.append(cur_state)
        log_action_probs.append(log_probs)
        entropies.append(entropy)

        ########################################
        # Stuff that we will only use for evaluation
        ########################################
        if dev_mode:
            eval_metrics["sampled_actions"].append(sampled_actions.detach().cpu())
            eval_metrics["kge_cur_pos"].append(observations.kge_cur_pos.detach().cpu())
            eval_metrics["kge_prev_pos"].append(observations.kge_prev_pos.detach().cpu())
            eval_metrics["kge_action"].append(observations.kge_action.detach().cpu())

            'KGE Metrics'
            eval_metrics["kg_extrinsic_rewards"].append(kg_extrinsic_rewards.detach().cpu())
            eval_metrics["kg_intrinsic_reward"].append(kg_intrinsic_reward.detach().cpu())
            eval_metrics["kg_dones"].append(kg_dones.detach().cpu())

    if dev_mode:
        eval_metrics = {k: torch.stack(v) for k, v in eval_metrics.items()}

    # Return Rewards of Rollout as a Tensor
    return log_action_probs, entropies, kg_rewards, eval_metrics

def batch_loop_dev(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for the development set to compute additional evaluation metrics.
    This function is similar to `batch_loop` but focuses on collecting metrics for debugging
    and analysis during development.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from the knowledge graph environment (KGE).
    - Evaluation metrics are collected for analysis.

    This function is found within `evaluate_training` calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame): 
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        steps_in_episode (int): 
            The number of steps to execute in each episode.

    Returns:
        - `pg_loss` (torch.Tensor): 
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]): 
            A dictionary containing additional evaluation metrics collected during the batch loop.

    Notes:
        - This function is specifically designed for development and debugging purposes.
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = next(nav_agent.parameters()).device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    source_ent = mini_batch["Source-Entity"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    question_embeddings = env.get_llm_embeddings(questions, device)

    logger.warning(f"About to go into rollout")
    log_probs, entropies, kg_rewards, eval_extras = rollout(
        steps_in_episode,
        nav_agent,
        env,
        question_embeddings,
        source_ent = source_ent,
        answer_id = answer_id,
        dev_mode=True,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################

    log_probs_t = torch.stack(log_probs).T
    entropies_t = torch.stack(entropies).T
    num_steps = log_probs_t.shape[-1]

    assert not torch.isnan(log_probs_t).any(), "NaN detected in the log probs (batch_loop_dev). Aborting training."

    #-------------------------------------------------------------------------
    'Knowledge Graph Environment Rewards'

    kg_rewards_t = (
        torch.stack(kg_rewards)
    ).permute(1,0,2) # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2) # Shape: (batch_size, num_steps)

    assert not torch.isnan(kg_rewards_t).any(), "NaN detected in the kg rewards (batch_loop_dev). Aborting training."

    #-------------------------------------------------------------------------
    'Discount and Merging of Rewards'

    # TODO: Check if a weight is needed for combining the rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(kg_rewards_t).to(device) # Shape: (batch_size, num_steps)
    G = torch.zeros_like(kg_rewards_t[:, 0]).to(device) # Shape: (batch_size, num_steps)

    for t in reversed(range(kg_rewards_t.size(1))):
        G = kg_rewards_t[:, t] + gamma * G
        discounted_rewards[:, t] = G

    # Sample-wise normalization of the rewards for stability
    # discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)
    
    #--------------------------------------------------------------------------
    'Loss Calculation'

    pg_loss = -(discounted_rewards * log_probs_t)  - nav_agent.beta * entropies_t # Have to negate it into order to do gradient ascent

    logger.warning(f"We just left dev rollout")

    return pg_loss, eval_extras


def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
    warmup: bool = False,
    adapter_scalar: float = 0.5,
    sigma_scalar: float = 0.1,
    expected_sigma: float = 0.03, # best gueess so far 0.01
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for training the navigation agent.
    This function performs reinforcement learning (RL) rollouts for a batch of data
    and computes the policy gradient loss for training.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from the knowledge graph environment (KGE).
    - The policy gradient loss is calculated based on the rewards and log probabilities of actions.

    This function is found within `train_nav_multihopkg` and calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame): 
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        steps_in_episode (int): 
            The number of steps to execute in each episode.

    Returns:
        - `pg_loss` (torch.Tensor): 
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]): 
            A dictionary containing additional evaluation metrics collected during the batch loop.

    Notes:
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
        - This function is designed for training and does not collect as many metrics as `batch_loop_dev`.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = next(nav_agent.parameters()).device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    source_ent = mini_batch["Source-Entity"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    hops = mini_batch["Hops"].tolist() if "Hops" in mini_batch.columns else None
    paths = mini_batch["Paths"].tolist() if "Paths" in mini_batch.columns else None
    question_embeddings = env.get_llm_embeddings(questions, device)

    loss = supervise_models(
        steps_in_episode=steps_in_episode,
        nav_agent=nav_agent,
        env=env,
        question_embeddings=question_embeddings,
        source_ent=source_ent,
        answer_id=answer_id,
        hops=hops,
        paths=paths,
        adapter_scalar=adapter_scalar,
        sigma_scalar=sigma_scalar,
        expected_sigma=expected_sigma,
    )
    return loss, {}

def evaluate_training(
    env: ITLGraphEnvironment,
    dev_df: pd.DataFrame,
    nav_agent: ContinuousPolicyGradient,
    steps_in_episode: int,
    batch_size_dev: int,
    batch_count: int,
    verbose: bool,
    visualize: bool,
    writer: SummaryWriter,
    question_tokenizer: PreTrainedTokenizer,
    wandb_on: bool,
    iteration: int,
    timestamp: str,
):
    """
    Evaluates the performance of the navigation agent on the development set.
    This function computes evaluation metrics, logs results, and optionally visualizes the evaluation process.

    This function is found within `train_multihopkg` and is called periodically during training. 
    It calls upon `batch_loop_dev` and `dump_evaluation_metrics`.

    Args:
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        dev_df (pd.DataFrame): 
            The development dataset containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        batch_size_dev (int): 
            The batch size for the development set.
        batch_count (int): 
            The current batch count during training.
        verbose (bool): 
            If `True`, additional information is logged for debugging purposes.
        visualize (bool): 
            If `True`, visualizations of the evaluation process are generated.
        writer (SummaryWriter): 
            A TensorBoard writer for logging metrics and visualizations.
        question_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing questions.
        wandb_on (bool): 
            If `True`, logs metrics to Weights & Biases (wandb).
        iteration (int): 
            The current iteration number, used for logging and tracking progress.
        answer_id (List[int], optional): 
            A list of IDs corresponding to the correct answer entities. Defaults to `None`.

    Returns:
        None

    Notes:
        - This function evaluates only the last batch of the development set.
        - Metrics are logged to TensorBoard and optionally to wandb.
        - The function ensures that the environment and models are in evaluation mode during the process.
    """
    num_batches = len(dev_df) // batch_size_dev
    nav_agent.eval()

    env.eval()
    # env.question_embedding_module.eval()
    assert (
        not env.question_embedding_module.training
    ), "The question embedding module must not be in training mode"

    batch_cumulative_metrics = {
        "dev/batch_count": [batch_count],
        "dev/pg_loss": [],
    }  # For storing results from all batches

    current_evaluations = (
        {}
    )  # For storing results from last batch. Otherwise too much info

    with torch.no_grad():

        # We will only evaluate on the last batch
        batch_id = num_batches - 1

        mini_batch = dev_df[
            batch_id * batch_size_dev : (batch_id + 1) * batch_size_dev
        ]

        if not isinstance(  # TODO: Remove this assertion once it is never ever met again
            mini_batch, pd.DataFrame
        ):  # For the lsp to give me a break
            raise RuntimeError(
                f"The mini batch is not a pd.DataFrame, but a {type(mini_batch)}. Please check the data loading code."
            )
        
        current_evaluations["reference_questions"] = mini_batch["Question"]
        current_evaluations["true_answer"] = mini_batch["Answer"]
        current_evaluations["source_entity"] = mini_batch["Source-Entity"]
        current_evaluations["true_answer_id"] = mini_batch["Answer-Entity"]

        # Get the Metrics
        pg_loss, eval_extras = batch_loop_dev(
            env,
            mini_batch,
            nav_agent,
            steps_in_episode,
        )

        'Extract all the variables from eval_extras'
        for k, v in eval_extras.items():
            current_evaluations[k] = v

        # Accumlate the metrics
        current_evaluations["pg_loss"] = pg_loss.detach().cpu()
        batch_cumulative_metrics["dev/pg_loss"].append(pg_loss.mean().item())

        ########################################
        # Take `current_evaluations` as
        # a sample of batches and dump its results
        ########################################
        if verbose and logger:

            # eval_extras has variables that we need
            just_dump_it_here = f"./logs/nav_sv_{env.knowledge_graph.model_name.lower()}_{timestamp}_evaluation_dumps.log"

            answer_id = current_evaluations["true_answer_id"].tolist()

            answer_kge_tensor = get_embeddings_from_indices(
                env.knowledge_graph.entity_embedding,
                torch.tensor(answer_id, dtype=torch.int),
            ).unsqueeze(1) # Shape: (batch, 1, embedding_dim)

            logger.warning(f"About to go into dump_evaluation_metrics")
            dump_evaluation_metrics(
                path_to_log=just_dump_it_here,
                evaluation_metrics_dictionary=current_evaluations,
                vector_entity_searcher=env.ann_index_manager_ent,															 
                vector_rel_searcher=env.ann_index_manager_rel,
                question_tokenizer=question_tokenizer,
                answer_tokenizer=None, # TODO: Ensure this an optional parameter
                answer_kge_tensor=answer_kge_tensor,
                id2entity=env.id2entity,					   
                id2relations=env.id2relation,
                entity2title=env.entity2title,
                relation2title=env.relation2title,
                kg_model_name=env.knowledge_graph.model_name,
                kg_ent_distance_func=env.knowledge_graph.absolute_difference,
                kg_rel_denormalize_func=env.knowledge_graph.denormalize_relation,
                kg_rel_wrap_func=env.knowledge_graph.wrap_relation,
                iteration=iteration,
                writer=writer,						  
                wandb_on=wandb_on,
                logger=logger,
                llm_answered_enabled=False
            )
            logger.warning(f"We just left dump_evaluation_metrics")

            logger.warning(f"Cleaning up the dev dictionaries")
            
            current_evaluations.clear()
            eval_extras.clear()

            if not mini_batch._is_view: # if a copy was created, delete after usage
                del mini_batch

def test_nav_multihopkg(
    env: ITLGraphEnvironment,
    nav_agent: ContinuousPolicyGradient,
    test_data: pd.DataFrame,
    steps_in_episode: int,
    batch_size_test: int,
    verbose: bool,
    desc: str = "Testing Batches",
):
    # Set in testing mode
    nav_agent.eval()
    env.eval()

    device = next(nav_agent.parameters()).device

    hits_1 = []
    hits_3 = []
    hits_5 = []
    hits_10 = []
    hits_20 = []
    distance = []
    mr = []
    mrr = []

    # Override topk=1
    topk = 1

    with torch.no_grad():
        for sample_offset_idx in tqdm(range(0, len(test_data), batch_size_test), desc=desc, leave=False):
            mini_batch = test_data[sample_offset_idx : sample_offset_idx + batch_size_test] 
            
            # Deconstruct the batch
            questions = mini_batch["Question"].tolist()
            source_ent = mini_batch["Source-Entity"].tolist()
            answer_id = mini_batch["Answer-Entity"].tolist()
            question_embeddings = env.get_llm_embeddings(questions, device)

            answer_tensor = get_embeddings_from_indices(
                    env.knowledge_graph.entity_embedding,
                    torch.tensor(answer_id, dtype=torch.int),
            ).unsqueeze(1).expand(-1, env.num_rollouts, -1) # Shape: (batch, num_rollouts, embedding_dim)

            # Get initial observation. A concatenation of centroid and question atm. Passed through the path encoder
            observations = env.reset(
                question_embeddings,
                answer_ent = answer_id,
                source_ent = source_ent
            )

            cur_state = observations.state

            path_log_prob = torch.zeros((len(answer_id), env.num_rollouts), dtype=torch.float).to(device)
            for t in range(steps_in_episode):
                sampled_actions, log_prob, _, _, _ = nav_agent(cur_state)
                observations, _, _ = env.step(sampled_actions)
                cur_state = observations.state

                path_log_prob += log_prob

            # TODO: Evaluate at every step,
            # current evaluation is at the end of the episode

            kg_intrinsic_reward = env.knowledge_graph.absolute_difference(
                observations.kge_cur_pos,
                answer_tensor,
            ).norm(dim=-1).cpu() # Shape: (batch, num_rollouts)

            _, entity_indices, distances = env.ann_index_manager_ent.search(observations.kge_cur_pos.detach().cpu(), topk)
            entity_indices = entity_indices.reshape(answer_tensor.size(0), env.num_rollouts) # throwing away last dim (topk)
            distances = distances.reshape(answer_tensor.size(0), env.num_rollouts)

            # TODO: Might want to consider log_prob for sorting instead of closest distance to Any Entity
            # sort distances from smallest to largest
            # sorted_indices = distances.argsort(axis=-1)
            sorted_indices = (-path_log_prob).cpu().numpy().argsort(axis=-1)  # Sort by log_prob descending
            entity_indices = np.take_along_axis(entity_indices, sorted_indices, axis=-1)
            distances = np.take_along_axis(distances, sorted_indices, axis=-1)

            # Given that the answer_id_tensor is (batch_size,) find the first instance where answer_id_tensor == entity_indices for entity_indices second dimension
            answer_ids_tensors = torch.tensor(answer_id).unsqueeze(1)
            results = (entity_indices == answer_ids_tensors)

            # Check if any True values exist
            has_match = results.any(axis=-1)  # True if answer found, False otherwise
            rank = torch.where(
                has_match,
                results.float().argmax(axis=-1) + 1,        # Actual rank if found
                torch.tensor(env.num_rollouts + 1)          # Penalty rank if not found
            )            

            hits_1.append(rank <= 1)
            hits_3.append(rank <= 3)
            hits_5.append(rank <= 5)
            hits_10.append(rank <= 10)
            hits_20.append(rank <= 20)

            mr.append(rank)
            mrr.append((1.0 / rank))        

            distance.append(kg_intrinsic_reward)

            del entity_indices
            del results

    hits_1 = torch.cat(hits_1).float().mean().item()
    hits_3 = torch.cat(hits_3).float().mean().item()
    hits_5 = torch.cat(hits_5).float().mean().item()
    hits_10 = torch.cat(hits_10).float().mean().item()
    hits_20 = torch.cat(hits_20).float().mean().item()
    mr = torch.cat(mr).float().mean().item()
    mrr = torch.cat(mrr).float().mean().item()
    distance = torch.cat(distance).float().mean().item()

    if verbose:
        print(f"Test Results:")
        print(f"Test Data Size: {len(test_data)}")
        print(f"Hits@1: {hits_1:.4f}, Hits@3: {hits_3:.4f}, Hits@5: {hits_5:.4f}")
        print(f"Hits@10: {hits_10:.4f}, Hits@20: {hits_20:.4f}")
        print(f"Mean Rank: {mr:.4f}, Mean Reciprocal Rank: {mrr:.4f}")
        print(f"Mean Distance to Answer: {distance:.4f}")

    return {
        "hits_1": hits_1,
        "hits_3": hits_3,
        "hits_5": hits_5,
        "hits_10": hits_10,
        "hits_20": hits_20,
        "mean_rank": mr,
        "mean_reciprocal_rank": mrr,
        "distance": distance,
    }

def train_nav_multihopkg(
    batch_size: int,
    batch_size_dev: int,
    epochs: int,
    nav_agent: ContinuousPolicyGradient,
    learning_rate: float,
    steps_in_episode: int,
    env: ITLGraphEnvironment,
    start_epoch: int,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dev_df: pd.DataFrame,
    mbatches_b4_eval: int,
    verbose: bool,
    visualize: bool,
    question_tokenizer: PreTrainedTokenizer,
    track_gradients: bool,
    num_batches_till_eval: int,
    adapter_scalar: float,
    sigma_scalar: float,
    expected_sigma: float,
    wandb_on: bool,
    timestamp: str,
):
    """
    Trains the navigation agent using reinforcement learning (RL) on a knowledge graph environment.
    This function performs training over multiple epochs and evaluates the model periodically on a development set.

    During training:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions in `rollout`.
    - Rewards are computed from the knowledge graph environment (KGE) in `batch_loop`.
    - The policy gradient loss is calculated and used to update the model parameters.
    - Evaluation is performed periodically using the `evaluate_training` function.

    This function is found within `main` and is called to initiate the training process.
    This function calls upon `batch_loop` and `evaluate_training`.

    Args:
        batch_size (int): 
            The batch size for training.
        batch_size_dev (int): 
            The batch size for the development set.
        epochs (int): 
            The total number of epochs to train the model.
        nav_agent (ContinuousPolicyGradient): 
            The policy network responsible for deciding actions based on the current state.
        learning_rate (float): 
            The learning rate for the optimizer.
        steps_in_episode (int): 
            The number of steps to execute in each episode.
        env (ITLGraphEnvironment): 
            The knowledge graph environment that provides observations, rewards, and state transitions.
        start_epoch (int): 
            The epoch to start training from (useful for resuming training).
        train_data (pd.DataFrame): 
            The training dataset containing questions, answers, relevant entities, and relations.
        dev_df (pd.DataFrame): 
            The development dataset for periodic evaluation.
        mbatches_b4_eval (int): 
            The number of mini-batches to process before performing evaluation.
        verbose (bool): 
            If `True`, additional information is logged for debugging purposes.
        visualize (bool): 
            If `True`, visualizations of gradients and weights histograms are tracked along with navigation movements.
        question_tokenizer (PreTrainedTokenizer): 
            The tokenizer used for processing questions.
        track_gradients (bool): 
            If `True`, tracks and logs gradient information for debugging.
        num_batches_till_eval (int): 
            The number of batches to process before inspecting vanishing gradients.
        wandb_on (bool): 
            If `True`, logs metrics to Weights & Biases (wandb).

    Returns:
        None

    Notes:
        - The function uses reinforcement learning to train the navigation agent.
        - Periodic evaluation is performed using the `evaluate_training` function.
        - Metrics and visualizations are logged to TensorBoard and optionally to wandb.
        - The function ensures that the environment and models are in training mode during the process.
    """
    # TODO: Get the rollout working
    # Print Model Parameters + Perhaps some more information
    if verbose:
        print(
            "--------------------------\n" "Model Parameters\n" "--------------------------"
        )
        for name, param in nav_agent.named_parameters():
            print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

        for name, param in env.named_parameters():
            if param.requires_grad: print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    # local_time = time.localtime()
    # timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    writer = SummaryWriter(log_dir=f'runs/nav_sv/{env.knowledge_graph.model_name.lower()}/{timestamp}/')

    named_param_map = {param: name for name, param in (list(nav_agent.named_parameters()) + list(env.named_parameters()))}
    optimizer = torch.optim.Adam(  # type: ignore
        filter(
            lambda p: p.requires_grad,
            list(env.concat_projector.parameters()) + list(nav_agent.parameters())
        ),
        lr=learning_rate
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=10000,
    #     gamma=0.95,
    # )

    modules_to_log: List[nn.Module] = [nav_agent]

    # Variable to pass for logging
    batch_count = 0

    # Replacement for the hooks
    if track_gradients:
        grad_logger = torch_module_logging.ModuleSupervisor({
            "navigation_agent" : nav_agent, 
        })

    ########################################
    # Epoch Loop
    ########################################
    for epoch_id in tqdm(range(epochs), desc="Epoch"):
        is_warmup = epoch_id < start_epoch

        logger.info("Epoch {}".format(epoch_id))
        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()
        env.train()

        ##############################
        # Batch Loop
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size), desc="Training Batches", leave=False):
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]

            assert isinstance(
                mini_batch, pd.DataFrame
            )  # For the lsp to give me a break

            ########################################
            # Training
            ########################################
            'Forward pass'

            optimizer.zero_grad()

            mse_loss, _ = batch_loop(
                env, mini_batch, nav_agent, steps_in_episode, warmup=is_warmup,
                adapter_scalar=adapter_scalar,
                sigma_scalar=sigma_scalar,
                expected_sigma=expected_sigma,
            )

            if torch.isnan(mse_loss).any():
                logger.error("NaN detected in the warmup loss. Aborting training.")
                sys.exit()

            mse_mean = mse_loss.mean()

            mse_mean.backward()
            
            if torch.all(nav_agent.mu_layer.weight.grad == 0):
                logger.warning("Gradients are zero for mu_layer!")
                logger.error("Aborting training.")
                sys.exit()

            # Inspecting vanishing gradient
            if sample_offset_idx % num_batches_till_eval == 0 and verbose:
                # Retrieve named parameters from the optimizer
                named_params = [
                    (named_param_map[param], param)
                    for group in optimizer.param_groups
                    for param in group['params']
                ]


                # Iterate and calculate gradients as needed
                for name, param in named_params:
                    if param.requires_grad and ('bias' not in name) and (param.grad is not None):
                        if name == 'weight': name = 'concat_projector.weight'               
                        grads = param.grad.detach().cpu()
                        weights = param.detach().cpu()

                        dist_tracker.write_dist_parameters(grads, name, "Gradient", writer, epoch_id)
                        dist_tracker.write_dist_parameters(weights, name, "Weights", writer, epoch_id)

                        if wandb_on:
                            wandb.log({f"{name}/Gradient": wandb.Histogram(grads.numpy().flatten())})
                            wandb.log({f"{name}/Weights": wandb.Histogram(weights.numpy().flatten())})
                        elif visualize:
                            dist_tracker.write_dist_histogram(
                                grads.numpy().flatten(),
                                name, 
                                'g', 
                                "Gradient Histogram", 
                                "Grad Value", 
                                "Frequency", 
                                writer, 
                                epoch_id
                            )
                            dist_tracker.write_dist_histogram(
                                weights.numpy().flatten(), 
                                name, 
                                'b', 
                                "Weights Histogram", 
                                "Weight Value", 
                                "Frequency", 
                                writer, 
                                epoch_id
                            )

            optimizer.step()
            # scheduler.step()
            # logger.info(f"[Warmup] update at epoch/batch {epoch_id}/{batch_count}")

            batch_count += 1

        # For dump evaluation
        evaluate_training(
            env = env,
            dev_df = dev_df,
            nav_agent = nav_agent,
            steps_in_episode = steps_in_episode,
            batch_size_dev = batch_size_dev,
            batch_count = batch_count,
            verbose = verbose,
            visualize = visualize,
            writer = writer,
            question_tokenizer = question_tokenizer,
            wandb_on = wandb_on,
            iteration = epoch_id,
            timestamp = timestamp,
        )

        # Evaluate the Model Performance at the End of the Epoch
        valid_eval_metrics = test_nav_multihopkg(
            env = env,
            nav_agent = nav_agent,
            test_data = dev_df,
            steps_in_episode = steps_in_episode,
            batch_size_test = batch_size_dev,
            verbose = True,
            desc = f"Validating Batches {epoch_id}",
        )

        for key, value in valid_eval_metrics.items():
            if wandb_on:
                wandb.log({f"valid/{key}": value})
            else:
                writer.add_scalar(f"valid/{key}", value, epoch_id)

    # Evaluate the Model Performance at the End
    test_eval_metrics = test_nav_multihopkg(
        env = env,
        nav_agent = nav_agent,
        test_data = test_data,
        steps_in_episode = steps_in_episode,
        batch_size_test = batch_size_dev,
        verbose = True
    )

    for key, value in test_eval_metrics.items():
        if wandb_on:
            wandb.log({f"test/{key}": value})
        else:
            writer.add_scalar(f"test/{key}", value, epoch_id)

def main():
    """
    The main entry point for training and evaluating the MultiHopKG model.

    This function orchestrates the entire process, including:
    - Initializing configurations, tokenizers, and logging.
    - Loading and preprocessing the training and development datasets.
    - Setting up the knowledge graph environment and navigation agent.
    - Training the navigation agent and language model using reinforcement learning.
    - Optionally logging metrics and visualizations to Weights & Biases (wandb).

    Workflow:
    1. **Initial Setup**:
       - Parses command-line arguments and configuration files.
       - Initializes tokenizers and logging.
       - Optionally waits for a debugger to attach if in debug mode.

    2. **Data Loading**:
       - Loads the knowledge graph dictionaries (entities and relations).
       - Loads and preprocesses the QA datasets (training and development).

    3. **Environment and Model Setup**:
       - Loads the pretrained knowledge graph embeddings and initializes the KGE model.
       - Sets up the approximate nearest neighbor (ANN) index for entity and relation embeddings.
       - Initializes the pretrained language model (`HunchBart`) and the navigation agent (`ContinuousPolicyGradient`).
       - Configures the ITLGraphEnvironment for reinforcement learning.

    4. **Training**:
       - Calls the `train_multihopkg` function to train the navigation agent and language model.
       - Periodically evaluates the model on the development set using `evaluate_training`.

    5. **Logging and Visualization**:
       - Optionally logs metrics and visualizations to TensorBoard and wandb.
       - Supports visualization of the knowledge graph embeddings.

    Args:
        None

    Returns:
        None

    Notes:
        - The function assumes that all required configurations and paths are provided via command-line arguments or configuration files.
        - The function supports debugging, visualization, and logging for enhanced monitoring and analysis.
    """
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, question_tokenizer, answer_tokenizer, logger = initial_setup()
    global wandb_run

    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    
    if args.wandb:
        args.wr_name = f"{args.wandb_project_name}_{'supervised'}_{timestamp}"
        logger.info(
            f"🪄 Initializing Weights and Biases. Under project name {args.wandb_project_name} and run name {args.wr_name}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            name=args.wr_name,
            config=vars(args),
            notes=args.wr_notes,
        )
        for k, v in wandb.config.items():
            setattr(args, k, v)

        print(f"Run URL: {wandb_run.url}")
        print(f"Args: {args}")

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42020))
        debugpy.wait_for_client()

        # USe debugpy to listen

    ########################################
    # Get the data
    ########################################
    logger.info(":: Setting up the data")

    # Load the KGE Dictionaries
    id2ent, ent2id, id2rel, rel2id =  data_utils.load_dictionaries(args.data_dir)

    # Load the QA Dataset
    # TODO: Modify code so it doesn't require answer_tokenizer
    train_df, dev_df, test_df, train_metadata = data_utils.load_qa_data(
        cached_metadata_path=args.cached_QAMetaData_path,
        raw_QAData_path=args.raw_QAData_path,
        question_tokenizer_name=args.question_tokenizer_name,
        answer_tokenizer_name=args.answer_tokenizer_name, 
        entity2id=ent2id,
        relation2id=rel2id,
        logger=logger,
        force_recompute=args.force_data_prepro
    )
    if not isinstance(dev_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
        raise RuntimeError(
            "The data was not loaded properly. Please check the data loading code."
        )

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)

    ########################################
    # Set the KG Environment
    ########################################
    # Agent needs a Knowledge graph as well as the environment
    logger.info(":: Setting up the knowledge graph")

    entity_embeddings = np.load(os.path.join(args.trained_model_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(args.trained_model_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(args.trained_model_path , "checkpoint"))
    kge_model = KGEModel.from_pretrained(
        model_name=args.model,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=args.gamma,
        state_dict=checkpoint["model_state_dict"]
    )

    # Information computed by knowldege graph for future dependency injection
    dim_entity = kge_model.get_entity_dim()
    dim_relation = kge_model.get_relation_dim()

    # Paths for triples
    train_triplets_path = os.path.join(args.data_dir, "train.triples")
    dev_triplets_path = os.path.join(args.data_dir, "dev.triples")
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")

    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index.
    # Will be needed for obtaining observations.
    ########################################
    
    logger.info(":: Setting up the ANN Index")

    ########################################
    # Setup the Vector Searchers
    ########################################
    # TODO: Improve the ANN index manager for rotational models
    if args.model == "pRotatE": # for rotational kge models
        ann_index_manager_ent = ANN_IndexMan_pRotatE(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
        ann_index_manager_rel = ANN_IndexMan_pRotatE(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
    else: # for non-rotational kge models
        ann_index_manager_ent = ANN_IndexMan(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )
        ann_index_manager_rel = ANN_IndexMan(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model).to(args.device)

    # Setting up the models
    logger.info(":: Setting up the environment")
    env = ITLGraphEnvironment(
        question_embedding_module=question_embedding_module,
        question_embedding_module_trainable=args.question_embedding_module_trainable,
        entity_dim=dim_entity,
        ff_dropout_rate=args.ff_dropout_rate,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        knowledge_graph=kge_model,
        relation_dim=dim_relation,
        node_data=args.node_data_path,
        node_data_key=args.node_data_key,
        rel_data=args.relationship_data_path,
        rel_data_key=args.relationship_data_key,
        id2entity=id2ent,
        entity2id=ent2id,
        id2relation=id2rel,
        relation2id=rel2id,
        ann_index_manager_ent=ann_index_manager_ent,
        ann_index_manager_rel=ann_index_manager_rel,
        num_rollouts=0,
        num_rollouts_test=args.num_rollouts_test,
        steps_in_episode=args.num_rollout_steps,
        trained_pca=None,
        graph_pca=None,
        graph_annotation=None,
        nav_start_emb_type=args.nav_start_emb_type,
        epsilon = args.nav_epsilon_error,
        use_kge_question_embedding=args.use_kge_question_embedding,
        add_transition_state=args.add_transition_state
    ).to(args.device)

    # env.concat_projector.to(args.device)

    # Now we load this from the embedding models

    # TODO: Reorganizew the parameters lol
    logger.info(":: Setting up the navigation agent")
    nav_agent = ContinuousPolicyGradient(
        beta=args.beta,
        gamma=args.rl_gamma,
        dim_action=dim_relation,
        dim_hidden=args.rnn_hidden,
        # dim_observation=args.history_dim,  # observation will be into history
        dim_observation = 3*dim_entity + 2*dim_relation if args.add_transition_state else 2*dim_entity + dim_relation,
    ).to(args.device)

    # ======================================
    # Visualizaing nav_agent models using Netron
    # Save a model into .onnx format
    # torch_input = torch.randn(12, 768)
    # onnx_program = torch.onnx.dynamo_export(nav_agent, torch_input)
    # onnx_program.save("models/images/nav_agent.onnx")
    # ======================================

    # TODO: Add checkpoint support
    # See args.start_epoch

    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)

    ######## ######## ########
    # Train:
    ######## ######## ########
    start_epoch = 0
    logger.info(":: Training the model")

    if args.visualize:
        args.verbose = True

    train_nav_multihopkg(
        batch_size=args.batch_size,
        batch_size_dev=args.batch_size_dev,
        epochs=args.epochs,
        nav_agent=nav_agent,
        learning_rate=args.learning_rate,
        steps_in_episode=args.num_rollout_steps,
        env=env,
        start_epoch=args.start_epoch,
        train_data=train_df,
        test_data=test_df,
        dev_df=dev_df,
        mbatches_b4_eval=args.batches_b4_eval,
        verbose=args.verbose,
        visualize=args.visualize,
        question_tokenizer=question_tokenizer,
        track_gradients=args.track_gradients,
        num_batches_till_eval=args.num_batches_till_eval,
        adapter_scalar=args.supervised_adapter_scalar,
        sigma_scalar=args.supervised_sigma_scalar,
        expected_sigma=args.supervised_expected_sigma,
        wandb_on=args.wandb,
        timestamp=timestamp,
    )

    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)

if __name__ == "__main__":
    main()