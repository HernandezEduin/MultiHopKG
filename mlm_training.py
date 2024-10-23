#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

import copy
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
import logging

from torch.nn.utils import clip_grad_norm_
# From transformers import general tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer, BertModel
from tqdm import tqdm
import argparse

import multihopkg.data_utils as data_utils
from multihopkg.emb.emb import EmbeddingBasedMethod
from multihopkg.emb.fact_network import (
    ComplEx,
    ConvE,
    DistMult,
    get_complex_kg_state_dict,
    get_conve_kg_state_dict,
    get_distmult_kg_state_dict,
)
from multihopkg.hyperparameter_range import hp_range
from multihopkg.knowledge_graph import KnowledgeGraph

# LG: This immediately parses things. A script basically.
from multihopkg.learn_framework import LFramework
from multihopkg.run_configs import alpha
from multihopkg.rl.graph_search.pg import ContinuousPolicyGradient, PolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment, GraphSearchPolicy
from multihopkg.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from multihopkg.utils.ops import flatten
from multihopkg.utils.convenience import create_placeholder
from multihopkg.logging import setup_logger
from multihopkg.utils.setup import set_seeds
from typing import Any, Dict, Tuple, List
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda



def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError


def construct_models(args):
    """
    Will load or construct the models for the experiment
    """
    # TODO: Get the model constructed well
    raise NotImplementedError
    models = {
        "GraphEmbedding": None,  # One of: Distmult, Complex, Conve
        "PolicyGradient": None,
        "ContinuousPolicy": None,
        "RewardShapingPolicyGradient": None,
    }

    # for model_name, model in models.items():
    #     if model_name == "GraphEmbedding":
    #         model = construct_graph_embedding_model(args)
    #     elif model_name == "PolicyGradient":
    #         model = construct_policy_gradient_model(args)
    #     elif model_name == "RewardShapingPolicyGradient":
    #         model = construct_reward_shaping_policy_gradient_model(args)
    #     else:
    #         raise NotImplementedError



# TODO: re-implement this ?
# def inference(lf):
# ... ( you can find it in ./multihopkg/experiments.py )



def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    torch.cuda.set_device(args.gpu)
    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    assert isinstance(args, argparse.Namespace)

    return args, tokenizer, logger



def losses_fn(mini_batch):
    # TODO:
    raise NotImplementedError
    # def stablize_reward(r):
    #     r_2D = r.view(-1, self.num_rollouts)
    #     if self.baseline == 'avg_reward':
    #         stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
    #     elif self.baseline == 'avg_reward_normalized':
    #         stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
    #     else:
    #         raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
    #     stabled_r = stabled_r_2D.view(-1)
    #     return stabled_r
    #
    # e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
    # output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)
    #
    # # Compute policy gradient loss
    # pred_e2 = output['pred_e2']
    # log_action_probs = output['log_action_probs']
    # action_entropy = output['action_entropy']
    #
    # # Compute discounted reward
    # final_reward = self.reward_fun(e1, r, e2, pred_e2)
    # if self.baseline != 'n/a':
    #     final_reward = stablize_reward(final_reward)
    # cum_discounted_rewards = [0] * self.num_rollout_steps
    # cum_discounted_rewards[-1] = final_reward
    # R = 0
    # for i in range(self.num_rollout_steps - 1, -1, -1):
    #     R = self.gamma * R + cum_discounted_rewards[i]
    #     cum_discounted_rewards[i] = R
    #
    # # Compute policy gradient
    # pg_loss, pt_loss = 0, 0
    # for i in range(self.num_rollout_steps):
    #     log_action_prob = log_action_probs[i]
    #     pg_loss += -cum_discounted_rewards[i] * log_action_prob
    #     pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)
    #
    # # Entropy regularization
    # entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
    # pg_loss = (pg_loss - entropy * self.beta).mean()
    # pt_loss = (pt_loss - entropy * self.beta).mean()
    #
    # loss_dict = {}
    # loss_dict['model_loss'] = pg_loss
    # loss_dict['print_loss'] = float(pt_loss)
    # loss_dict['reward'] = final_reward
    # loss_dict['entropy'] = float(entropy.mean())
    # if self.run_analysis:
    #     fn = torch.zeros(final_reward.size())
    #     for i in range(len(final_reward)):
    #         if not final_reward[i]:
    #             if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
    #                 fn[i] = 1
    #     loss_dict['fn'] = fn
    #
    # return loss_dict

# TODO: Finish this inner training loop.
# TODO: 1: Assumes you arae happy with batch being passed (meaning you have to check its not too small)

def prep_questions(questions: List[torch.Tensor], model: BertModel):
    embedded_questions = model(questions)
    return embedded_questions
    
    
def batch_loop(
    mini_batch: List[torch.Tensor], # Perhaps change this ?
    grad_norm: float,
    kg: KnowledgeGraph,
    navigator: nn.Module,
    optimizer: torch.optim.Optimizer, # type: ignore
    pn: GraphSearchPolicy,
    num_rollout_steps: int,
) -> Dict[str,Any]:

    # TODO: Decide on the batch metrics.
    batch_metrics = {
        "loss": [],
        "entropy": [],
    }

    optimizer.zero_grad()

    # TODO: Prep the questions here
    questions_embeddings = prep_questions(torch.Tensor(mini_batch))

    # TODO: Run the simulation here
    experience = rollout(kg, num_rollout_steps, pn, navigator, questions, False)

    # TODO: Then we can call the loss in hindsight on the simulation performance.
    # loss = losses_fn(mini_batch)
    # loss['model_loss'].backward()
    # if grad_norm > 0:
    #     clip_grad_norm_(model.parameters(), grad_norm)

    optimizer.step()

    batch_metrics["loss"].append(loss['print_loss'])
    if 'entropy' in loss:
        batch_metrics["entropy"].append(loss['entropy'])


    # TODO: Need to figure out what `run_analysis` is doing and whether we want it
    # TOREM: If unecessary
    # if self.run_analysis:
    #     if rewards is None:
    #         rewards = loss['reward']
    #     else:
    #         rewards = torch.cat([rewards, loss['reward']])
    #     if fns is None:
    #         fns = loss['fn']
    #     else:
    #         fns = torch.cat([fns, loss['fn']])
    return batch_metrics


def train_multihopkg(
    batch_size: int,
    epochs: int,
    nav_agent: nn.Module,
    grad_norm: float,
    kg: KnowledgeGraph,
    learning_rate: float,
    num_rollout_steps: int,
    pn: GraphSearchPolicy,
    start_epoch: int,
    train_data: List[torch.Tensor],
):

    print("We got to multihopkg training")
    exit()

    # Print Model Parameters + Perhaps some more information
    print('Model Parameters')
    print('--------------------------')
    for name, param in nav_agent.named_parameters():
        print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))

    # Just use Adam Optimizer by defailt
    optimizer = torch.optim.Adam( # type: ignore
        filter(lambda p: p.requires_grad, nav_agent.parameters()), lr=learning_rate
    ) 

    #TODO: Metrics to track
    metrics_to_track = {'loss', 'entropy'}
    for epoch_id in range(start_epoch, epochs):
        logger.info('Epoch {}'.format(epoch_id))

        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()
    
        # TOREM: Perhapas no need for this shuffle.
        # random.shuffle(train_data)
        batch_losses = []
        entropies = []

        # TODO: Understand if this is actually necessary here
        # if self.run_analysis:
        #     rewards = None
        #     fns = None

        ##############################
        # Batch Iteration Starts Here.
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size)):
            mini_batch = train_data[sample_offset_idx:sample_offset_idx + batch_size]
            batch_metrics = batch_loop(
                mini_batch,
                grad_norm,
                kg,
                nav_agent,
                optimizer,
                pn,
                num_rollout_steps
            )

            # TODO: Do something with the mini batch

        # TODO: Check on the metrics:

        # TODO: (?) This is analysis. We might not need it.
        # Check training statistics
        # stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
        # if entropies:
        #     stdout_msg += ' entropy = {}'.format(np.mean(entropies))
        # print(stdout_msg)
        # self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
        # if self.run_analysis:
        #     print('* Analysis: # path types seen = {}'.format(self.num_path_types))
        #     num_hits = float(rewards.sum())
        #     hit_ratio = num_hits / len(rewards)
        #     print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
        #     num_fns = float(fns.sum())
        #     fn_ratio = num_fns / len(fns)
        #     print('* Analysis: false negative ratio = {}'.format(fn_ratio))
        #
        # # Check dev set performance
        # if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
        #     self.eval()
        #     self.batch_size = self.dev_batch_size
        #     dev_scores = self.forward(dev_data, verbose=False)
        #     print('Dev set performance: (correct evaluation)')
        #     _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
        #     metrics = mrr
        #     print('Dev set performance: (include test set labels)')
        #     src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
        #     # Action dropout anneaking
        #     if self.model.startswith('point'):
        #         eta = self.action_dropout_anneal_interval
        #         if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
        #             old_action_dropout_rate = self.action_dropout_rate
        #             self.action_dropout_rate *= self.action_dropout_anneal_factor 
        #             print('Decreasing action dropout rate: {} -> {}'.format(
        #                 old_action_dropout_rate, self.action_dropout_rate))
        #     # Save checkpoint
        #     if metrics > best_dev_metrics:
        #         self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
        #         best_dev_metrics = metrics
        #         with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
        #             o_f.write('{}'.format(epoch_id))
        #     else:
        #         # Early stopping
        #         if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
        #             break
        #     dev_metrics_history.append(metrics)
        #     if self.run_analysis:
        #         num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
        #         dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
        #         hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
        #         fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
        #         if epoch_id == 0:
        #             with open(num_path_types_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(self.num_path_types))
        #             with open(dev_metrics_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(metrics))
        #             with open(hit_ratio_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(hit_ratio))
        #             with open(fn_ratio_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(fn_ratio))
        #         else:
        #             with open(num_path_types_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(self.num_path_types))
        #             with open(dev_metrics_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(metrics))
        #             with open(hit_ratio_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(hit_ratio))
        #             with open(fn_ratio_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(fn_ratio))
        #
        #

def initialize_path(questions: torch.Tensor):
    # Questions must be turned into queries
    raise NotImplementedError
    

def rollout(
    # TODO: self.mdl should point to (policy network)
    kg: KnowledgeGraph,
    num_steps,
    navigator_agent: ContinuousPolicyGradient,
    graphman: ITLGraphEnvironment,
    questions: torch.Tensor, 
    visualize_action_probs=False,
):
    """
    Will execute rollouts in parallel.
    args:
        kg: Knowledge graph environment.
        num_steps: Number of rollout steps.
        navigator_agent: Policy network.
        graphman: Graph search policy network.
        questions: Questions already pre-embedded to be answered (num_rollouts, question_dim)
        visualize_action_probs: If set, save action probabilities for visualization.
    """
        
    assert (num_steps > 0)

    # Initialization
    # TOREM: Figure out how to get the dimension of the relationships and embeddings 
    entity_shape = create_placeholder(torch.Tensor, "entity_shape","mlm_training.py::rollout()")

    # These are all very reinforcement-learning things
    log_action_probs = []
    action_entropy = []

    # Dummy nodes ? TODO: Figur eout what they do.
    # TODO: Perhaps here we can enter through the centroid.
    # For now we still with these dummy
    r_s = int_fill_var_cuda(entity_shape, kg.dummy_start_r)
    # NOTE: We repeat these entities until we get the right shape:
    # TODO: make sure we keep all seen nodes up to date
    seen_nodes = int_fill_var_cuda(entity_shape, kg.dummy_e).unsqueeze(1)
    path_components = []

    # Save some history
    # path_trace = [(r_s, e_s)]
    path_trace = [(graphman.get_centroid())] # We can just change it to be different places we end up at.
    # NOTE:(LG): Must be run as `.reset()` for ensuring environment `pn` is stup

    # TODO: initialize the path. However tha tmay be
    # Something along these lines is what the initial path should look like for us.
    initial_path = [(graphman.get_centroid(), questions)]
    # pn.initialize_path(kg) # TOREM: Unecessasry to ask pn to form it for us.
    for t in range(num_steps):
        
        # I Dont think changing this is necessary
        last_r, e = path_trace[-1]

        # TODO: Make obseervations not rely on the question
        obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]

        # Our observations are composed simply of the places that we end up in. Perhaps the closest embedding that we find using something like ANN


        # TODO: (Mega): Oh yeah this is where we are getting all the shapes contorted and such.
        # Frankly, I think this is unecessary since we dont need to query available action spaces but rather just sample it
        # db_outcomes, inv_offset, policy_entropy = pn.transit(
        #     e, obs, kg
        # )

        sample_outcome = navigator_agent.sample_action(db_outcomes, inv_offset)
        action = sample_outcome['action_sample']
        # pn.update_path(action, kg) # TODO: Confirm this is actually needed
        action_prob = sample_outcome['action_prob']
        # log_action_probs.append(ops.safe_log(action_prob)) # TODO: Compute this again ( if necessary) 
        # action_entropy.append(policy_entropy) # TOREM: Comes from `transit` not sure if I shoudl remove it
        seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
        path_trace.append(action)

        if visualize_action_probs:
            top_k_action = sample_outcome['top_actions']
            top_k_action_prob = sample_outcome['top_action_probs']
            path_components.append((e, top_k_action, top_k_action_prob))

    pred_e2 = path_trace[-1][1]
    self.record_path_trace(path_trace)

    return {
        'pred_e2': pred_e2,
        'log_action_probs': log_action_probs,
        'action_entropy': action_entropy,
        'path_trace': path_trace,
        'path_components': path_components
    }

def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, tokenizer, logger = initial_setup()

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)

    ## Agent needs a Knowledge graph as well as the environment
    knowledge_graph = KnowledgeGraph(
        bandwidth = args.bandwidth,
        data_dir = args.data_dir,
        model = args.model,
        entity_dim = args.entity_dim,
        relation_dim = args.relation_dim,
        emb_dropout_rate = args.emb_dropout_rate,
        num_graph_convolution_layers = args.num_graph_convolution_layers,
        use_action_space_bucketing = args.use_action_space_bucketing,
        bucket_interval = args.bucket_interval,
        test = args.test,
        relation_only = args.relation_only,
    )
    # Setting up the models
    logger.info(":: (1/3) Loaded embedding model")
    env = ITLGraphEnvironment(
        entity_dim=args.entity_dim,
        ff_dropout_rate=args.ff_dropout_rate,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        knowledge_graph=knowledge_graph,
        relation_dim=args.relation_dim,
        relation_only=args.relation_only,
        relation_only_in_path=args.relation_only_in_path,
        xavier_initialization=args.xavier_initialization,
    )
    logger.info(":: (2/3) Loaded environment module")


    nav_agent = ContinuousPolicyGradient(
        args.use_action_space_bucketing,
        args.num_rollouts,
        args.baseline,
        args.beta,
        args.gamma,
        args.action_dropout_rate,
        args.action_dropout_anneal_factor,
        args.action_dropout_anneal_interval,
        args.beam_size,
        knowledge_graph,
        env, # What you just created above
        args.num_rollout_steps,
        args.model_dir,
        args.model,
        args.data_dir,
        args.batch_size,
        args.train_batch_size,
        args.dev_batch_size,
        args.start_epoch,
        args.num_epochs,
        args.num_wait_epochs,
        args.num_peek_epochs,
        args.learning_rate,
        args.grad_norm,
        args.adam_beta1,
        args.adam_beta2,
        args.train,
        args.run_analysis,
    )



    logger.info(":: (3/3) Loaded navigation agent")
    logger.info(":: Training the model")


    # TODO: Add checkpoint support:
    start_epoch = 0

    ######## ######## ########
    # Train:
    ######## ######## ########
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")

    text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_data, metadata = data_utils.process_qa_data(
        args.raw_QAPathData_path,
        args.cached_QAPathData_path,
        text_tokenizer,
    )
    list_train_data = list(train_data.values)
    

    # TODO: Load the validation data
    # dev_path = os.path.join(args.data_dir, "dev.triples")
    # dev_data = data_utils.load_triples(
    #     dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities
    # )


    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)
    start_epoch = 0
    dev_data = None

    train_multihopkg(
        args.batch_size,
        args.epochs,
        nav_agent,
        args.grad_norm,
        knowledge_graph,
        args.learning_rate,
        args.num_rollout_steps,
        args.start_epoch,
        start_epoch,
        list_train_data,
    )

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main(),
