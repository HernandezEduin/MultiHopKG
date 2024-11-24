"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
from functools import cmp_to_key
import os
import pickle
import random
from typing import List, Optional, Set, Tuple, Dict
import time

import torch
import torch.nn as nn

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
from src.utils.logs import create_logger
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda
import pdb

IdLookUpDict = Dict[int, Dict[int, Set[int]]]

class KnowledgeGraph(nn.Module):
    """
    The discrete knowledge graph is stored with an adjacency list.
    """
    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.adj_list = None
        self.bandwidth = args.bandwidth
        self.args = args

        self.action_space = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects: Optional[IdLookUpDict] = None
        self.all_entities: Optional[Set[int]] = None
        self.all_entities_list: Optional[List[int]] = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        print('** Create {} knowledge graph **'.format(args.model))
        self.model = args.model
        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir, args.add_reversed_training_edges)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None

        # Create the logger for sanity checking
        self.logger = create_logger(__class__.__name__)
        
        self.define_modules()
        self.initialize_modules()

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        print('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
        print('Sanity check: {} types loaded'.format(len(self.type2id)))
        with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
        print('Sanity check: {} relations loaded'.format(len(self.relation2id)))
       
        # Load graph structures
        if self.args.model.startswith('point'): 
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
            with open(adj_list_path, 'rb') as f:
                self.adj_list = pickle.load(f)
            self.vectorize_action_space(data_dir)

        print('Loading graph data .................... ✅')

    def vectorize_action_space(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores
                    
        # Sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        print('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))
        
        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def get_unique_r_space(e1):
            if e1 in self.adj_list:
                return list(self.adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            num_facts_saved_in_action_table = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                action_space_buckets_discrete[key].append(action_space)
                num_facts_saved_in_action_table += len(action_space)
            print('Sanity check: {} facts saved in action table'.format(
                num_facts_saved_in_action_table - self.num_entities))
            for key in action_space_buckets_discrete:
                print('Vectorizing action spaces bucket {}...'.format(key))
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            print('Vectorizing action spaces...')
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)
            
            if self.args.model.startswith('rule'):
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        all_entities = set()
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    all_entities.add(e1)
                    all_entities.add(e2)
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects
        self.all_entities = all_entities
        self.all_entities_list = list(all_entities)
       
        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v
        
        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, 'dev.triples')
        test_path = os.path.join(self.args.data_dir, 'test.triples')
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                print(line)
                if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.adj_list[e1_id]:
                    self.adj_list[e1_id][r_id] = set()
                if not e2_id in self.adj_list[e1_id][r_id]:
                    self.adj_list[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        print('{} fuzzy facts added'.format(count))

        self.vectorize_action_space(self.args.data_dir)

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_all_entity_ids(self):
        return torch.arange(self.num_entities)

    def get_entity_embeddings(self, e):
        if e.dtype != torch.int64:
            pdb.set_trace()
        return self.EDropout(self.entity_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def constrain_radius_on_complex_embeddings(self):
        assert (
            self.model == "operational_rotate"
        ), "normalized_embeddings works under assumption of RotatE"
        
        with torch.no_grad():
            # Only normalize if both real and imaginary parts exist
            if isinstance(self.entity_embeddings, nn.Embedding) and isinstance(self.entity_img_embeddings, nn.Embedding):
                self.entity_embeddings.weight.data, self.entity_img_embeddings.weight.data = project_to_unit_circle(
                    self.entity_embeddings.weight.data,
                    self.entity_img_embeddings.weight.data
                )
                
            if isinstance(self.relation_embeddings, nn.Embedding) and isinstance(self.relation_img_embeddings, nn.Embedding):
                self.relation_embeddings.weight.data, self.relation_img_embeddings.weight.data = project_to_unit_circle(
                    self.relation_embeddings.weight.data,
                    self.relation_img_embeddings.weight.data
                )

    def clip_embeddings(self, max_norm: float):
        assert (
            self.model == "operational_rotate"
        ), "normalized_embeddings works under assumption of RotatE"
        "B4 you use this make sure your model is compatible."

        # Now we check for all the embeddins
        all_embeddings = {
                "entity_embeddings" : self.entity_embeddings,
                "relation_embeddings" : self.relation_embeddings,
                "entity_img_embeddings" : self.entity_img_embeddings,
                # "relation_img_embeddings" : self.relation_img_embeddings,
            }

        if torch.any(torch.isnan(self.entity_embeddings.weight)):
            pdb.set_trace()
        for k,e in all_embeddings.items():
            assert isinstance(e, nn.Embedding), f"Embedding {k} is not an Embedding"
            minv,maxv = torch.min(e.weight), torch.max(e.weight)
            self.logger.debug(f"B4 Clipping Embedding {k} min {minv} and max {maxv}")
            torch.nn.utils.clip_grad_norm_(e.weight, max_norm)
            minv,maxv = torch.min(e.weight), torch.max(e.weight)
            self.logger.debug(f"After Clippping Embedding {k} min {minv} and max {maxv}")
         
    def log_gradients(self):
        # Now we check for all the embeddins
        all_embeddings = {
                "entity_embeddings" : self.entity_embeddings,
                "relation_embeddings" : self.relation_embeddings,
                "entity_img_embeddings" : self.entity_img_embeddings,
                # "relation_img_embeddings" : self.relation_img_embeddings, # Only tryign RotatE now so not needed
            }
        for i,(k,e) in enumerate(all_embeddings.items()):
            assert isinstance(e, nn.Embedding), f"Embedding {k} is not an Embedding"
            assert e.weight.grad is not None, f"Embedding {k} has no gradient"

            # Compute the min and max of the gradients
            minv,maxv = torch.min(e.weight.grad), torch.max(e.weight.grad)
            mean =  torch.mean(e.weight.grad)
            uniformly_random_mask = torch.rand(e.weight.grad.size())
            biased_random_mask = uniformly_random_mask < 0.001
            samples = e.weight.grad[biased_random_mask].detach().flatten()
            self.logger.debug(f"{i}: Log Gradient ) Embedding {k} min {minv} and max {maxv} and mean {mean}") 
            self.logger.debug(f"{i}: Log Gradient ) We have {len(samples)} samples looking like {samples}")

        

    def negative_sampling(
        self, mini_batch: List[Tuple[int, int, int]], filter: bool = False
    ) -> List[Tuple[int, int, int]]:
        """
        Will obtain a triplet that does not exist in the KB
        Will only kee the relationship the same
        """
        if not isinstance(self.all_objects, dict) \
            or not isinstance( self.all_subjects, dict) \
            or not isinstance( self.all_entities, set) \
            or not isinstance(self.all_entities_list, list):
            raise ValueError("Please load the knowledge graph first")

        if filter:
            return self._nonvectorized_filtered_negative_sampling(
                mini_batch, self.all_objects, self.all_subjects, self.all_entities
            )

        else:
            return self._vectorized_nonfiltered_negative_sampling(
                mini_batch, self.all_objects, self.all_subjects, self.all_entities_list
            )

    def _nonvectorized_filtered_negative_sampling(
        self,
        mini_batch: List[Tuple[int, int, int]],
        all_objects_ids: IdLookUpDict,
        all_subjects_ids: IdLookUpDict,
        entity_universe_ids: Set[int],
    ):
        # Let me time this operation
        start_time = time.time()
        entity_universe = entity_universe_ids
        negative_batch = []


        for e1, e2, r in mini_batch:
            # Head or Tail Negative Sampling happens uniformly at random>
            if random.random() < 0.5:  # sample the objects/tail
                negative_sample_space = entity_universe
                if filter:
                    possible_objects = all_objects_ids[e1][r]
                    negative_sample_space = entity_universe_ids - possible_objects
                complement_e2 = random.choice(list(negative_sample_space))
                negative_batch.append((e1, complement_e2, r))
            else:
                negative_sample_space = entity_universe
                if filter:
                    possible_subjects = all_objects_ids[e2][r]
                    negative_sample_space = entity_universe - possible_subjects
                complement_e1 = random.choice(list(negative_sample_space))
                negative_batch.append((complement_e1, e2, r))
        # print(f"Negative Sampling (Filtered, Nonvectorized) takes {time.time() - start_time} seconds")

        return negative_batch

    def _vectorized_nonfiltered_negative_sampling(
        self,
        mini_batch: List[Tuple[int, int, int]],
        all_objects_ids: IdLookUpDict,
        all_subjects_ids: IdLookUpDict,
        entity_universe_list: List[int],
    ) -> List[Tuple[int, int, int]]:
        """
        Will obtain a triplet that does not exist in the KB
        Will only kee the relationship the same
        """
        time_start = time.time()
        ## Now, we will vectorize the negative sampling
        # First sample at random from the universe
        random_draws = torch.randint(
            0, len(entity_universe_list), (len(mini_batch),)
        ).tolist()
        # Then, decide which elements in batch get heads or tails
        head_or_tail = torch.randint(0, 2, (len(mini_batch),)).tolist()
        # Then, insert them into the batch
        negative_batch = [
            (0, 0, 0),
        ] * len(mini_batch)
        for i in range(len(mini_batch)):
            if head_or_tail[i] == 0:
                negative_batch[i] = (
                    mini_batch[i][0],
                    entity_universe_list[random_draws[i]],
                    mini_batch[i][2],
                )
            else:
                negative_batch[i] = (
                    entity_universe_list[random_draws[i]],
                    mini_batch[i][1],
                    mini_batch[i][2],
                )

        # print(f'Negative (Filtered and Vectorized) Sampling takes {time.time() - time_start} seconds')
        return negative_batch

    def virtual_step(self, e_set, r):
        """
        Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
        labeled r and the source entity is in e_set.
        """
        batch_size = len(e_set)
        e_set_1D = e_set.view(-1)
        r_space = self.action_space[0][0][e_set_1D]
        e_space = self.action_space[0][1][e_set_1D]
        e_space = (
            r_space.view(batch_size, -1) == r.unsqueeze(1)
        ).long() * e_space.view(batch_size, -1)
        e_set_out = []
        for i in range(len(e_space)):
            e_set_out_b = var_cuda(unique(e_space[i].data))
            e_set_out.append(e_set_out_b.unsqueeze(0))
        e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
        return e_set_out

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            if self.args.model == "complex" or self.args.model == "operational_rotate":
                self.entity_img_embeddings = nn.Embedding(
                    self.num_entities, self.entity_dim
                )
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == "complex":
            self.relation_img_embeddings = nn.Embedding(
                self.num_relations, self.relation_dim
            )
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        # TODO: Clear this mess
        
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
            # nn.init.uniform_(self.entity_embeddings.weight)
        else:
            raise RuntimeError(
                "This for LG to ensure that the relation embedding is initialized"
                "You may delete it"
            )
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        # nn.init.uniform_(self.relation_embeddings.weight)

        # DEFAULT: Not really doing anything with this atm
        self.relation_img_embeddings = None

        if self.args.model == "complex" or self.args.model == "operational_rotate":
            self.entity_img_embeddings = nn.Embedding(
                self.num_entities, self.entity_dim
            )
            nn.init.xavier_normal_(self.entity_img_embeddings.weight)
            # nn.init.uniform_(self.entity_img_embeddings.weight)

        if self.args.model == "operational_rotate":
            assert all([
                isinstance(self.entity_embeddings, nn.Embedding),
                isinstance(self.entity_img_embeddings, nn.Embedding),
                isinstance(self.relation_embeddings, nn.Embedding),
            ])
            entity_embeddings_tensor, entity_img_embeddings_tensor = project_to_unit_circle(self.entity_embeddings.weight.data, self.entity_img_embeddings.weight.data)
            self.entity_embeddings.weight.data = entity_embeddings_tensor
            self.entity_img_embeddings.weight.data = entity_img_embeddings_tensor
            # TODO: maybe lock the rotate here? to below 2pi

        ########################################
        # Lets log a snippet of the embeddings for initialization
        ########################################
        self.logger.debug("\n" + "="*50 + "\nENTITY EMBEDDINGS\n" + "="*50)
        for i in range(min(10, self.num_entities)):
            self.logger.debug(f"Entity [{i:3d}]: {self.entity_embeddings.weight[i].tolist()}")
        
        self.logger.debug("\n" + "="*50 + "\nRELATION EMBEDDINGS\n" + "="*50)
        for i in range(min(10, self.num_relations)):
            self.logger.debug(f"Relation [{i:3d}]: {self.relation_embeddings.weight[i].tolist()}")
        
        self.logger.debug("\n" + "="*50 + "\nENTITY IMAGE EMBEDDINGS\n" + "="*50)
        for i in range(min(10, self.num_entities)):
            self.logger.debug(f"Entity Image [{i:3d}]: {self.entity_img_embeddings.weight[i].tolist()}")


    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID        

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
    
def project_to_unit_circle(real_tensor: torch.Tensor, img_tensor: torch.Tensor):
    with torch.no_grad():
        # Calculate magnitude of complex numbers
        magnitude = torch.sqrt(real_tensor**2 + img_tensor**2)
        
        # Create mask for non-zero magnitudes
        non_zero_mask = magnitude > 0
        
        # Initialize output tensors
        real_normalized = torch.ones_like(real_tensor)
        img_normalized = torch.zeros_like(img_tensor)
        
        # Normalize non-zero elements
        real_normalized[non_zero_mask] = real_tensor[non_zero_mask] / magnitude[non_zero_mask]
        img_normalized[non_zero_mask] = img_tensor[non_zero_mask] / magnitude[non_zero_mask]
        
        return real_normalized, img_normalized

