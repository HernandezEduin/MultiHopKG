import os
import argparse
import json
import logging

import numpy as np
import torch

from multihopkg.exogenous.sun_models import KGEModel

def save_train_configs(args: argparse.Namespace, save_path: str = None) -> None:
    '''
    Save the the training configurations to a json file from argparse.
    Assumes the path to save the config is in args.save_path unless save_path is provided.
    '''
    
    # covert dict to argparse.Namespace
    if isinstance(args, dict):
        args = argparse.Namespace(**args)

    if save_path is not None:
        args.save_path = save_path

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson, indent=4)

def save_kge_model(
    model: KGEModel, 
    optimizer: torch.optim.Optimizer, 
    save_variable_list: dict, 
    save_dir: str, 
    autoencoder_flag: bool = False
):
    '''
    Save the parameters of the kge model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_dir, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_dir, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_dir, 'relation_embedding'), 
        relation_embedding
    )

    if model.model_name == 'TransH':
        norm_vector = model.norm_vector.detach().cpu().numpy()
        np.save(
            os.path.join(save_dir, 'norm_vector'),
            norm_vector
        )

    if autoencoder_flag:
        encoded_relation = model.relation_encoder(model.relation_embedding)
        np.save(
            os.path.join(save_dir, 'encoded_relation'),
            encoded_relation.detach().cpu().numpy()
        )     
        decoded_relation = model.relation_decoder(encoded_relation)
        np.save(
            os.path.join(save_dir, 'decoded_relation'),
            decoded_relation.detach().cpu().numpy()
        )

def update_best_kge_model(
        model: KGEModel,
        optimizer: torch.optim.Optimizer,
        save_variable_list: dict,
        save_dir: str,
        metric_name: str,
        metric_value: float,
        best_metric_value: float,
        best_model_path: str,
        autoencoder_flag: bool = False,
        maximize: bool = True,
        logger: logging.Logger = None
):
    """
    Overwrite previous best model in root save_dir if metric is improved.
    """
    improved = (best_metric_value is None) or ((metric_value > best_metric_value) if maximize else (metric_value < best_metric_value))
    if improved:
        old = best_metric_value
        best_metric_value = metric_value
        save_kge_model(model, optimizer, save_variable_list, save_dir, autoencoder_flag)
        if logger is not None:
            logger.info(f"Best model updated in root: {save_dir} with {metric_name}: {metric_value:.5f} (prev best: {old})")
        best_metric_value = metric_value
        best_model_path = save_dir
    else:
        if logger is not None:
            logger.info(f"Current {metric_name}: {metric_value:.5f} did not improve over best {metric_name}: {best_metric_value:.5f}. Best model remains at: {best_model_path}")
    return best_metric_value, best_model_path

def load_selective_kge_embeddings(
        kge_model: KGEModel, 
        init_checkpoint: str, 
        reload_entities: bool = False, 
        reload_relationship: bool = False,
        logger: logging.Logger = None
        ) -> None:
    """
    Loads only entity or relation embeddings from a checkpoint directory instead of the full model.
    """
    checkpoint_path = os.path.join(init_checkpoint, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    current_state = kge_model.state_dict()

    reload_keys = []
    if reload_entities:
        # All keys for entity embedding
        reload_keys.extend([k for k in state_dict.keys() if "entity_embedding" in k])
    if reload_relationship:
        reload_keys.extend([k for k in state_dict.keys() if "relation_embedding" in k])
        # If you have autoencoder weights for relations, add those here if you wish:
        # reload_keys.extend([k for k in state_dict.keys() if "relation_encoder" in k or "relation_decoder" in k])

    # Overwrite only requested keys
    for key in reload_keys:
        if key in current_state:
            current_state[key] = state_dict[key]
    kge_model.load_state_dict(current_state, strict=False)
    if logger is not None:
        logger.info(f"Reloaded embeddings: {', '.join(reload_keys)} from {checkpoint_path}")