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