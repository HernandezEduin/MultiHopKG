import re
import logging
import wandb

from typing import Dict, List

def sort_metric_key(metric_name: str) -> List[str]:
    """
    Custom sort key to handle metrics with numbers, like HITS@10.
    It splits the metric name into text and number parts for correct sorting.
    """
    parts = re.split(r'(\d+)', metric_name)
    # Convert numeric parts to integers for proper numeric sorting
    return [int(part) if part.isdigit() else part.lower() for part in parts]

def log_kge_metrics(
        mode: str, 
        step: int, 
        metrics: Dict[str, float],
        logger: logging.Logger,
        wandb: wandb = None
    ) -> None:
    '''
    Print the evaluation logs
    '''
    for metric in sorted(metrics.keys(), key=sort_metric_key):
        logger.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

    # Log to wandb as well
    if wandb is not None and wandb.run is not None:
        wandb.log({f"{mode}_{metric.replace(' ', '_')}": value for metric, value in metrics.items()}, step=step)