import os
import shutil
import logging

from typing import List

def clean_up_checkpoints(
    save_path: str, 
    logger: logging.Logger = None
    ) -> None:
    """
    Clean up the training checkpoints directory by removing it if it exists.
    """
    
    checkpoints_dir = os.path.join(save_path, "checkpoints")
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        logger.warning(f"Checkpoints directory '{checkpoints_dir}' has been deleted.")

def clean_up_folder(
        save_path: str, 
        ignore_files_types: List[str]=['.log'],
        logger: logging.Logger = None
    ) -> None:
    """
    Clean up the folder by removing all files and subdirectories if it is empty or contains only ignored file types.
    """
    # Check if folder is empty or if it contains only ignored files, if it does, erase it
    if not os.path.exists(save_path):
        logger.info(f"Folder '{save_path}' does not exist. No action taken.")
        return
    
    if not os.listdir(save_path):
        # If folder exists but is empty, erase it
        shutil.rmtree(save_path)
        logger.warning(f"Folder '{save_path}' has been deleted because it was empty.")

    # List all files and directories in the folder, if it contains only ignored files, erase it
    files_and_dirs = os.listdir(save_path)
    if all(
        file.endswith(tuple(ignore_files_types)) for file in files_and_dirs
    ):
        shutil.rmtree(save_path)
        logger.warning(f"Folder '{save_path}' has been deleted because it contained only ignored files: {ignore_files_types}.")
