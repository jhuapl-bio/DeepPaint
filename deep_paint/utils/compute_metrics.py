"""Compute mean and std dev of a dataset."""
import logging

import fire
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_paint.lightning.datamodule import ForwardDataset


def compute_metrics(
    data_dir: str,
    metadata_path: str,
    image_ext: str,
    path_col: str,
    channels: bool,
    channel_map: dict,
    batch_size: int,
    num_workers: int
):
    """
    Compute dataset mean and standard deviation for normalization.
    
    Parameters
    ----------
    data_dir : str
        directory containing images.
    metadata_path : str
        path to metadata file.
    image_ext : str
        image file extension.
    path_col : str
        column in metadata file containing image paths.
    channels : bool
        whether dataset is multi-channel (not RGB)
    channel_map : dict
        mapping of channel names to wavelength notation.
    batch_size : int
        batch size for dataloader
    num_workers : int
        number of workers for dataloader
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Create dataset and dataloader
    compute_dataset = ForwardDataset(
        data_dir=data_dir,
        metadata=metadata,
        image_ext=image_ext,
        path_col=path_col,
        channels=channels,
        channel_map=channel_map,
        data_transforms=torchvision.transforms.ToTensor()
    )
    compute_dataloader = DataLoader(
        compute_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        num_workers=num_workers
    )
    # Calculating mean and std (per channel)
    mean = 0.
    meansq = 0.
    nb_samples = 0.

    logging.info("Computing mean and std of training dataset for normalization.")
    with tqdm(compute_dataloader, total=len(compute_dataloader), desc="Batch", leave=False) as batch_pbar:
        for data in batch_pbar:
            # Data is in BxCxHxW format
            batch_samples = data.size(0)
            # Aggregate H x W into one dimension
            data = data.view(batch_samples, data.size(1), -1)
            # Calculate metrics per channel (should be 1 if grayscale)
            mean += data.mean(dim=2).sum(0) # mean
            meansq += (data ** 2).mean(dim=2).sum(0) # mean square
            # Keep track of number of batch samples
            nb_samples += batch_samples

    # Normalize by number of batch samples
    mean /= nb_samples
    meansq /= nb_samples
    # std = sqrt(E[X^2] - (E[X])^2)
    std = np.sqrt(meansq - mean ** 2)

    mean = [round(m.item(), 4) for m in mean]
    std = [round(s.item(), 4) for s in std]
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean, std

if __name__=="__main__":
    fire.Fire(compute_metrics)
