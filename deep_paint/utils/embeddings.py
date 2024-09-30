"""
Get embeddings from a trained model on a user-specified dataset.
"""
import logging
from pathlib import Path
from typing import Any
from importlib import import_module

import yaml
import fire
import torch
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_module(module_path: str) -> Any:
    """Parse the module from the YAML configuration."""
    module_path, class_name = module_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def parse_transforms(transforms_dict: dict) -> Any:
    """
    Parse data transforms from the YAML configuration.
    
    Assumes that `transforms.Compose` is wrapped around the transforms.

    Example:
    --------
    data_transforms:
        class_path: torchvision.transforms.Compose
        init_args:
            transforms:
            - class_path: torchvision.transforms.ToTensor
            - class_path: torchvision.transforms.Normalize
                init_args:
                    mean: [0.5, 0.5, 0.5]
                    std: [0.5, 0.5, 0.5]
    """
    transforms = []
    for transform in transforms_dict["init_args"]['transforms']:
        transform_class = parse_module(transform['class_path'])
        init_args = transform.get('init_args', {})
        if init_args:
            transforms.append(transform_class(**init_args))
        else:
            transforms.append(transform_class())
    return parse_module(transforms_dict['class_path'])(transforms)

def get_embeddings(config: str):
    """
    Get embeddings from a single trained model on a user-specified dataset.

    Parameters
    ----------
    cfg: str
        Path to yaml configuration file that contains args
    """
    # Load in yaml file
    with open(config, "r", encoding="utf-8") as yaml_file:
        try:
            args = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Error loading config: {exc}") from exc

    # Seed
    pl.seed_everything(args["data"]["init_args"]["seed"], workers=True)

    # Set torch precision depending on device
    device_name = torch.cuda.get_device_name()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "A100" or "H100" in device_name:
        torch.set_float32_matmul_precision('high')

    if "ckpt_path" not in args:
        raise ValueError("No checkpoint path provided.")

    # Create save path for embeddings
    path_parts = list(Path(args["ckpt_path"]).with_suffix("").parts)
    path_parts[-2] = "embeddings"
    save_path = Path(*path_parts) / args["save_file"]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint file (should be absolute path)
    LightningModel = parse_module(args['model']['class_path'])
    lightning_model = LightningModel.load_from_checkpoint(args['ckpt_path'])

    # Create torch model from lightning model
    layers = list(lightning_model.model.children())[:-1]
    model = torch.nn.Sequential(*layers)

    # Custom logic for DenseNet
    if not isinstance(layers[-1], torch.nn.AdaptiveAvgPool2d):
        model.add_module("relu", torch.nn.ReLU(inplace=True))
        model.add_module("avgpool", torch.nn.AdaptiveAvgPool2d((1, 1)))

    # Set to eval
    model = model.to(device)
    model.eval()

    # Init datamodule
    LightningDataModule = parse_module(args['data']['class_path'])

    # Parse transforms
    data_init_args = args["data"]["init_args"]
    if "train_transforms" in data_init_args and data_init_args["train_transforms"] is not None:
        data_init_args["train_transforms"] = parse_transforms(data_init_args["train_transforms"])
    if "test_transforms" in data_init_args and data_init_args["test_transforms"] is not None:
        data_init_args["test_transforms"] = parse_transforms(data_init_args["test_transforms"])

    # Setup datamodule
    lightning_datamodule = LightningDataModule(**data_init_args)
    lightning_datamodule.setup("predict")

    # Get embeddings
    all_embeddings = []
    all_indices = []
    for x, y in tqdm(lightning_datamodule.predict_dataloader(), desc="Batch"):
        with torch.no_grad():
            embeddings = torch.flatten(model(x.to(device)), 1)
        all_embeddings.append(embeddings.cpu().detach().numpy())
        all_indices.append(y[1].cpu().detach().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    df_embeddings = pd.DataFrame(all_embeddings)
    df_embeddings["index"] = all_indices

    # Merge embeddings with metadata
    combined_embeddings = df_embeddings.merge(
        lightning_datamodule.predict_metadata, on="index", how="left").reset_index(drop=True)

    # Save model embeddings
    logging.info("Saving model embeddings.")
    combined_embeddings.to_csv(save_path, index=False)


if __name__ == '__main__':
    fire.Fire(get_embeddings)
