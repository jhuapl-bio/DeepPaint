"""PyTorch Lightning Datamodules and Datasets."""
import os
import logging
from pathlib import Path
from typing import Callable, List, Optional
from PIL import Image

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from deep_paint.utils.split import split
from deep_paint.utils.config import get_value


class ForwardDataset(Dataset):
    """
    Base Dataset class.

    The folder structure takes the form:

    data_dir/
    ├── metadata/
    │   └── metadata.csv
    └── images/
        ├── Experiment1/
        │   ├── Plate1/
        │   │   ├── image1.png
        │   │   ├── image2.png
        │   │   └── ...
        │   ├── Plate2/
        │   │   ├── image1.png
        │   │   ├── image2.png
        │   │   └── ...
        │   └── ...
        ├── Experiment2/
        │   ├── Plate1/
        │   │   ├── image1.png
        │   │   ├── image2.png
        │   │   └── ...
        │   └── ...
        └── ...

    Parameters
    ----------
    data_dir: str
        path to dataset directory
    metadata: pandas.DataFrame
        pandas DataFrame of image metadata
    image_ext: str
        image extension (ex: .png, .TIF)
    path_col: str
        metadata column name that contains image relative path
    channels: bool, default false
        whether the dataset is multi-channel
    channel_map: dict, optional
        dictionary that maps name of channel to wavelength notation
    data_transforms: list, optional
        data transformations applied on each dataset sample
    """

    def __init__(
        self,
        data_dir: str,
        metadata: pd.DataFrame,
        image_ext: str,
        path_col: str,
        channels: Optional[bool] = False,
        channel_map: Optional[dict] = None,
        data_transforms: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        assert isinstance(data_dir, str), "Please provide a string path to the image directory."
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        assert isinstance(metadata, pd.DataFrame), 'Metadata must be a pandas DataFrame.'
        self.metadata = metadata
        self.image_ext = image_ext
        assert path_col in self.metadata.columns, f"Column {path_col} not found in metadata."
        self.path_col = path_col
        if channels and channel_map is None:
            logging.warning("Channel map not specified, will not aggregate channels.")
        self.channels = channels
        self.channel_map = channel_map
        self.data_transforms = data_transforms

    def _stack_channels(self, rel_path: str):
        """
        Combines n images (1 per channel) into an n-dimensional array.

        Parameters
        ----------
        rel_path: str
            Relative path to image.
        """
        images = []
        for channel in self.channel_map.keys():
            image_path = Path(
                self.image_dir, rel_path + self.channel_map[channel]
            ).with_suffix(self.image_ext)
            image = Image.open(image_path)
            image_array = np.asarray(image)
            images.append(image_array)
        return np.stack(images)

    def __getitem__(self, idx):
        rel_path = self.metadata.loc[idx, self.path_col]
        if self.channels:
            x = self._stack_channels(rel_path)  # C x H x W
        else:
            image_path = Path(self.image_dir, rel_path).with_suffix(self.image_ext)
            image = Image.open(image_path)
            x = np.asarray(image)

        # Resize and convert to tensor [0,1]
        if self.data_transforms:
            x = np.moveaxis(x, 0, -1)  # H x W x C
            x = self.data_transforms(x)

        return x

    def __len__(self):
        return len(self.metadata)


class DeepDataset(ForwardDataset):
    """
    Dataset class for a 'n-channel' biological dataset.

    Parameters
    ----------
    data_dir: str
        path to dataset directory
    metadata: pandas.DataFrame
        pandas DataFrame of image metadata
    image_ext: str, default '.png'
        image extension (ex: .png, .TIF)
    path_col: str
        metadata column name that contains image relative path
    label_col: str, default 'label'
        metadata column name that contains class labels
    channels: bool, default false
        whether the dataset is multi-channel
    channel_map: dict, optional
        dictionary that maps name of channel to wavelength notation
    data_transforms: list, optional
        data transformations applied on each dataset sample
    """

    def __init__(
        self,
        *args,
        label_col: str = 'label',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert label_col in self.metadata.columns, f"Column name: {label_col} not in metadata."
        self.label_col = label_col
        # Create explicit index column
        if "index" not in self.metadata.columns:
            self.metadata["index"] = self.metadata.index

    def __getitem__(self, idx):
        x = super().__getitem__(idx)

        label = self.metadata.loc[idx, self.label_col]
        index = self.metadata.loc[idx, "index"]
        y = (label, index)

        return x, y


class DeepLightningDataModule(pl.LightningDataModule):
    """
    LightningDataModule class for the DeepDataset Class.

    Refer to the DeepDataset class for more information.

    Parameters
    ----------
    data_dir: str
        path to dataset directory
    metadata_path: str, optional
        full path to metadata csv file or name of metadata file in `metadata` directory
    image_ext: str
        image extension (ex: .png, .TIF)
    path_col: str
        metadata column name that contains image relative path
    label_col: str
        column name that contains class labels
    stratify_col: str, optional
        column name on which to stratify split
    group_col: str, optional
        column name on which to group metadata by
    channels: bool, default false
        whether the dataset is multi-channel
    channel_map: dict, optional
        dictionary that maps name of channel to wavelength notation
    train_transforms: Callable, optional
        data transformations applied on each training sample
    test_transforms: Callable, optional
        data transformations applied on each test sample
    sizes: list, default [0.7, 0.15, 0.15]
        size of train/val/test split
    batch_size: int, default 16
        batch size for dataloader
    num_workers: int, default 8
        number of dataloader workers
    seed: int, default 42
        random seed
    sampling: str, optional
        type of sampling to implement during training
    shuffle: bool, default False
        whether to shuffle the training set (validation/test set is never shuffled)
    train_metadata: str, optional
        relative path to the training metadata file - mutually exclusive with `metadata_path`
    val_metadata: str, optional
        relative path to the validation metadata file - mutually exclusive with `metadata_path`
    test_metadata: str, optional
        relative path to the test metadata file - mutually exclusive with `metadata_path`
    predict_metadata: str, optional
        one of [train, val, test] - specifies which metadata to use for inference
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: Optional[str],
        image_ext: str,
        path_col: str,
        label_col: str,
        stratify_col: Optional[str] = None,
        group_col: Optional[str] = None,
        channels: Optional[bool] = False,
        channel_map: Optional[dict] = None,
        train_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        sizes: Optional[List[float]] = [0.7, 0.15, 0.15],
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 42,
        sampling: Optional[str] = None,
        shuffle: bool = False,
        train_metadata: Optional[str] = None,
        val_metadata: Optional[str] = None,
        test_metadata: Optional[str] = None,
        predict_metadata: Optional[str] = None
    ):
        super().__init__()
        self.prepare_data_per_node = True
        self.data_dir = self._parse_path(data_dir, "data")
        self.metadata_dir = os.path.join(self.data_dir, "metadata")
        # Stratify by label if `stratify_col` is not explicitly specified
        self.image_ext = image_ext
        self.path_col = path_col
        self.label_col = label_col
        self.stratify_col = stratify_col if stratify_col is not None else label_col
        self.group_col = group_col
        self.channels = channels
        self.channel_map = channel_map
        self.sizes = sizes
        self.seed = seed
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        if metadata_path is None:
            self.train_metadata = self._read_df(train_metadata)
            self.val_metadata = self._read_df(val_metadata)
            self.test_metadata = self._read_df(test_metadata)
            self.predict_metadata = self._assign_predict_metadata(predict_metadata)
            self.metadata = None
        else:
            self.metadata = self._read_df(metadata_path)
            if self.sizes:
                metadata_dfs = split(
                    metadata=self.metadata,
                    stratify_col=self.stratify_col,
                    group_col=self.group_col,
                    sizes=self.sizes,
                    random_state=self.seed
                )
                self.train_metadata = metadata_dfs[0]
                self.val_metadata = metadata_dfs[1]
                if len(metadata_dfs) > 2:
                    self.test_metadata = metadata_dfs[2]
                else:
                    self.test_metadata = None
                self.predict_metadata = self._assign_predict_metadata(predict_metadata)
            else:
                # Neither sizes nor train/val/test metadata provided
                logging.info(
                    "No train/val/test metadata provided. Using entire metadata for inference.")
                self.predict_metadata = self.metadata

        # Dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling = sampling
        self.shuffle = shuffle

    def prepare_data(self):
        # Data should be available locally or from shared drive
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val/test datasets for use in dataloader(s)
        if stage == "fit":
            self.train = DeepDataset(
                data_dir=self.data_dir,
                metadata=self.train_metadata,
                image_ext=self.image_ext,
                path_col=self.path_col,
                label_col=self.label_col,
                channels=self.channels,
                channel_map=self.channel_map,
                data_transforms=self.train_transforms
            )
            self.val = DeepDataset(
                data_dir=self.data_dir,
                metadata=self.val_metadata,
                image_ext=self.image_ext,
                path_col=self.path_col,
                label_col=self.label_col,
                channels=self.channels,
                channel_map=self.channel_map,
                data_transforms=self.test_transforms
            )
        # Assign val dataset for use in dataloader(s)
        if stage == "validate":
            self.val = DeepDataset(
                data_dir=self.data_dir,
                metadata=self.val_metadata,
                image_ext=self.image_ext,
                path_col=self.path_col,
                label_col=self.label_col,
                channels=self.channels,
                channel_map=self.channel_map,
                data_transforms=self.test_transforms
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = DeepDataset(
                data_dir=self.data_dir,
                metadata=self.test_metadata,
                image_ext=self.image_ext,
                path_col=self.path_col,
                label_col=self.label_col,
                channels=self.channels,
                channel_map=self.channel_map,
                data_transforms=self.test_transforms
            )
        # Assign predict dataset for use in dataloader(s)
        if stage == "predict":
            if self.predict_metadata is None:
                raise ValueError(
                    "No metadata provided for inference")
            self.predict = DeepDataset(
                data_dir=self.data_dir,
                metadata=self.predict_metadata,
                image_ext=self.image_ext,
                path_col=self.path_col,
                label_col=self.label_col,
                channels=self.channels,
                channel_map=self.channel_map,
                data_transforms=self.test_transforms
            )

    def train_dataloader(self):
        # Only implement weighted sampling for training set
        if self.sampling == "oversample":
            classes = self.train.metadata[self.label_col]
            class_sample_count = np.unique(classes, return_counts=True)[1]
            weights = 1.0 / class_sample_count
            samples_weight = weights[classes]
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = None

        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=(torch.cuda.is_available()),
            num_workers=self.num_workers,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            num_workers=self.num_workers,
        )

    def _read_df(self, path: str):
        """
        Read a DataFrame from a CSV file.

        Parameters
        ----------
        path: str
            path to a metadata file (.csv format)
        """
        if path:
            if ".csv" not in path:
                raise ValueError("Only csv files are supported metadata files.")
            if path[0] == "/":
                self.metadata_dir = ""
                df_path = path
            else:
                df_path = os.path.join(self.metadata_dir, path)
            df = pd.read_csv(Path(df_path))
            # Necessary to be compatible with Dataset classes
            if "index" not in df.columns:
                df["index"] = df.index
            return df
        return None

    def _assign_predict_metadata(self, predict_metadata):
        """
        Assign metadata for inference.

        Parameters
        ----------
        predict_metadata: str
            one of [train, val, test] - specifies which metadata to use for inference
        """
        if predict_metadata:
            if predict_metadata not in ["train", "val", "test"]:
                raise ValueError("predict_metadata must be one of [train, val, test]")
            if predict_metadata == "train":
                return self.train_metadata
            elif predict_metadata == "val":
                return self.val_metadata
            else:
                return self.test_metadata
        return None

    @staticmethod
    def _parse_path(path: str, key: str):
        if path.startswith("/"):
            return path
        return os.path.join(get_value(key=key), path)
