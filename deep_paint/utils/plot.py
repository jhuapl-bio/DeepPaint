"""Plotting functions for confusion matrices and PCA embeddings."""
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from torchmetrics.functional.classification import (
    binary_confusion_matrix, multiclass_confusion_matrix)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_confusion_matrix(
    backend: str,
    num_classes: int,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metadata: pd.DataFrame,
    label_col: Optional[str] = None,
    class_col: Optional[str] = None,
    title: str = 'Confusion Matrix',
    save: bool = False,
    save_path: Optional[str] = None,
    file_name: str = 'confusion_matrix',
    logger: Optional[TensorBoardLogger] = None,
    logger_prefix: Optional[str] = None,
    logger_epoch: Optional[int] = None,
) -> None:
    """
    Plot a pretty confusion matrix given a matrix of tensors.

    Parameters
    ----------
    backend: str
        backend to use for plotting
    num_classes: int
        number of classes
    y_pred: torch.Tensor
        predicted labels
    y_true: torch.Tensor
        true labels
    metadata: pd.DataFrame
        image metadata DataFrame
    label_col: str, optional
        column name of labels in metadata
    class_col: str, optional
        column name of class names in metadata
    title: str, default 'Confusion Matrix'
        title of plot
    save: bool, default False
        whether to save confusion matrix plot
    save_path: str, optional
        path to save confusion matrix image to
    file_name: str, default 'confusion_matrix'
        name of file to save plot as
    logger: TensorBoardLogger, optional
        instance of TensorboardLogger
    logger_prefix: str, optional
        directory to log image to
    logger_epoch: int, optional
        epoch to log image as
    """
    # Compute confusion matrix
    if num_classes == 2:
        cm = binary_confusion_matrix(y_pred, y_true).cpu().numpy()
    elif num_classes > 2:
        cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=num_classes).cpu().numpy()
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}. Must be greater than 2.")

    # Should specify both or neither
    if label_col is not None and class_col is not None:
        # Assert columns are in metadata
        assert label_col in metadata.columns, f"Column '{label_col}' not in metadata."
        assert class_col in metadata.columns, f"Column '{class_col}' not in metadata."
        # Get class labels from metadata
        metadata.sort_values(by=label_col, inplace=True) # Need to sort by labels to get correct order
        class_labels = metadata[class_col].unique()
    else:
        class_labels = np.arange(num_classes).astype('str') # str needed for correct figure labeling

    # Create plotly figure
    if backend == 'matplotlib':
        fig, ax = plt.subplots(1)
        # Create dataframe for easy sns plotting
        df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        sns_cm = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', ax=ax)
        sns_cm.set_title(title)
        sns_cm.set_xlabel('Predicted Label')
        sns_cm.set_ylabel('True Label')
    elif backend == 'plotly':
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues',
            text_auto=True,
            title=title
        )
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'matplotlib' or 'plotly'.")
    # Log images to logger according to:
    # https://lightning.ai/docs/pytorch/stable/extensions/logging.html#manual-logging-non-scalar-artifacts
    if logger:
        if isinstance(logger, TensorBoardLogger):
            assert backend == 'matplotlib', "TensorBoardLogger only supports matplotlib backend."
            if logger_epoch:
                step = logger_epoch
            else:
                step = 0
            logger.experiment.add_figure(f"confusion_matrix/{logger_prefix}", fig, global_step=step)
        else:
            raise ValueError(f"Invalid logger: {type(logger)}. Must be TensorBoardLogger.")
    if save:
        if save_path is None:
            logging.warning("No save path specified. Will save to home directory.")
            save_path = os.path.expanduser("~")
        if backend == 'matplotlib':
            plt.savefig(f"{save_path}/{file_name}.png", bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            fig.save_image(f"{save_path}/{file_name}.png", scale=2)
        else:
            raise ValueError(f"Invalid backend: {backend}. Must be 'matplotlib' or 'plotly'.")

def plot_embeddings(
    backend: str,
    embeddings: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    transform: bool = True,
    color_by: str = 'label',
    color_type: str = 'discrete',
    hover_data: Optional[str] = None,
    title: str = 'PCA Image Embeddings',
    save: bool = False,
    save_path: Optional[str] = None,
    file_name: Optional[str] = None,
    logger: Optional[TensorBoardLogger] = None,
    logger_prefix: Optional[str] = None,
    logger_epoch: Optional[int] = None,
) -> None:
    """
    Plot the embeddings of a trained network in a 2D space.

    Parameters
    ----------
    backend: str
        backend to use for plotting
    embeddings: pd.DataFrame
        embeddings DataFrame
    predictions: pd.DataFrame
        predictions DataFrame
    metadata: pd.DataFrame
        metadata DataFrame
    transform: bool, default True
        whether to transform embeddings before zscoring
    color_by: str, default 'label'
        column name to color the embeddings plot by
    color_type: str, default 'discrete'
        'discrete' or 'continuous' variable
    hover_data: str, optional
        column to include in hover data
    title: str, default 'PCA Image Embeddings'
        title of plot
    save: bool, default False
        whether to save the pca plot
    save_path: str, optional
        path to save the plot to
    file_name: str, optional
        name of file to save pca plot as
    logger: TensorBoardLogger, optional
        instance of TensorboardLogger
    logger_prefix: str, optional
        directory to log image to
    logger_epoch: int, optional
        epoch to log image as
    """
    # If transform, apply a log transform
    if transform:
        embeddings = np.log(embeddings + np.abs(np.min(embeddings))+ 1)

    # Scale to unit variance
    x = StandardScaler().fit_transform(embeddings)
    x = pd.DataFrame(x, columns=embeddings.columns)

    # Principal component analysis
    pca_model = PCA(n_components=2)
    pca = pca_model.fit_transform(x)

    # Concatenate predictions and principal components
    pca_df = pd.DataFrame(pca, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, predictions], axis=1)

    # Merge predictions/principal components with metadata
    pca_df = pca_df.merge(metadata, how='outer', on='index')

    # Create plot
    assert color_by in pca_df.columns, f'Column \'{color_by}\' not in metadata.'
    if color_type == 'discrete':
        pca_df[color_by] = pca_df[color_by].astype('category')
    if backend == 'matplotlib':
        fig, ax = plt.subplots(1)
        sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue=color_by,
            ax=ax,
            alpha=0.5
        )
        ax.set_title(title)
        ax.set_xlabel(f"PC 1 ({str(repr(round(pca_model.explained_variance_ratio_[0] * 100, 1)))}%)")
        ax.set_ylabel(f"PC 2 ({str(repr(round(pca_model.explained_variance_ratio_[1] * 100, 1)))}%)")
    elif backend == 'plotly':
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=color_by,
            hover_data=hover_data,
            labels={
                'PC1': f"PC 1 ({str(repr(round(pca_model.explained_variance_ratio_[0] * 100, 1)))}%)",
                'PC2': f"PC 2 ({str(repr(round(pca_model.explained_variance_ratio_[1] * 100, 1)))}%)"
            },
            title=title,
            opacity=0.5
        )
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'matplotlib' or 'plotly'.")
    # Log images to logger according to:
    # https://lightning.ai/docs/pytorch/stable/extensions/logging.html#manual-logging-non-scalar-artifacts
    if logger:
        if isinstance(logger, TensorBoardLogger):
            assert backend == 'matplotlib', "TensorBoardLogger only supports matplotlib backend."
            if logger_epoch:
                step = logger_epoch
            else:
                step = 0
            logger.experiment.add_figure(f"pca_embeddings/{logger_prefix}", fig, global_step=step)
        else:
            raise ValueError(f"Invalid logger: {type(logger)}. Must be TensorBoardLogger.")
    if save:
        if save_path is None:
            logging.warning("No save path specified. Will save to home directory.")
            save_path = os.path.expanduser("~")
        if backend == 'matplotlib':
            plt.savefig(f"{save_path}/{file_name}.png", bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            fig.save_image(f"{save_path}/{file_name}.png", scale=2)
        else:
            raise ValueError(f"Invalid backend: {backend}. Must be 'matplotlib' or 'plotly'.")
