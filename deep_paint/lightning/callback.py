"""Pytorch Lightning custom callback classes."""
from typing import Dict, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
import pandas as pd
import lightning.pytorch as pl
from lightning import LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn

from deep_paint.utils.plot import plot_confusion_matrix


class PlotConfusionMatrix(pl.Callback):
    """
    Callback to plot a confusion matrix during training.

    Parameters
    ----------
    plot_every_n_epochs: int, default 5
        Plot the confusion matrix every n epochs.
    backend: str, default "matplotlib"
        The plotting backend to use. Either "matplotlib" or "plotly".
    save: bool, default False
        Whether to save the plot.
    save_path: str, optional
        The directory to save the plot.
    file_name: str, optional
        The file name of the plot to save as.
    label_col: str, optional
        The column name of the label in the metadata.
    class_col: str, optional
        The column name of the class in the metadata.
    """
    def __init__(
        self,
        plot_every_n_epochs: int = 5,
        backend: str = "matplotlib",
        save: bool = False,
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
        label_col: Optional[str] = None,
        class_col: Optional[str] = None,
    ):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.backend = backend
        self.save = save
        self.save_path = save_path
        self.file_name = file_name
        self.label_col = label_col
        self.class_col = class_col

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (pl_module.current_epoch + 1) % self.plot_every_n_epochs == 0:
            outputs = pl_module.validation_step_outputs
            metadata = trainer.datamodule.val.metadata
            logger_epoch = pl_module.current_epoch + 1
            self._plot_confusion_matrix(
                outputs=outputs,
                pl_module=pl_module,
                metadata=metadata,
                logger_prefix="val",
                logger_epoch=logger_epoch
            )
        elif trainer.state.fn == TrainerFn.VALIDATING:
            outputs = pl_module.validation_step_outputs
            metadata = trainer.datamodule.val.metadata
            self._plot_confusion_matrix(
                outputs=outputs,
                pl_module=pl_module,
                metadata=metadata,
                logger_prefix="eval"
            )
        else:
            pass

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        outputs = pl_module.test_step_outputs
        metadata = trainer.datamodule.test.metadata
        self._plot_confusion_matrix(
            outputs=outputs,
            pl_module=pl_module,
            metadata=metadata,
            logger_prefix="test"
        )

    def _plot_confusion_matrix(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        pl_module: LightningModule,
        metadata: pd.DataFrame,
        logger_prefix: Optional[str] = None,
        logger_epoch: Optional[int] = None
    ):
        """
        Wrapper function of sos_chip.utils.plot.plot_confusion_matrix.

        Refer to the documentation of sos_chip.utils.plot.plot_confusion_matrix
        for more details.
        """
        y_true = torch.cat([out['y_true'] for out in outputs]).cpu()
        y_pred = torch.cat([out['y_pred'] for out in outputs]).cpu()

        plot_confusion_matrix(
            backend=self.backend,
            y_pred=y_pred,
            y_true=y_true,
            metadata=metadata,
            num_classes=pl_module.num_classes,
            label_col=self.label_col,
            class_col=self.class_col,
            save=self.save,
            save_path=self.save_path,
            file_name=self.file_name,
            logger=pl_module.logger,
            logger_prefix=logger_prefix,
            logger_epoch=logger_epoch
        )


class ModuleFreezeUnfreeze(pl.callbacks.BaseFinetuning):
    """
    Custom callback to freeze and unfreeze modules during training.
    """
    def __init__(
        self,
        layers: Union[str, List[str]],
        freeze: bool = True,
    ):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        self.layers = layers
        self.freeze = freeze

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """
        Freeze/unfreeze layers of the model before training.
        """
        # No assumptions about the state of the layers beforehand
        if self.freeze:
            pl_module.unfreeze()
            for param_name, param in pl_module.named_parameters():
                if any(layer in param_name for layer in self.layers):
                    param.requires_grad = False
        else:
            pl_module.freeze()
            for param_name, param in pl_module.named_parameters():
                if any(layer in param_name for layer in self.layers):
                    param.requires_grad = True

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer
    ) -> None:
        pass