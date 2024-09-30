"""PyTorch Lightning wrapper class for CNN models."""
import ssl
import logging
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.nn.modules.loss import _Loss
import torchvision
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


class DeepLightningModule(pl.LightningModule):
    """
    LightningModule Wrapper for DenseNet161 and other CNN Models

    Parameters
    ----------
    backbone: str
        Name of CNN backbone.
    weights: str
        Name of torchvision pre-trained weights.
    num_classes: int
        Number of classes.
    num_channels: int
        Number of input channels.
    loss: _Loss
        Loss function.
    """

    def __init__(
        self,
        backbone: Optional[str] = None,
        weights: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_channels: Optional[int] = None,
        loss: Optional[_Loss] = None,
        **kwargs
    ):
        super().__init__()
        # Initialize model
        if backbone is None:
            backbone = RXRX2_DEFAULT["backbone"]
        if weights is None:
            logging.info("Loading model without torchvision weights.")
        else:
            logging.info(f"Loading model with {weights} weights.")
        if num_classes is None:
            num_classes = RXRX2_DEFAULT["num_classes"]
        self.num_classes = num_classes
        if num_channels is None:
            num_channels = RXRX2_DEFAULT["num_channels"]
        self.num_channels = num_channels
        self.model = self._load_model(backbone, num_classes, num_channels, weights=weights)
        if loss is None:
            loss = RXRX2_DEFAULT["loss"]
        self.loss = loss
        # Required for ckpt to save hyperparameters
        self.save_hyperparameters()
        # Metrics
        if self.num_classes == 2:
            self.train_f1_score = BinaryF1Score()
            self.val_f1_score = BinaryF1Score()
            self.test_f1_score = BinaryF1Score()
        else:
            self.train_f1_score = MulticlassF1Score(num_classes=self.num_classes)
            self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes)
            self.test_f1_score = MulticlassF1Score(num_classes=self.num_classes)
        # Required as of lightning v2.0.0
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y[0]
        y_true = label.long()
        logits = self.forward(x)
        y_pred = torch.argmax(logits, dim=1)

        # Log loss
        loss = self.loss(logits, y_true)
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        # Log f1 score
        self.train_f1_score(y_pred, y_true)
        self.log("f1/train", self.train_f1_score, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "y_true": y_true.detach(), "y_pred": y_pred.detach()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y[0]
        index = y[1]
        y_true = label.long()
        logits = self.forward(x)
        y_pred = torch.argmax(logits, dim=1)

        # Log loss
        loss = self.loss(logits, y_true)
        self.log("loss/val", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        # Log f1 score
        self.val_f1_score(y_pred, y_true)
        self.log("f1/val", self.val_f1_score, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        # Retain validation step outputs
        validation_step_outputs = {
            "loss": loss,
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "logits": logits.detach(),
            "index": index
        }
        self.validation_step_outputs.append(validation_step_outputs)
        return validation_step_outputs

    def on_validation_epoch_end(self):
        # Free memory
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        label = y[0]
        index = y[1]
        y_true = label.long()
        logits = self.forward(x)
        y_pred = torch.argmax(logits, dim=1)

        # Retain test step outputs
        test_step_outputs = {
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "logits": logits.detach(),
            "index": index
        }
        self.test_step_outputs.append(test_step_outputs)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_true = torch.cat([out['y_true'] for out in outputs])
        y_pred = torch.cat([out['y_pred'] for out in outputs])
        self.test_f1_score(y_pred, y_true)
        self.log("f1/test", self.test_f1_score, prog_bar=True, logger=True)
        # Free memory
        self.test_step_outputs.clear()
        logging.info("Testing of model completed.")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        label = y[0]
        index = y[1]
        y_true = label.long()
        logits = self.forward(x)
        y_pred = torch.argmax(logits, dim=1)

        # Retain predict step outputs
        predict_step_outputs = {
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "logits": logits.detach(),
            "index": index
        }
        self.predict_step_outputs.append(predict_step_outputs)
        return predict_step_outputs

    def configure_optimizers(self):
        # Adheres to the dictionary format for LightningModule.configure_optimizers() API
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1.E-3, momentum=0.9, nesterov=True, weight_decay=1.E-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "name": "lr" # Name to use for logging
            }
        }

    def _load_model(
        self,
        backbone: str,
        num_classes: int,
        num_channels: int,
        weights: Optional[str] = None,
        **kwargs
    ):
        """
        Loads a CNN model from the torchvision library and modifies layers.

        Currently only the DenseNet, ResNet and ResNext architectures are supported.
        However, this function can be expanded to include more models and customizations.

        Parameters
        ----------
        backbone : str
            The name of the model to load (e.g., 'densenet161').
        num_classes : int
            The number of classes for the last layer.
        num_channels : int
            The number of input channels.
        weights : Optional[str]
            The name of the pre-trained weights to load.
        Returns
        -------
        model : torch.nn.Module
            The loaded model with the adapted first/last layer.
        """
        if hasattr(torchvision.models, backbone):
            # Disable SSL verification for torchvision models
            ssl._create_default_https_context = ssl._create_unverified_context
            # Load the torchvision model
            model = getattr(torchvision.models, backbone)(weights=weights, **kwargs)
            if "densenet" in backbone:
                num_ftrs = model.classifier.in_features
                model.classifier = torch.nn.Linear(num_ftrs, num_classes)
                if num_channels != 3:
                    model.features.conv0 = torch.nn.Conv2d(
                        num_channels, model.features.conv0.out_channels,
                        kernel_size=model.features.conv0.kernel_size,
                        stride=model.features.conv0.stride,
                        padding=model.features.conv0.padding,
                        bias=model.features.conv0.bias
                    )
            elif "resnet" in backbone or "resnext" in backbone:
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, num_classes)
                if num_channels != 3:
                    model.conv1 = torch.nn.Conv2d(
                        num_channels, model.conv1.out_channels,
                        kernel_size=model.conv1.kernel_size,
                        stride=model.conv1.stride,
                        padding=model.conv1.padding,
                        bias=model.conv1.bias
                    )
            else:
                raise NotImplementedError(f'Backbone {backbone} not currently implemented.')
        else:
            raise ValueError(f'Backbone {backbone} not found in torchvision models.')
        return model


RXRX2_DEFAULT = {
    "backbone": "densenet161",
    "num_classes": 2,
    "num_channels": 6,
    "loss": torch.nn.CrossEntropyLoss(reduction="mean")
}
