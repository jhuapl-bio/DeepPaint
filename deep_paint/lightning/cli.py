"""PyTorch Lightning custom CLI."""
import os
import logging

import torch
import pandas as pd
from lightning.pytorch.cli import LightningCLI


class DeepLightningCLI(LightningCLI):
    """
    Custom LightningCLI class.

    Refer to the LightningCLI documentation for more information.
    (https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html)
    """
    def after_fit(self):
        """
        Run validation after training.
        """
        # Get best model checkpoint
        model_path = self.trainer.checkpoint_callback.best_model_path
        best_model_path = model_path.replace("epoch=", "best_epoch=")
        os.rename(model_path, best_model_path)
        logging.info(f"Best model checkpoint: {best_model_path}")

        # Run validation epoch
        best_model = self._model_class.load_from_checkpoint(best_model_path)
        self.trainer.validate(best_model, self.datamodule)

    def after_predict(self):
        """
        Run post-prediction tasks.
        """
        # Unpack outputs
        outputs = self.model.predict_step_outputs
        y_true = torch.cat([out['y_true'] for out in outputs]).cpu()
        y_pred = torch.cat([out['y_pred'] for out in outputs]).cpu()
        index = torch.cat([out['index'] for out in outputs]).cpu()
        logits = torch.cat([out['logits'] for out in outputs]).cpu().numpy()
        logits_cols = [f"logits_{i}" for i in range(self.model.num_classes)]

        # Create dataframe of predictions
        predictions_df = pd.DataFrame(logits, columns=logits_cols)
        predictions_df["y_true"] = y_true
        predictions_df["y_pred"] = y_pred
        predictions_df["index"] = index

        # Save predictions to csv
        save_path = os.path.join(self.trainer.logger.log_dir, "predictions.csv")
        predictions_df.to_csv(save_path, index=False)
