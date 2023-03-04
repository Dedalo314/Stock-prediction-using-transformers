import logging

from torch import Tensor, optim, nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

class LightningClassifier(pl.LightningModule):
    r"""
    Classifier abstraction for different models

    The model passed to the init is used in the forward.
    """
    def __init__(self, cfg, classifier) -> None:
        super(LightningClassifier, self).__init__()
        self.classifier = classifier.to(self.device)
        self.metric = nn.BCELoss()
        self.val_accuracy = Accuracy(task="binary")
        self.train_accuracy = Accuracy(task="binary")
        self._conf = cfg
        self.learning_rate = cfg.lr
        
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        pred = self(X)
        y = y.type(dtype=pred.dtype)
        loss = self.metric(pred, y)
        self.train_accuracy(pred, y)
        self.log("Loss/train", loss)
        self.log(
            "Accuracy/train", 
            self.train_accuracy, 
            on_step=True, 
            on_epoch=False
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        self._shared_eval(test_batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        logging.debug(f"X: {X}")
        logging.debug(f"y: {y}")
        pred = self(X)
        logging.debug(f"pred: {pred}")
        y = y.type(dtype=pred.dtype)
        loss = self.metric(pred, y)
        if prefix == "val": 
            self.val_accuracy(pred, y)
        self.log(f"Loss/{prefix}", loss)
        if prefix == "val": 
            self.log(
                f"Accuracy/{prefix}", 
                self.val_accuracy, 
                on_step=True, 
                on_epoch=True
            )