from torch.utils.tensorboard import SummaryWriter
import torch
import logging
import os
import datetime
from tqdm import tqdm

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from trading.utils import import_class

class LightningDataModule(pl.LightningDataModule):
    """
    Module to load datasets composed of financial values
    and datetimes.
    """

    def __init__(self, cfg):
        super(LightningDataModule, self).__init__()
        self._conf = cfg
        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        dataset_class = import_class(self._conf.dataset.dataset_class)

        self.train_data = dataset_class(cfg=self._conf.dataset, split="train")
        self.validation_data = dataset_class(cfg=self._conf.dataset, split="validation")
        self.test_data = dataset_class(cfg=self._conf.dataset, split="test")
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data, 
            batch_size=self._conf.train.batch_size, 
            shuffle=self._conf.train.shuffle
        )
        return train_dataloader
    
    def val_dataloader(self):
        validation_dataloader = DataLoader(
            self.validation_data, 
            batch_size=self._conf.val.batch_size, 
            shuffle=False
        )
        return validation_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_data, 
            batch_size=self._conf.test.batch_size, 
            shuffle=False
        )
        return test_dataloader
