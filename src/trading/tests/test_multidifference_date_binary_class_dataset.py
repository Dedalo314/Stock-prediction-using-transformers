import pytest

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from trading.data.lightning_data_module import LightningDataModule

def _load_data(cfg_data):
    data_module = LightningDataModule(cfg=cfg_data)
    data_module.prepare_data()
    return data_module

def test_multidifference_date_binary_class_dataset():
    cfg_data = OmegaConf.load('trading/training/conf/data/multidifference_date_binaryclass.yaml')
    cfg_data.dataset.train_input_folder = "../Data/nasdaq_train"
    cfg_data.dataset.val_input_folder = "../Data/nasdaq_val"
    cfg_data.dataset.test_input_folder = "../Data/nasdaq_test"

    data_module = _load_data(cfg_data=cfg_data)

    # Check shape
    val_dataloader = data_module.val_dataloader()
    train_dataloader = data_module.train_dataloader()
    test_dataloader = data_module.test_dataloader()
    for X, y in val_dataloader:
        assert X.shape[0] == cfg_data.val.batch_size
        assert X.shape[1] == len(cfg_data.dataset.financial_values_cols) + len(cfg_data.dataset.differential_financial_values_cols) + 3
        assert X.shape[2] == cfg_data.dataset.num_values_in
        break
    for X, y in train_dataloader:
        assert X.shape[0] == cfg_data.train.batch_size
        assert X.shape[1] == len(cfg_data.dataset.financial_values_cols) + len(cfg_data.dataset.differential_financial_values_cols) + 3
        assert X.shape[2] == cfg_data.dataset.num_values_in
        break
    for X, y in test_dataloader:
        assert X.shape[0] == cfg_data.test.batch_size
        assert X.shape[1] == len(cfg_data.dataset.financial_values_cols) + len(cfg_data.dataset.differential_financial_values_cols) + 3
        assert X.shape[2] == cfg_data.dataset.num_values_in
        break