import math
import os
import sys
import argparse
import logging

import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
import pytorch_lightning as pl

from trading.utils import get_logger, import_class
from trading.models.LightningClassifier import LightningClassifier
from trading.data.lightning_data_module import LightningDataModule

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data_module = LightningDataModule(cfg=cfg.data)

    model_class = import_class(cfg.model.model_class)

    logger.info(f"Batch size: {cfg.data.train.batch_size}")
    logger.info(f"Model dimension: {cfg.model.d_model}")
    logger.info(f"No. heads self-attention: {cfg.model.nhead}")

    model = model_class(cfg=cfg.model)
    classifier = LightningClassifier(cfg=cfg.model, classifier=model)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator, 
        devices=cfg.trainer.devices, 
        max_epochs=cfg.trainer.epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        auto_lr_find=cfg.trainer.auto_lr_find,
        accumulate_grad_batches=4
    )
    trainer.tune(classifier, data_module)
    if "checkpoint" in cfg.model:
        trainer.fit(classifier, data_module, ckpt_path=cfg.model.checkpoint)
    else:
        trainer.fit(classifier, data_module)

try:
    main()
except Exception as ex:
    logger.exception(ex)
    raise
