from trading.models.orpheus.orpheus5classifier import Orpheus5Classifier
from trading.training.utils import train_epochs
from trading.utils import get_logger
from trading.datasets.datasets import StockDatetimeBinaryClassDataset
import datetime
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(filename='logs/transformer_orpheus5classifier.log')

try:
    logger.info("")
    logger.info(
        "######################################################################")
    logger.info(f"Starting new run {datetime.datetime.now()}")
    logger.info(
        "######################################################################")

    PERCENTAGE_TRAINING = 0.67
    BATCH_SIZE = 11
    D_MODEL = 64
    N_HEAD = 1
    NUM_STOCK_VALUES_IN = 64*7*4*3  # 1 year of data to predict one week average
    NUM_STOCK_TO_AVG = 64
    CHECKPOINT_EPOCH_PERIOD = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Model dimension: {D_MODEL}")
    logger.info(f"No. heads self-attention: {N_HEAD}")

    # Dataset
    dataset_file = os.path.join('../Data',
                                "tsla_downloaded_all.csv")
    df = pd.read_csv(dataset_file, sep=",")
    train_data = StockDatetimeBinaryClassDataset(df=df, num_stock_values_in=NUM_STOCK_VALUES_IN,
                                                 num_stock_to_avg=NUM_STOCK_TO_AVG, for_training=True)
    validation_data = StockDatetimeBinaryClassDataset(df=df, num_stock_values_in=NUM_STOCK_VALUES_IN,
                                                      num_stock_to_avg=NUM_STOCK_TO_AVG, for_training=False)

    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True)

    orpheus = Orpheus5Classifier(d_model=D_MODEL, nhead=N_HEAD, max_seq_length=NUM_STOCK_VALUES_IN,
                                 in_channels=14, n_layers=1, dim_feedforward=256, batch_first=True,
                                 device=device, dtype=torch.float64)

    epochs = 5
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(orpheus.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1, verbose=True)
    train_epochs(epochs=epochs, train_dataloader=train_dataloader, model=orpheus, optimizer=optimizer,
                 loss_fn=loss_fn, device=device, validation_dataloader=validation_dataloader,
                 scheduler=scheduler, checkpoint_epoch_period=CHECKPOINT_EPOCH_PERIOD)
except Exception as ex:
    logger.exception(ex)
    raise
