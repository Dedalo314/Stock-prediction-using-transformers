"""
This script trains a transformer-based decoder layer to
predict daily stock values. There are 64 stock data points
per day, so the embeddings are going to have 64 values.
"""
import datetime
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trading.datasets.datasets import CustomStockDataset
from trading.utils import train, validation
from trading.models.custom_transformer import CustomTransformerDecoder


logging.basicConfig(filename='logs/training_decoder.log', encoding='utf-8', level=logging.INFO)

try:
    logging.info("")
    logging.info(
        "######################################################################")
    logging.info(f"Starting new run {datetime.datetime.now()}")
    logging.info(
        "######################################################################")

    PERCENTAGE_TRAINING = 0.67
    BATCH_SIZE = 16
    # Our inputs are 64 values value, the close values per day
    D_MODEL = 64
    N_HEAD = 4
    SEQ_LENGTH = DAYS_FOR_PREDICTION = 30
    SEQ_LENGTH_1D = DAYS_FOR_PREDICTION * STOCK_VALUES_PER_DAY
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Device: {device}")
    logging.info(f"Seq. length: {SEQ_LENGTH}")
    logging.info(f"Model dimension: {D_MODEL}")
    logging.info(f"No. heads self-attention: {N_HEAD}")

    # Dataset
    dataset_file = os.path.join('../Data',
                                "tsla_downloaded_all.csv")
    train_data = CustomStockDataset(
        stock_file=dataset_file, stock_values_per_day=STOCK_VALUES_PER_DAY, seq_length=SEQ_LENGTH, for_training=True)
    validation_data = CustomStockDataset(
        stock_file=dataset_file, stock_values_per_day=STOCK_VALUES_PER_DAY, seq_length=SEQ_LENGTH)

    # Dataloaders
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

    # Transformer Decoder Layer
    transformer = CustomTransformerDecoder(
        d_model=D_MODEL, nhead=N_HEAD, seq_length=SEQ_LENGTH, n_layers=6, dim_feedforward=1024, 
        batch_first=True, device=device, dtype=torch.float64)
    old_model_name = "transformer-2022-10-31 19:00:25.397364.pth"
    transformer.load_state_dict(torch.load(f"Models/{old_model_name}"))
    epochs = 500
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    writer = SummaryWriter()
    it_train, it_validation = 0, 0
    for t in tqdm(range(epochs)):
        it_train = train(train_dataloader, transformer, loss_fn, optimizer, writer, it_train, device)
        it_validation, validation_loss = validation(validation_dataloader, transformer, loss_fn, writer, it_validation, device)
        scheduler.step(validation_loss)
        writer.add_scalar("lr/train", optimizer.param_groups[0]["lr"], t)
        writer.flush()
    logging.info("Done!")
    writer.close()
    model_name = f"transformer-{datetime.datetime.now()}.pth"
    torch.save(transformer.state_dict(), f"Models/{model_name}")
    logging.info(f"Model saved to Models/{model_name}")
except Exception as ex:
    logging.exception(ex)
    raise