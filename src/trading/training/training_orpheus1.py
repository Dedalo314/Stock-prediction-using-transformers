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

from trading.datasets.datasets import EzStockDataset
from trading.utils import train, validation
from trading.models.old_orpheus import Orpheus1DEPRECATED

logging.basicConfig(filename='logs/transformer_orpheus1.log', encoding='utf-8', level=logging.INFO)

try:
    logging.info("")
    logging.info(
        "######################################################################")
    logging.info(f"Starting new run {datetime.datetime.now()}")
    logging.info(
        "######################################################################")

    PERCENTAGE_TRAINING = 0.67
    BATCH_SIZE = 16
    D_MODEL = 512
    N_HEAD = 4
    NUM_STOCK_VALUES_IN = 64*7*4*3  # almost 3 months of data
    NUM_STOCK_VALUES_OUT = 64*7  # 1 week out
    CHECKPOINT_EPOCH_PERIOD = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Model dimension: {D_MODEL}")
    logging.info(f"No. heads self-attention: {N_HEAD}")

    # Dataset
    dataset_file = os.path.join('../Data',
                                "tsla_downloaded_all.csv")
    df = pd.read_csv(dataset_file, sep=",")
    train_data = EzStockDataset(close_data=df["close"].tolist(), num_values_in=NUM_STOCK_VALUES_IN,
                                num_values_out=NUM_STOCK_VALUES_OUT, for_training=True)
    validation_data = EzStockDataset(close_data=df["close"].tolist(), num_values_in=NUM_STOCK_VALUES_IN,
                                     num_values_out=NUM_STOCK_VALUES_OUT, for_training=False)

    # Dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True)

    # Transformer Decoder Layer
    orpheus = Orpheus1DEPRECATED(d_model=D_MODEL, nhead=N_HEAD, max_seq_length=NUM_STOCK_VALUES_IN, out_channels=1,
                       in_channels=1, conv_kernel_sizes=[7, 4, 3], n_layers=6, dim_feedforward=2048,
                       batch_first=True, device=device, dtype=torch.float64)

    epochs = 20
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(orpheus.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.5, verbose=True)
    writer = SummaryWriter()
    step = 0
    for epoch in tqdm(range(epochs)):
        writer.add_scalar("lr/train", optimizer.param_groups[0]["lr"], epoch)
        step = train(train_dataloader, orpheus, loss_fn,
                     optimizer, writer, step, device)
        validation_loss = validation(
            validation_dataloader, orpheus, loss_fn, writer, epoch, device)
        scheduler.step()
        writer.flush()
        if epoch % CHECKPOINT_EPOCH_PERIOD == 0:
            ckpt_name = f"orpheus1-checkpoint-epoch-{epoch}-step-{step}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': orpheus.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'validation_loss': validation_loss
            }, f"Checkpoints/{ckpt_name}")
            logging.info(f"Checkpoint saved to Checkpoints/{ckpt_name}")
    logging.info("Done!")
    writer.close()
    ckpt_name = f"orpheus1-checkpoint-epoch-{epochs}-step-{step}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': orpheus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }, f"Checkpoints/{ckpt_name}")
    logging.info(f"Model saved to Models/{model_name}")
except Exception as ex:
    logging.exception(ex)
    raise
