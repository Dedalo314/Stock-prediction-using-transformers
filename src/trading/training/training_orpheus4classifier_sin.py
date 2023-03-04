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
from torch.utils.tensorboard import SummaryWriter

from trading.datasets.datasets import EzSinBinaryClassDataset
from trading.utils import train, validation
from trading.models.old_orpheus import Orpheus4Classifier

filehandler = logging.FileHandler(filename='logs/transformer_orpheus4classifier_sin.log', mode="w+")
logging.basicConfig(encoding='utf-8', level=logging.INFO, handlers=[filehandler])

try:
    logging.info("")
    logging.info(
        "######################################################################")
    logging.info(f"Starting new run {datetime.datetime.now()}")
    logging.info(
        "######################################################################")

    PERCENTAGE_TRAINING = 0.67
    BATCH_SIZE = 11
    D_MODEL = 64
    N_HEAD = 1
    NUM_VALUES_IN = 64*7*4*3  # 1 year of data to predict one week average
    NUM_TO_AVG = 1
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
    train_data = EzSinBinaryClassDataset(length=len(df["close"]), num_values_in=NUM_VALUES_IN,
                                num_to_avg=NUM_TO_AVG, for_training=True)
    validation_data = EzSinBinaryClassDataset(length=len(df["close"]), num_values_in=NUM_VALUES_IN,
                                     num_to_avg=NUM_TO_AVG, for_training=False)

    # Dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True)

    # Transformer Decoder Layer
    orpheus = Orpheus4Classifier(d_model=D_MODEL, nhead=N_HEAD, max_seq_length=NUM_VALUES_IN,
                       in_channels=1, n_layers=1, dim_feedforward=256, batch_first=True, 
                       device=device, dtype=torch.float64)

    epochs = 5
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(orpheus.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1, verbose=True)
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
            ckpt_name = f"orpheus4classifier-checkpoint-epoch-{epoch}-step-{step}.pth"
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
    ckpt_name = f"orpheus4classifier-checkpoint-epoch-{epochs - 1}-step-{step}.pth"
    if not os.path.exists(f"Checkpoints/{ckpt_name}"):
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': orpheus.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'validation_loss': validation_loss
        }, f"Checkpoints/{ckpt_name}")
        logging.info(f"Model saved to Models/{ckpt_name}")
except Exception as ex:
    logging.exception(ex)
    raise
