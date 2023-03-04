import copy
import functools
import logging
import math
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import (Dropout, LayerNorm, Linear, Module, ModuleList,
                      MultiheadAttention, Conv1d, LeakyReLU, ConvTranspose1d,
                      TransformerEncoder, TransformerEncoderLayer)
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from tqdm import tqdm

from trading.utils import get_logger

class CustomStockDataset(Dataset):
    def __init__(self, stock_file: str, stock_values_per_day: int, seq_length: int, sep: str = ",", percentage_training: float = 0.67,
                 for_training: bool = False, transform=None, target_transform=None):
        df_tsla = pd.read_csv(stock_file, sep=",")

        if for_training:
            self.stock_values = self._get_stock_values_train(
                df_tsla, percentage_training, stock_values_per_day)
        else:
            self.stock_values = self._get_stock_values_validation(
                df_tsla, percentage_training, stock_values_per_day)

        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.stock_values.shape[0] - self.seq_length

    def __getitem__(self, idx):
        src_seq = self.stock_values[idx:idx + self.seq_length]
        tgt_seq = self.stock_values[idx + 1:idx + self.seq_length + 1]
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            tgt_seq = self.target_transform(tgt_seq)
        return src_seq, tgt_seq

    def _convert_to_multiple_of(self, n, tgt):
        """Returns the highest number multiple of tgt and lower than n
        n and tgt are positive integers.
        """
        return (n // tgt) * tgt

    def _get_stock_values_train(self, df: pd.DataFrame, percentage_training: float, stock_values_per_day: int) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_row = math.ceil(len(df)*percentage_training)
        # Ensure that the number of training values are
        # a multiple of stock_values_per_day, so that
        # it can be reshaped to a (n,64).
        stock_values_train = df["close"].iloc[:self._convert_to_multiple_of(
            first_validation_row, stock_values_per_day)].to_numpy()
        logging.debug(f"Stock values train shape: {stock_values_train.shape}")
        stock_values_train = np.reshape(
            stock_values_train, (-1, stock_values_per_day))
        logging.debug(
            f"Stock values train after reshaping: {stock_values_train.shape}")
        return stock_values_train

    def _get_stock_values_validation(self, df: pd.DataFrame, percentage_training: float, stock_values_per_day: int) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_row = math.ceil(len(df)*percentage_training)
        # Ensure that the number of validation values are
        # a multiple of stock_values_per_day, so that
        # it can be reshaped to a (n,64).
        stock_values_validation = df["close"].iloc[self._convert_to_multiple_of(
            first_validation_row, stock_values_per_day):self._convert_to_multiple_of(
            len(df), stock_values_per_day)].to_numpy()
        logging.debug(
            f"Stock values validation shape: {stock_values_validation.shape}")
        stock_values_validation = np.reshape(
            stock_values_validation, (-1, stock_values_per_day))
        logging.debug(
            f"Stock values validation after reshaping: {stock_values_validation.shape}")
        return stock_values_validation


class EzStockDataset(Dataset):
    def __init__(self, close_data: list, num_values_in: int, num_values_out: int,
                 percentage_training: float = 0.67, for_training: bool = False,
                 transform=None, target_transform=None):

        if for_training:
            self.stock_values = self._get_stock_values_train(
                close_data, percentage_training)
        else:
            self.stock_values = self._get_stock_values_validation(
                close_data, percentage_training)

        self.num_values_in = num_values_in
        self.num_values_out = num_values_out
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.stock_values.shape[0] - (self.num_values_in + self.num_values_out - 1)

    def __getitem__(self, idx):
        src_seq = self.stock_values[idx:idx + self.num_values_in]
        tgt_seq = self.stock_values[idx + self.num_values_in:idx +
                                    self.num_values_in + self.num_values_out]
        src_seq = np.reshape(src_seq, (1, src_seq.shape[0]))
        tgt_seq = np.reshape(tgt_seq, (1, tgt_seq.shape[0]))
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            tgt_seq = self.target_transform(tgt_seq)
        return src_seq, tgt_seq

    def _get_stock_values_train(self, close_data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(close_data)*percentage_training)
        stock_values_train = np.asarray(close_data[:first_validation_idx])
        logging.debug(f"Stock values train shape: {stock_values_train.shape}")
        return stock_values_train

    def _get_stock_values_validation(self, close_data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(close_data)*percentage_training)
        stock_values_validation = np.asarray(close_data[first_validation_idx:])
        logging.debug(
            f"Stock values validation shape: {stock_values_validation.shape}")
        return stock_values_validation


class EzStockBinaryClassDataset(Dataset):
    def __init__(self, close_data: list, num_values_in: int, num_values_to_avg: int,
                 percentage_training: float = 0.67, for_training: bool = False,
                 transform=None, target_transform=None):

        if for_training:
            self.stock_values = self._get_stock_values_train(
                close_data, percentage_training)
        else:
            self.stock_values = self._get_stock_values_validation(
                close_data, percentage_training)

        self.num_values_in = num_values_in
        self.num_values_to_avg = num_values_to_avg
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.stock_values.shape[0] - (self.num_values_in + self.num_values_to_avg - 1)

    def __getitem__(self, idx):
        src_seq = self.stock_values[idx:idx + self.num_values_in]
        avg_next_stock_values = np.mean(self.stock_values[idx + self.num_values_in:idx +
                                                          self.num_values_in + self.num_values_to_avg])
        label = 1 if avg_next_stock_values > src_seq[-1] else 0
        src_seq = np.reshape(src_seq, (1, src_seq.shape[0]))
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.tensor(src_seq, dtype=torch.float64), torch.tensor(label, dtype=torch.float64)

    def _get_stock_values_train(self, close_data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(close_data)*percentage_training)
        stock_values_train = np.asarray(close_data[:first_validation_idx])
        logging.debug(f"Stock values train shape: {stock_values_train.shape}")
        return stock_values_train

    def _get_stock_values_validation(self, close_data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(close_data)*percentage_training)
        stock_values_validation = np.asarray(close_data[first_validation_idx:])
        logging.debug(
            f"Stock values validation shape: {stock_values_validation.shape}")
        return stock_values_validation


class ValueDatetimeBinaryClassDataset(Dataset):
    """
    Each item contains 6 rows in this order:
    - stock_values
    - year_values
    - month_values
    - day_values
    - hour_values
    - min_values
    """
    def __init__(self, cfg, split: str = "train",
                 transform=None, target_transform=None):
        super(ValueDatetimeBinaryClassDataset, self).__init__()
        
        self._conf = cfg
        self.logger = logging.getLogger(__name__)

        df = pd.read_csv(self._conf.input_csv, sep=self._conf.sep)
        df[self._conf.datetime_col] = pd.to_datetime(df[self._conf.datetime_col], infer_datetime_format=True)

        if split == "train":
            (self.financial_values, self.year_values, self.month_values,
            self.day_values, self.hour_values,
            self.min_values) = self._get_values_train(
                df, self._conf.percentage_training, self._conf.percentage_val_of_training)
        elif split == "validation":
            (self.financial_values, self.year_values, self.month_values,
            self.day_values, self.hour_values,
            self.min_values) = self._get_values_validation(
                df, self._conf.percentage_training, self._conf.percentage_val_of_training)
        elif split == "test":
            (self.financial_values, self.year_values, self.month_values,
            self.day_values, self.hour_values,
            self.min_values) = self._get_values_test(
                df, self._conf.percentage_training)
        else:
            raise ValueError(f"Unrecognized split {split}")

        self.num_values_in = self._conf.num_values_in
        self.num_values_to_avg = self._conf.num_values_to_avg
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.financial_values.shape[0] - (self.num_values_in + self.num_values_to_avg - 1)

    def __getitem__(self, idx):
        avg_next_stock_values = np.mean(self.financial_values[idx + self.num_values_in:idx +
                                                          self.num_values_in + self.num_values_to_avg])
        src_seq = np.array([self.financial_values[idx:idx + self.num_values_in], 
                            self.year_values[idx:idx + self.num_values_in], 
                            self.month_values[idx:idx + self.num_values_in],
                            self.day_values[idx:idx + self.num_values_in], 
                            self.hour_values[idx:idx + self.num_values_in], 
                            self.min_values[idx:idx + self.num_values_in]])
        label = np.array(1.0 if avg_next_stock_values > self.financial_values[idx + self.num_values_in - 1] else 0.0)
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            label = self.target_transform(label)
        return src_seq, label

    def _get_values_train(self, df: pd.DataFrame, percentage_training: float, percentage_val_of_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(df)*percentage_training*(1 - percentage_val_of_training))
        df_train = df.iloc[:first_validation_idx]
        return self._get_values(df_train)

    def _get_values_validation(self, df: pd.DataFrame, percentage_training: float, percentage_val_of_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(df)*percentage_training*(1 - percentage_val_of_training))
        first_test_idx = math.ceil(len(df)*percentage_training)
        df_val = df.iloc[first_validation_idx:first_test_idx]
        return self._get_values(df_val)

    def _get_values_test(self, df: pd.DataFrame, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_test_idx = math.ceil(len(df)*percentage_training)
        df_test = df.iloc[first_test_idx:]
        return self._get_values(df_test)

    def _get_values(self, df: pd.DataFrame):
        financial_values = np.asarray(df[self._conf.value_col])
        year_values = np.asarray(df[self._conf.datetime_col].dt.year)
        month_values = np.asarray(df[self._conf.datetime_col].dt.month)
        day_values = np.asarray(df[self._conf.datetime_col].dt.day)
        hour_values = np.asarray(df[self._conf.datetime_col].dt.hour)
        min_values = np.asarray(df[self._conf.datetime_col].dt.minute)
        self.logger.debug(f"Values shape: {financial_values.shape}")
        return financial_values, year_values, month_values, day_values, hour_values, min_values


class EzSinBinaryClassDataset(Dataset):
    def __init__(self, length: int, num_values_in: int, num_to_avg: int,
                 percentage_training: float = 0.67, for_training: bool = False,
                 transform=None, target_transform=None):

        period = 100  # samples
        data = np.sin(np.linspace(0, 2*np.pi*length / period, length)).tolist()

        if for_training:
            self.time_values = self._get_values_train(
                data, percentage_training)
        else:
            self.time_values = self._get_values_validation(
                data, percentage_training)

        self.num_values_in = num_values_in
        self.num_to_avg = num_to_avg
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.time_values.shape[0] - (self.num_values_in + self.num_to_avg - 1)

    def __getitem__(self, idx):
        src_seq = self.time_values[idx:idx + self.num_values_in]
        avg_next_values = np.mean(self.time_values[idx + self.num_values_in:idx +
                                                   self.num_values_in + self.num_to_avg])
        label = 1 if avg_next_values > src_seq[-1] else 0
        src_seq = np.reshape(src_seq, (1, src_seq.shape[0]))
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.tensor(src_seq, dtype=torch.float64), torch.tensor(label, dtype=torch.float64)

    def _get_values_train(self, data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(data)*percentage_training)
        stock_values_train = np.asarray(data[:first_validation_idx])
        logging.debug(f"Stock values train shape: {stock_values_train.shape}")
        return stock_values_train

    def _get_values_validation(self, data: list, percentage_training: float) -> np.array:
        # Only percentage_training of the dataset for training
        first_validation_idx = math.ceil(len(data)*percentage_training)
        stock_values_validation = np.asarray(data[first_validation_idx:])
        logging.debug(
            f"Stock values validation shape: {stock_values_validation.shape}")
        return stock_values_validation
