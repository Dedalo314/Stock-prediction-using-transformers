import logging
import math
import glob
from random import randint
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from trading.utils import get_logger


class MultidifferenceDateBinaryClassDataset(Dataset):
    """
    Every item is a random chunk from a random financial
    product in a given folder.

    Each item contains N rows in this order:
    - financial_values: one row per col specified
    - differential_financial_values: one row per col specified
    - year_values
    - month_values
    - day_values
    """

    def __init__(self, cfg, split: str = "train",
                 transform=None, target_transform=None):
        super(MultidifferenceDateBinaryClassDataset, self).__init__()

        self._cfg = cfg
        self.logger = logging.getLogger(__name__)

        if split == "train":
            input_folder = self._cfg.train_input_folder
        elif split == "validation":
            input_folder = self._cfg.val_input_folder
        elif split == "test":
            input_folder = self._cfg.test_input_folder
        else:
            raise ValueError(f"Split {split} not recognized")

        self.logger.info(f"Folder {input_folder} for {split} split")
        self.financial_values = {
            name: [] for name in self._cfg.financial_values_cols
        }
        self.differential_financial_values = {
            name: [] for name in self._cfg.differential_financial_values_cols
        }
        self.year_values = []
        self.month_values = []
        self.day_values = []
        self.financial_product_len = []
        for csv_file in tqdm(glob.glob(os.path.join(input_folder, "*.csv"))):
            forbidden_substr = set([",,"])
            bad_file = False
            with open(csv_file) as f:    
                for line in f:
                    if self._should_remove_line(line, forbidden_substr):  
                        bad_file = True
            if bad_file:
                self.logger.warning(f"Skip bad file {csv_file}")
                continue

            self.logger.debug(f"Read {csv_file}")
            try:
                df = pd.read_csv(csv_file, sep=self._cfg.sep)
            except Exception as ex:
                self.logger.warning(f"File {csv_file} corrupted. Skipping")
                continue
            df[self._cfg.datetime_col] = pd.to_datetime(
                df[self._cfg.datetime_col], format="%d-%m-%Y"
            )
        
            differential_financial_values, financial_values, year_values, month_values, day_values = self._get_values(df)
            max_index = len(df) - (self._cfg.num_values_in + self._cfg.num_values_to_avg - 1)
            if max_index <= 0:
                continue
            for col in self._cfg.financial_values_cols:
                self.financial_values[col].append(financial_values[col])
            for col in self._cfg.differential_financial_values_cols:
                self.differential_financial_values[col].append(differential_financial_values[col])
            self.year_values.append(year_values)
            self.month_values.append(month_values)
            self.day_values.append(day_values)
            self.financial_product_len.append(len(df))

        self.num_values_in = self._cfg.num_values_in
        self.num_values_to_avg = self._cfg.num_values_to_avg
        self.transform = transform
        self.target_transform = target_transform
        self.logger.info(f"{len(self.financial_product_len)} files in {split} split")

    def __len__(self):
        return len(self.financial_product_len)

    def __getitem__(self, idx):
        """
        The target is done taking the average of the next financial
        values of the column specified in target_col.
        """
        start_pos = self._get_random_start_index(
            financial_product_len=self.financial_product_len[idx]
        )
        src_seq = np.array(
            [
                self.financial_values[col][idx][
                    start_pos:start_pos + self.num_values_in
                ] for col in self._cfg.financial_values_cols
            ] + [
                self.differential_financial_values[col][idx][
                    start_pos:start_pos + self.num_values_in
                ] for col in self._cfg.differential_financial_values_cols
            ] + [
                self.year_values[idx][start_pos:start_pos + self.num_values_in],
                self.month_values[idx][start_pos:start_pos + self.num_values_in],
                self.day_values[idx][start_pos:start_pos + self.num_values_in]
            ]
        )
        if self._cfg.target_col in self.financial_values.keys():
            avg_next_stock_values = np.mean(
                self.financial_values[self._cfg.target_col][idx][
                    start_pos + self.num_values_in:start_pos + self.num_values_in + self.num_values_to_avg
                ]
            )
            label = np.array(
                1.0 if avg_next_stock_values > self.financial_values[self._cfg.target_col][idx][start_pos + self.num_values_in - 1]
                else 0.0
            )
        else:
            net_difference = np.sum(
                self.differential_financial_values[self._cfg.target_col][idx][
                    start_pos + self.num_values_in:start_pos + self.num_values_in + self.num_values_to_avg
                ]
            )
            label = np.array(
                1.0 if net_difference > 0 else 0.0
            )
        if self.transform:
            src_seq = self.transform(src_seq)
        if self.target_transform:
            label = self.target_transform(label)
        return src_seq, label

    def _get_random_start_index(self, financial_product_len: int) -> int:
        max_index = financial_product_len - (self.num_values_in + self.num_values_to_avg - 1)
        return randint(0, max_index - 1)

    def _get_values(self, df: pd.DataFrame):
        financial_values = {}
        differential_financial_values = {}
        for col in self._cfg.financial_values_cols:
            financial_values[col] = np.asarray(df[col])
        for col in self._cfg.differential_financial_values_cols:
            df[col] = df[col].diff()
            df.at[0, col] = 0
            differential_financial_values[col] = np.asarray(df[col])
        year_values = np.asarray(df[self._cfg.datetime_col].dt.year)
        month_values = np.asarray(df[self._cfg.datetime_col].dt.month)
        day_values = np.asarray(df[self._cfg.datetime_col].dt.day)
        return differential_financial_values, financial_values, year_values, month_values, day_values

    def _should_remove_line(self, line: str, stop_words: list):
        return any([word in line for word in stop_words])
