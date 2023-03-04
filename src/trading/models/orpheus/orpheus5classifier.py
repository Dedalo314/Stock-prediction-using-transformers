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
                      TransformerEncoder, TransformerEncoderLayer, MaxPool1d,
                      Sigmoid, ParameterList, InstanceNorm1d)
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from tqdm import tqdm
import pytorch_lightning as pl

from trading.models.custom_transformer import (PositionalEncoding, CustomTransformerEncoderLayer)

class Orpheus5Classifier(Module):
    r"""Transformer decoder built as a CNN encoder, 
    a transformer encoder and FFN classifier.

    Improvements in encoder w.r.t. Orpheus4Classifier:
    - Datetime is encoded using time2vec
    - Embedding for classification added to transformer (similar to CLS in BERT)
    - Instance norm instead of LayerNorm (apart from the Transformer Encoder)
    - Dropout not applied in CNN. It is typical not to apply it
    """
    def __init__(self, cfg) -> None:
        super(Orpheus5Classifier, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.time2vec_freqs = ParameterList([
            Parameter(
                torch.nn.init.uniform_(torch.zeros(1), -1, 1), 
                requires_grad=True
            ) 
        for i in range(2*cfg.num_time_features_to_expand)])
        self.time2vec_phases = ParameterList([
            Parameter(
                torch.nn.init.uniform_(torch.zeros(1), -1 , 1), 
                requires_grad=True
            )
        for i in range(2*cfg.num_time_features_to_expand)])
        
        self._num_time_features_to_expand = cfg.num_time_features_to_expand
        self._in_channels = cfg.in_channels
        self.norm_input = torch.nn.InstanceNorm1d(
            cfg.max_seq_length, eps=cfg.layer_norm_eps)
        self.stride_1 = cfg.conv_strides[0]
        self.kernel_size_1 = cfg.conv_kernel_sizes[0]
        # After conv1, 2687 frames
        self.stride_pool_1 = cfg.pool_strides[0]
        self.kernel_size_pool_1 = cfg.pool_kernel_sizes[0]
        # After pool1, 1343 frames
        self.stride_2 = cfg.conv_strides[1]
        self.kernel_size_2 = cfg.conv_kernel_sizes[1]
        # After conv2, 671 frames
        self.stride_pool_2 = cfg.pool_strides[1]
        self.kernel_size_pool_2 = cfg.pool_kernel_sizes[1]
        # After pool2, 335 frames
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.conv1d_1 = Conv1d(in_channels=cfg.in_channels, out_channels=(cfg.d_model // 2), kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect')
        self.leaky_relu_1 = LeakyReLU()
        self.maxpool1d_1 = MaxPool1d(kernel_size=self.kernel_size_pool_1, stride=self.stride_pool_1, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.norm_conv1d_1 = InstanceNorm1d(
            cfg.d_model // 2, eps=cfg.layer_norm_eps
        )
        self.conv1d_2 = Conv1d(in_channels=(cfg.d_model // 2), out_channels=cfg.d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect')
        self.leaky_relu_2 = LeakyReLU()
        self.maxpool1d_2 = MaxPool1d(kernel_size=self.kernel_size_pool_2, stride=self.stride_pool_2, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.norm_conv1d_2 = InstanceNorm1d(
            cfg.d_model, eps=cfg.layer_norm_eps
        )

        self.cls = Parameter(
            torch.nn.init.uniform_(
                torch.zeros(1, 1, cfg.d_model), 
                -1/math.sqrt(cfg.d_model), 
                1/math.sqrt(cfg.d_model)
            ),
            requires_grad=True
        )
        self.positional_encoding = PositionalEncoding(
            num_hiddens=cfg.d_model, dropout=cfg.transformer_dropout, max_len=cfg.max_seq_length
        )
        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=cfg.d_model, 
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.transformer_dropout,
            layer_norm_eps=cfg.layer_norm_eps,
            batch_first=cfg.batch_first,
            norm_first=cfg.norm_first
        )
        norm_out_trafo = torch.nn.InstanceNorm1d(
            cfg.d_model, eps=cfg.layer_norm_eps
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=cfg.n_layers,
            norm=norm_out_trafo,
            enable_nested_tensor=True
        )

        self.dropout_1 = Dropout(cfg.final_dropout)
        self.linear_1 = Linear(in_features=cfg.d_model, out_features=1)
        self.activation_out = Sigmoid()

    def _get_conv_output_length(self, kernel_size: float, stride: float, prev_length: float, 
                            dilation: float = 1, padding: float = 0) -> float:
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # and https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        return math.floor((prev_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def _add_classification_embedding(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                self.cls.expand(x.shape[0], -1, -1),
                x
            ], 
            dim=1
        )

    def time2vec(self, x: Tensor) -> Tensor:
        # Time2vec approach for encoding periodic time values
        # x shape (batch_size, features, sequence_length)
        # Assume features to add at the end
        y = torch.zeros((x.shape[0], self._in_channels, x.shape[2]), device=x.device)

        y[:, :x.shape[1], :] = torch.clone(x)
        num_features_to_add = (self._in_channels - x.shape[1]) / 2
        assert num_features_to_add == self._num_time_features_to_expand
        for i in range(self._num_time_features_to_expand):
            y[:, x.shape[1]+2*i, :] = torch.cos(
                self.time2vec_freqs[2*i]*x[:, x.shape[1] - self._num_time_features_to_expand + i, :] + self.time2vec_phases[2*i]
            )
            y[:, x.shape[1]+2*i+1, :] = torch.sin(
                self.time2vec_freqs[2*i + 1]*x[:, x.shape[1] - self._num_time_features_to_expand + i, :] + self.time2vec_phases[2*i + 1]
            )
        return y
        
    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            src: the input to the encoder (required).
        """
        # Time2vec https://arxiv.org/abs/1907.05321
        x = self.time2vec(x)
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = self.norm_input(x)
        x = self.conv1d_1(x)
        x = self.leaky_relu_1(x)
        x = self.maxpool1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_2(x)
        x = self.leaky_relu_2(x)
        x = self.maxpool1d_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_2(x)
        # Add classification embedding
        x = self._add_classification_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # Extract only cls output
        x = x[:, 0, :]
        # Final linear layer for classification
        x = self.dropout_1(x)
        x = self.activation_out(self.linear_1(x)).squeeze(dim=-1)
        return x