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
                      Sigmoid, ParameterList)
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from tqdm import tqdm
from trading.models.custom_transformer import (PositionalEncoding, CustomTransformerDecoderLayerDEPRECATED,
                                       CustomTransformerEncoderLayer)


class Orpheus1DEPRECATED(Module):
    r"""DEPRECATED: Uses old Decoder layer with multihead attention, although multihead
    attention was not used so it is the same as Orpheus1. Thus, no need to retrain with
    Orpheus1 these models. Multihead attention removed just to reduce mem usage, but it 
    was not used in forward.

    Transformer decoder built as a CNN encoder, 
    a combination of multiple CustomTransformerDecoderLayer 
    and a transposed CNN decoder.
    """

    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int, out_channels: int, conv_kernel_sizes: list,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus1DEPRECATED, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # Conv reduces by more than 64*7 = 448 in the end (i.e., compresses info to a week)
        self.kernel_size_1 = conv_kernel_sizes[0]
        self.kernel_size_2 = conv_kernel_sizes[1]
        self.kernel_size_3 = conv_kernel_sizes[2]
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.stride_1 = conv_kernel_sizes[0]
        self.stride_2 = conv_kernel_sizes[1]
        self.stride_3 = conv_kernel_sizes[2]
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.conv1d_3 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_3,
                               stride=self.stride_3, padding=self.padding_3, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu = LeakyReLU()
        self.norm_input = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=max_seq_length)
        # output = regressor(torch.rand(seq_length, d_model), tgt_mask=tgt_mask)

        self.decoder_layers = torch.nn.ModuleList([CustomTransformerDecoderLayerDEPRECATED(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first,
            device=device, dtype=dtype, nhead=nhead, batch_first=batch_first) for i in range(n_layers)])

        # Lout of size Lin/(kernel_size_2 * kernel_size_3)
        self.deconv1d_1 = ConvTranspose1d(in_channels=d_model, out_channels=out_channels, kernel_size=self.kernel_size_1,
                                          stride=self.stride_1, padding=0, output_padding=0, groups=1, bias=True,
                                          dilation=1, padding_mode='zeros', **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        logging.debug(f"Size of input tensor before CNN: {x.size()}")
        x = self.conv1d_1(x)
        x = self.leaky_relu(x)
        x = self.conv1d_2(x)
        x = self.leaky_relu(x)
        x = self.conv1d_3(x)
        x = self.leaky_relu(x)
        logging.debug(f"Size of input tensor after CNN: {x.size()}")
        x = torch.transpose(x, 1, 2)
        x = self.norm_input(x)
        x = self.positional_embedding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        x = torch.transpose(x, 1, 2)
        x = self.deconv1d_1(x)
        logging.debug(f"Size of input tensor after transposed CNN: {x.size()}")
        return x


class Orpheus1(Module):
    r"""Transformer decoder built as a CNN encoder, 
    a combination of multiple CustomTransformerDecoderLayer 
    and a transposed CNN decoder.
    """

    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int, out_channels: int, conv_kernel_sizes: list,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus1, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # Conv reduces by more than 64*7 = 448 in the end (i.e., compresses info to a week)
        self.kernel_size_1 = conv_kernel_sizes[0]
        self.kernel_size_2 = conv_kernel_sizes[1]
        self.kernel_size_3 = conv_kernel_sizes[2]
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.stride_1 = conv_kernel_sizes[0]
        self.stride_2 = conv_kernel_sizes[1]
        self.stride_3 = conv_kernel_sizes[2]
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.conv1d_3 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_3,
                               stride=self.stride_3, padding=self.padding_3, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu = LeakyReLU()
        self.norm_input = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=max_seq_length)
        # output = regressor(torch.rand(seq_length, d_model), tgt_mask=tgt_mask)

        self.decoder_layers = torch.nn.ModuleList([CustomTransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first,
            device=device, dtype=dtype, nhead=nhead, batch_first=batch_first) for i in range(n_layers)])

        # Lout of size Lin/(kernel_size_2 * kernel_size_3)
        self.deconv1d_1 = ConvTranspose1d(in_channels=d_model, out_channels=out_channels, kernel_size=self.kernel_size_1,
                                          stride=self.stride_1, padding=0, output_padding=0, groups=1, bias=True,
                                          dilation=1, padding_mode='zeros', **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        logging.debug(f"Size of input tensor before CNN: {x.size()}")
        x = self.conv1d_1(x)
        x = self.leaky_relu(x)
        x = self.conv1d_2(x)
        x = self.leaky_relu(x)
        x = self.conv1d_3(x)
        x = self.leaky_relu(x)
        logging.debug(f"Size of input tensor after CNN: {x.size()}")
        x = torch.transpose(x, 1, 2)
        x = self.norm_input(x)
        x = self.positional_embedding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        x = torch.transpose(x, 1, 2)
        x = self.deconv1d_1(x)
        logging.debug(f"Size of input tensor after transposed CNN: {x.size()}")
        return x


class Orpheus2(Module):
    r"""Transformer decoder built as a CNN encoder, 
    a combination of multiple CustomTransformerDecoderLayer 
    and a transposed CNN decoder.

    Improvements:
    - Use LayerNorm between conv layers
    - LinearRegression of inputs added to outputs, so that only the error is predicted by the trafo
    - Each LeakyRelu a different instance
    """

    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int, out_channels: int, conv_kernel_sizes: list,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus2, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.stride_1 = self.kernel_size_1 = conv_kernel_sizes[0]
        self.stride_2 = self.kernel_size_2 = conv_kernel_sizes[1]
        self.stride_3 = self.kernel_size_3 = conv_kernel_sizes[2]
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_1 = LeakyReLU()
        self.norm_conv1d_1 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_2 = LeakyReLU()
        self.norm_conv1d_2 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_3 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_3,
                               stride=self.stride_3, padding=self.padding_3, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_3 = LeakyReLU()
        self.norm_conv1d_3 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=max_seq_length)
        # output = regressor(torch.rand(seq_length, d_model), tgt_mask=tgt_mask)

        self.encoder_layers = torch.nn.ModuleList([CustomTransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first,
            device=device, dtype=dtype, nhead=nhead, batch_first=batch_first) for i in range(n_layers)])
        self.norm_out_trafo = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)

        # Lout of size Lin/(kernel_size_2 * kernel_size_3)
        self.deconv1d_1 = ConvTranspose1d(in_channels=d_model, out_channels=out_channels, kernel_size=self.kernel_size_1,
                                          stride=self.stride_1, padding=0, output_padding=0, groups=1, bias=True,
                                          dilation=1, padding_mode='zeros', **factory_kwargs)
        self.lr_w = Parameter(torch.nn.init.normal_(torch.FloatTensor(
            max_seq_length), std=(1/max_seq_length)).to(device))
        self.lr_b = Parameter(torch.nn.init.constant_(
            torch.Tensor(1), 0).to(device))

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        lr_out = torch.sum(self.lr_w * x, -1).squeeze() + self.lr_b
        assert (lr_out.shape[0] == x.shape[0] and len(lr_out.shape) == 1)
        x = self.conv1d_1(x)
        x = self.leaky_relu_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_2(x)
        x = self.leaky_relu_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_3(x)
        x = self.leaky_relu_3(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_3(x)
        x = self.positional_embedding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.norm_out_trafo(x)
        x = torch.transpose(x, 1, 2)
        x = self.deconv1d_1(x)
        # Add result of linear regression (broadcasting, lr_out should be a scalar)
        x = x + lr_out.reshape(x.shape[0], 1, 1)
        return x


class Orpheus2Std(Module):
    r"""Transformer decoder built as a CNN encoder, 
    a combination of multiple CustomTransformerDecoderLayer 
    and a transposed CNN decoder.

    Improvements over Orpheus2:
    - Std Pytorch transformer encoder used

    """

    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int, out_channels: int, conv_kernel_sizes: list,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus2Std, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.stride_1 = self.kernel_size_1 = conv_kernel_sizes[0]
        self.stride_2 = self.kernel_size_2 = conv_kernel_sizes[1]
        self.stride_3 = self.kernel_size_3 = conv_kernel_sizes[2]
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_1 = LeakyReLU()
        self.norm_conv1d_1 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_2 = LeakyReLU()
        self.norm_conv1d_2 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_3 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size_3,
                               stride=self.stride_3, padding=self.padding_3, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_3 = LeakyReLU()
        self.norm_conv1d_3 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout, activation=activation,
                                                            layer_norm_eps=layer_norm_eps,
                                                            batch_first=batch_first,
                                                            norm_first=norm_first,
                                                            **factory_kwargs)
        norm_out_trafo = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.transformer_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                      num_layers=n_layers, norm=norm_out_trafo,
                                                      enable_nested_tensor=True)

        # Lout of size Lin/(kernel_size_2 * kernel_size_3)
        self.deconv1d_1 = ConvTranspose1d(in_channels=d_model, out_channels=out_channels, kernel_size=self.kernel_size_1,
                                          stride=self.stride_1, padding=0, output_padding=0, groups=1, bias=True,
                                          dilation=1, padding_mode='zeros', **factory_kwargs)
        self.lr_w = Parameter(torch.nn.init.normal_(torch.FloatTensor(
            max_seq_length), std=(1/max_seq_length)).to(device))
        self.lr_b = Parameter(torch.nn.init.constant_(
            torch.Tensor(1), 0).to(device))

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        lr_out = torch.sum(self.lr_w * x, -1).squeeze() + self.lr_b
        assert (lr_out.shape[0] == x.shape[0] and len(lr_out.shape) == 1)
        x = self.conv1d_1(x)
        x = self.leaky_relu_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_2(x)
        x = self.leaky_relu_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_3(x)
        x = self.leaky_relu_3(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_3(x)
        x = self.transformer_encoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.deconv1d_1(x)
        # Add result of linear regression (broadcasting, lr_out should be a scalar)
        x = x + lr_out.reshape(x.shape[0], 1, 1)
        return x


class Orpheus3Classifier(Module):
    r"""Classifier built as a CNN encoder, 
    a transformer encoder and FFN classifier.

    Improvements in encoder w.r.t. Orpheus2:
    - Std transformer encoder
    - Include pooling
    - Downsample more uniformly
    - Downsample rates are not customizable and they are smaller
    """
    def get_conv_output_length(self, kernel_size: float, stride: float, prev_length: float, 
                            dilation: float = 1, padding: float = 0) -> float:
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # and https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        return math.floor((prev_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus3Classifier, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.stride_1 = 2
        self.kernel_size_1 = 3
        length_1 = self.get_conv_output_length(self.kernel_size_1, self.stride_1, max_seq_length)
        # After conv1, 2687 frames
        self.stride_pool_1 = 2
        self.kernel_size_pool_1 = 2
        length_2 = self.get_conv_output_length(self.kernel_size_pool_1, self.stride_pool_1, length_1)
        # After pool1, 1343 frames
        self.stride_2 = 2
        self.kernel_size_2 = 3
        length_3 = self.get_conv_output_length(self.kernel_size_2, self.stride_2, length_2)
        # After conv2, 671 frames
        self.stride_pool_2 = 2
        self.kernel_size_pool_2 = 2
        length_4 = self.get_conv_output_length(self.kernel_size_pool_2, self.stride_pool_2, length_3)
        # After pool2, 335 frames
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=(d_model // 2), kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_1 = LeakyReLU()
        self.maxpool1d_1 = MaxPool1d(kernel_size=self.kernel_size_pool_1, stride=self.stride_pool_1, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.dropout_1 = Dropout(0.1)
        self.norm_conv1d_1 = LayerNorm(
            d_model // 2, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=(d_model // 2), out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_2 = LeakyReLU()
        self.maxpool1d_2 = MaxPool1d(kernel_size=self.kernel_size_pool_2, stride=self.stride_pool_2, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.dropout_2 = Dropout(0.1)
        self.norm_conv1d_2 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout, activation=activation,
                                                            layer_norm_eps=layer_norm_eps,
                                                            batch_first=batch_first,
                                                            norm_first=norm_first,
                                                            **factory_kwargs)
        norm_out_trafo = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=max_seq_length)
        self.transformer_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                      num_layers=n_layers, norm=norm_out_trafo,
                                                      enable_nested_tensor=True)
        self.dropout_3 = Dropout(0.1)

        # Average the frames with one linear regression per feature
        # Output in the form Nbatches x Nframes x Nfeatures
        # Matrix 1 x Nframes x Nfeatures
        self.w_avg = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((1, length_4, d_model))).to(device), 
                                requires_grad=True)
        self.linear_1 = Linear(in_features=d_model, out_features=1, **factory_kwargs)
        self.activation_out = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = self.conv1d_1(x)
        x = self.leaky_relu_1(x)
        x = self.maxpool1d_1(x)
        x = self.dropout_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_2(x)
        x = self.leaky_relu_2(x)
        x = self.maxpool1d_2(x)
        x = self.dropout_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_2(x)
        x = self.positional_embedding(x)
        x = self.transformer_encoder(x)
        # Sum over frames after weighting their values per feature (lr per feature). 
        # All batches the same by broadcasting
        # Tensor of Nbatches x Nfeatures returned
        x = self.dropout_3(x)
        x = torch.sum(self.w_avg * x, 1).squeeze()
        # Final linear layer for classification
        x = self.activation_out(self.linear_1(x)).squeeze()
        return x

class Orpheus4Classifier(Module):
    r"""Transformer decoder built as a CNN encoder, 
    a transformer encoder and FFN classifier.

    Improvements in encoder w.r.t. Orpheus3Classifier:
    - Final average of frames is not learned to avoid collapsing
    - Normalize input
    """
    def get_conv_output_length(self, kernel_size: float, stride: float, prev_length: float, 
                            dilation: float = 1, padding: float = 0) -> float:
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # and https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        return math.floor((prev_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


    def __init__(self, d_model: int, nhead: int, max_seq_length: int, in_channels: int,
                 n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Orpheus4Classifier, self).__init__()

        # No mask is needed as only the past is used
        # as input to the model during training
        # To make Lout = Lin / stride, no padding, dilation=1 and kernel_size = stride
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.norm_input = LayerNorm(
            max_seq_length, eps=layer_norm_eps, **factory_kwargs)
        self.stride_1 = 2
        self.kernel_size_1 = 3
        length_1 = self.get_conv_output_length(self.kernel_size_1, self.stride_1, max_seq_length)
        # After conv1, 2687 frames
        self.stride_pool_1 = 2
        self.kernel_size_pool_1 = 2
        length_2 = self.get_conv_output_length(self.kernel_size_pool_1, self.stride_pool_1, length_1)
        # After pool1, 1343 frames
        self.stride_2 = 2
        self.kernel_size_2 = 3
        length_3 = self.get_conv_output_length(self.kernel_size_2, self.stride_2, length_2)
        # After conv2, 671 frames
        self.stride_pool_2 = 2
        self.kernel_size_pool_2 = 2
        length_4 = self.get_conv_output_length(self.kernel_size_pool_2, self.stride_pool_2, length_3)
        # After pool2, 335 frames
        self.padding_1, self.padding_2, self.padding_3 = 0, 0, 0
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=(d_model // 2), kernel_size=self.kernel_size_1,
                               stride=self.stride_1, padding=self.padding_1, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_1 = LeakyReLU()
        self.maxpool1d_1 = MaxPool1d(kernel_size=self.kernel_size_pool_1, stride=self.stride_pool_1, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.dropout_1 = Dropout(0.1)
        self.norm_conv1d_1 = LayerNorm(
            d_model // 2, eps=layer_norm_eps, **factory_kwargs)
        self.conv1d_2 = Conv1d(in_channels=(d_model // 2), out_channels=d_model, kernel_size=self.kernel_size_2,
                               stride=self.stride_2, padding=self.padding_2, padding_mode='reflect', **factory_kwargs)
        self.leaky_relu_2 = LeakyReLU()
        self.maxpool1d_2 = MaxPool1d(kernel_size=self.kernel_size_pool_2, stride=self.stride_pool_2, padding=0,
                                     dilation=1, return_indices=False, ceil_mode=False)
        self.dropout_2 = Dropout(0.1)
        self.norm_conv1d_2 = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout, activation=activation,
                                                            layer_norm_eps=layer_norm_eps,
                                                            batch_first=batch_first,
                                                            norm_first=norm_first,
                                                            **factory_kwargs)
        norm_out_trafo = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=max_seq_length)
        self.transformer_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                      num_layers=n_layers, norm=norm_out_trafo,
                                                      enable_nested_tensor=True)
        self.dropout_3 = Dropout(0.1)
        self.linear_1 = Linear(in_features=d_model, out_features=1, **factory_kwargs)
        self.activation_out = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            x: the input to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = self.norm_input(x)
        x = self.conv1d_1(x)
        x = self.leaky_relu_1(x)
        x = self.maxpool1d_1(x)
        x = self.dropout_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_1(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_2(x)
        x = self.leaky_relu_2(x)
        x = self.maxpool1d_2(x)
        x = self.dropout_2(x)
        x = torch.transpose(x, 1, 2)
        x = self.norm_conv1d_2(x)
        x = self.positional_embedding(x)
        x = self.transformer_encoder(x)
        # Sum over frames after weighting their values per feature (lr per feature). 
        # All batches the same by broadcasting
        # Tensor of Nbatches x Nfeatures returned
        x = self.dropout_3(x)
        x = torch.mean(x, 1).squeeze()
        # Final linear layer for classification
        x = self.activation_out(self.linear_1(x)).squeeze()
        return x