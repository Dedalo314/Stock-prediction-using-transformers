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

class PositionalEncoding(Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class CustomTransformerDecoderLayerDEPRECATED(Module):
    r"""DEPRECATED: MutiheadAttention removed, not used. Kept for compatibility.

    Modified version of the TransformerDecoderLayer by PyTorch, for time-series prediction.
    The encoder-decoder attention has been removed and only self-attention and FFN layers are used.
    The idea is to use just the decoder to predict the time-series values from the past values, 
    masking the future values.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerDecoderLayerDEPRECATED, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask,
                                   tgt_key_padding_mask)
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        attn_output, att_output_weights = self.self_attn(x, x, x,
                                                         attn_mask=attn_mask,
                                                         key_padding_mask=key_padding_mask)
        logging.debug(
            f"Attention output weights (first 3 rows): {att_output_weights[:3]}")
        x = attn_output[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_clones(module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

    def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(
            "activation should be relu/gelu, not {}".format(activation))


class CustomTransformerEncoderLayer(Module):
    r"""Modified version of the TransformerDecoderLayer by PyTorch, for time-series prediction.
    The encoder-decoder attention has been removed and only self-attention and FFN layers are used.
    The idea is to use just the decoder to predict the time-series values from the past values, 
    masking the future values.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask,
                                   tgt_key_padding_mask)
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        attn_output, att_output_weights = self.self_attn(x, x, x,
                                                         attn_mask=attn_mask,
                                                         key_padding_mask=key_padding_mask)
        logging.debug(
            f"Attention output weights (first 3 rows): {att_output_weights[:3]}")
        x = attn_output[0]
        return self.dropout1(x)

    # multihead attention block
    # def _mha_block(self, x: Tensor, mem: Tensor,
    #                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    #     x = self.multihead_attn(x, mem, mem,
    #                             attn_mask=attn_mask,
    #                             key_padding_mask=key_padding_mask,
    #                             need_weights=False)[0]
    #     return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_clones(module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

    def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(
            "activation should be relu/gelu, not {}".format(activation))


class CustomTransformerDecoderDEPRECATED(Module):
    r"""Transformer decoder built as a combination of multiple
    CustomTransformerDecoderLayer.
    """

    def __init__(self, d_model: int, nhead: int, seq_length: int, n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerDecoderDEPRECATED, self).__init__()

        # Create mask based on seq_length
        tgt_mask = torch.tril(torch.ones(
            (seq_length, seq_length))).type(dtype).to(device)
        tgt_mask[tgt_mask == 0] = float('-inf')
        tgt_mask[tgt_mask == 1] = 0
        self.tgt_mask = tgt_mask
        self.norm_input = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=seq_length)
        # output = regressor(torch.rand(seq_length, d_model), tgt_mask=tgt_mask)

        self.decoder_layers = torch.nn.ModuleList([CustomTransformerDecoderLayerDEPRECATED(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first,
            device=device, dtype=dtype, nhead=nhead, batch_first=batch_first) for i in range(n_layers)])

    def forward(self, tgt: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            tgt: the sequence to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = self.norm_input(tgt)
        x = self.positional_embedding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, tgt_mask=self.tgt_mask)
        return x


class CustomTransformerEncoder(Module):
    r"""Transformer decoder built as a combination of multiple
    CustomTransformerDecoderLayer.
    """

    def __init__(self, d_model: int, nhead: int, seq_length: int, n_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoder, self).__init__()

        # Create mask based on seq_length
        tgt_mask = torch.tril(torch.ones(
            (seq_length, seq_length))).type(dtype).to(device)
        tgt_mask[tgt_mask == 0] = float('-inf')
        tgt_mask[tgt_mask == 1] = 0
        self.tgt_mask = tgt_mask
        self.norm_input = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.positional_embedding = PositionalEncoding(
            num_hiddens=d_model, dropout=dropout, max_len=seq_length)
        # output = regressor(torch.rand(seq_length, d_model), tgt_mask=tgt_mask)

        self.decoder_layers = torch.nn.ModuleList([CustomTransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first,
            device=device, dtype=dtype, nhead=nhead, batch_first=batch_first) for i in range(n_layers)])

    def forward(self, tgt: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.

        Args:
            tgt: the sequence to the decoder (required).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = self.norm_input(tgt)
        x = self.positional_embedding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, tgt_mask=self.tgt_mask)
        return x