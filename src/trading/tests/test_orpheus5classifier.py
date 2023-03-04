import pytest

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from trading.models.orpheus.orpheus5classifier import Orpheus5Classifier
from trading.models.LightningClassifier import LightningClassifier

def _create_batch(input_size: list, batch_size: int = 2):
    return torch.randn([batch_size] + input_size)

def _create_target(batch_size: int = 2):
    return torch.ones([batch_size])

def test_orpheus5classifier():
    cfg = OmegaConf.load('trading/training/conf/orpheus5classifier.yaml')
    cfg.data.dataset.input_csv = "../Data/tsla_downloaded_all.csv"
    model = Orpheus5Classifier(cfg=cfg.model)

    input_size = [6, cfg.data.dataset.num_values_in]

    batch = _create_batch(input_size=input_size)

    # Check requires_grad
    for name, param in model.named_parameters():
        assert param.requires_grad
    
    # Check output is a prob
    pred = model(batch)
    assert torch.all(pred >= 0) and torch.all(pred <= 1)

    # Check backpropagation
    target = _create_target()
    loss = torch.nn.functional.binary_cross_entropy(input=pred, target=target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Param {name} has None gradient"
        assert not torch.all(param.grad == 0), f"Param {name} has 0 as gradient"


