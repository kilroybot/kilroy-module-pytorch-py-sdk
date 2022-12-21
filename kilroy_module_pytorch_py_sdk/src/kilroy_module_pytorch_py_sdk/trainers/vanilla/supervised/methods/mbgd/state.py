from dataclasses import dataclass
from typing import Dict, Any

from kilroy_module_pytorch_py_sdk.losses.policy import PolicyLoss
from kilroy_module_pytorch_py_sdk.metrics import LossMetric


@dataclass
class PolicyMetricsState:
    base_batch_loss: LossMetric
    base_epoch_loss: LossMetric
    combined_batch_loss: LossMetric
    combined_epoch_loss: LossMetric


@dataclass
class PolicyLossState:
    loss: PolicyLoss
    params: Dict[str, Dict[str, Any]]


@dataclass
class PolicyState:
    metrics: PolicyMetricsState
    loss: PolicyLossState


@dataclass
class State:
    policy: PolicyState
    batch_size: int
