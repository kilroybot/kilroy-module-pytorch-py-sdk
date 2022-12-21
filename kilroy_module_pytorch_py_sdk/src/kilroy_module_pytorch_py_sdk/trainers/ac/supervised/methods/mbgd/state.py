from dataclasses import dataclass
from typing import Dict, Any

from kilroy_module_pytorch_py_sdk.losses.policy import PolicyLoss
from kilroy_module_pytorch_py_sdk.losses.value import ValueLoss
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
class ValueMetricsState:
    base_batch_loss: LossMetric
    base_epoch_loss: LossMetric


@dataclass
class ValueLossState:
    loss: ValueLoss
    params: Dict[str, Dict[str, Any]]


@dataclass
class ValueState:
    metrics: ValueMetricsState
    loss: ValueLossState


@dataclass
class State:
    policy: PolicyState
    value: ValueState
    batch_size: int
