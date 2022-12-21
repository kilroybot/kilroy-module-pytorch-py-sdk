from dataclasses import dataclass
from typing import Dict, Any, Optional

from kilroy_module_pytorch_py_sdk.losses.distribution import DistributionLoss
from kilroy_module_pytorch_py_sdk.metrics import LossMetric


@dataclass
class LossState:
    loss: DistributionLoss
    params: Dict[str, Dict[str, Any]]


@dataclass
class MetricsState:
    direct_loss: Optional[LossMetric]
    aggregated_loss: Optional[LossMetric]


@dataclass
class State:
    weight: float
    loss: LossState
    metrics: MetricsState
