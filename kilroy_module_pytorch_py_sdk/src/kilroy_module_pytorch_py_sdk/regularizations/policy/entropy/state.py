from dataclasses import dataclass
from typing import Optional

from kilroy_module_pytorch_py_sdk.metrics import LossMetric


@dataclass
class MetricsState:
    direct_loss: Optional[LossMetric]
    aggregated_loss: Optional[LossMetric]


@dataclass
class State:
    weight: float
    metrics: MetricsState
