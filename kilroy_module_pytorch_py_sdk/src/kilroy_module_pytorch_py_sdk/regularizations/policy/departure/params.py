from typing import Dict, Any, Optional

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.losses.distribution import (
    JensenShannonMetricDistributionLoss,
)


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class LossParams(SerializableModel):
    type: str = JensenShannonMetricDistributionLoss.category
    params: Dict[str, Dict[str, Any]] = {}


class MetricsParams(SerializableModel):
    direct_loss: Optional[MetricParams] = MetricParams(
        name="departureDirectLoss",
        label="Departure Direct Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    aggregated_loss: Optional[MetricParams] = MetricParams(
        name="departureAggregatedLoss",
        label="Departure Aggregated Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )


class Params(SerializableModel):
    weight: float = 1.0
    loss: LossParams = LossParams()
    metrics: MetricsParams = MetricsParams()
