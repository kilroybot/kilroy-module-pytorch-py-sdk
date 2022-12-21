from typing import Optional

from kilroy_module_py_shared import SerializableModel


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class MetricsParams(SerializableModel):
    direct_loss: Optional[MetricParams] = MetricParams(
        name="entropyDirectLoss",
        label="Entropy Direct Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    aggregated_loss: Optional[MetricParams] = MetricParams(
        name="entropyAggregatedLoss",
        label="Entropy Aggregated Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )


class Params(SerializableModel):
    weight: float = 1.0
    metrics: MetricsParams = MetricsParams()
