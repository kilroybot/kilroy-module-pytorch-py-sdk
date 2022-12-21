from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.losses.policy import (
    NegativeLogLikelihoodPolicyLoss,
)


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class PolicyMetricsParams(SerializableModel):
    base_batch_loss: MetricParams = MetricParams(
        name="supervisedBaseBatchLoss",
        label="Supervised Base Batch Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    base_epoch_loss: MetricParams = MetricParams(
        name="supervisedBaseEpochLoss",
        label="Supervised Base Epoch Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )
    combined_batch_loss: MetricParams = MetricParams(
        name="supervisedNormalizedBatchLoss",
        label="Supervised Combined Batch Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    combined_epoch_loss: MetricParams = MetricParams(
        name="supervisedNormalizedEpochLoss",
        label="Supervised Combined Epoch Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )


class PolicyLossParams(SerializableModel):
    type: str = NegativeLogLikelihoodPolicyLoss.category
    params: Dict[str, Dict[str, Any]] = {}


class PolicyParams(SerializableModel):
    metrics: PolicyMetricsParams = PolicyMetricsParams()
    loss: PolicyLossParams = PolicyLossParams()


class Params(SerializableModel):
    policy: PolicyParams = PolicyParams()
    batch_size: int = 32
