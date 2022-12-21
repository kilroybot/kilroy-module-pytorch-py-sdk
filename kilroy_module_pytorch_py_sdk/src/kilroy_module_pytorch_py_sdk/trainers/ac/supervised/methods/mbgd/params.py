from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.losses.policy import (
    NegativeLogLikelihoodPolicyLoss,
)
from kilroy_module_pytorch_py_sdk.losses.value import MeanSquaredErrorValueLoss


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class PolicyMetricsParams(SerializableModel):
    base_batch_loss: MetricParams = MetricParams(
        name="policySupervisedBaseBatchLoss",
        label="Policy Supervised Base Batch Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    base_epoch_loss: MetricParams = MetricParams(
        name="policySupervisedBaseEpochLoss",
        label="Policy Supervised Base Epoch Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )
    combined_batch_loss: MetricParams = MetricParams(
        name="policySupervisedCombinedBatchLoss",
        label="Policy Supervised Combined Batch Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    combined_epoch_loss: MetricParams = MetricParams(
        name="policySupervisedCombinedEpochLoss",
        label="Policy Supervised Combined Epoch Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )


class PolicyLossParams(SerializableModel):
    type: str = NegativeLogLikelihoodPolicyLoss.category
    params: Dict[str, Dict[str, Any]] = {}


class PolicyParams(SerializableModel):
    metrics: PolicyMetricsParams = PolicyMetricsParams()
    loss: PolicyLossParams = PolicyLossParams()


class ValueMetricsParams(SerializableModel):
    base_batch_loss: MetricParams = MetricParams(
        name="valueSupervisedBaseBatchLoss",
        label="Value Supervised Base Batch Loss",
        x_axis_key="batch",
        x_axis_label="Batch",
    )
    base_epoch_loss: MetricParams = MetricParams(
        name="valueSupervisedBaseEpochLoss",
        label="Value Supervised Base Epoch Loss",
        x_axis_key="epoch",
        x_axis_label="Epoch",
    )


class ValueLossParams(SerializableModel):
    type: str = MeanSquaredErrorValueLoss.category
    params: Dict[str, Dict[str, Any]] = {}


class ValueParams(SerializableModel):
    metrics: ValueMetricsParams = ValueMetricsParams()
    loss: ValueLossParams = ValueLossParams()


class Params(SerializableModel):
    policy: PolicyParams = PolicyParams()
    value: ValueParams = ValueParams()
    batch_size: int = 32
