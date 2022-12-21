from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.policy import (
    DeltaPolicyStopCondition,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.value import (
    PlateauValueStopCondition,
)


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class PolicyMetricsParams(SerializableModel):
    episode_score: MetricParams = MetricParams(
        name="policyReinforcedRealScore",
        label="Policy Reinforced Real Score",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    episode_reward: MetricParams = MetricParams(
        name="policyReinforcedRealReward",
        label="Policy Reinforced Real Reward",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    bootstrap_episode_reward: MetricParams = MetricParams(
        name="policyReinforcedBootstrapReward",
        label="Policy Reinforced Bootstrap Reward",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    base_bootstrap_iteration_loss: MetricParams = MetricParams(
        name="policyReinforcedBaseIterationLoss",
        label="Policy Reinforced Base Iteration Loss",
        x_axis_key="iteration",
        x_axis_label="Iteration",
    )
    base_bootstrap_episode_loss: MetricParams = MetricParams(
        name="policyReinforcedBaseEpisodeLoss",
        label="Policy Reinforced Base Episode Loss",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    combined_bootstrap_iteration_loss: MetricParams = MetricParams(
        name="policyReinforcedCombinedIterationLoss",
        label="Policy Reinforced Combined Iteration Loss",
        x_axis_key="iteration",
        x_axis_label="Iteration",
    )
    combined_bootstrap_episode_loss: MetricParams = MetricParams(
        name="policyReinforcedCombinedEpisodeLoss",
        label="Policy Reinforced Combined Episode Loss",
        x_axis_key="episode",
        x_axis_label="Episode",
    )


class PolicyStopConditionParams(SerializableModel):
    type: str = DeltaPolicyStopCondition.category
    params: Dict[str, Dict[str, Any]] = {}


class PolicyParams(SerializableModel):
    metrics: PolicyMetricsParams = PolicyMetricsParams()
    stop_condition: PolicyStopConditionParams = PolicyStopConditionParams()


class ValueMetricsParams(SerializableModel):
    base_iteration_loss: MetricParams = MetricParams(
        name="valueReinforcedBaseIterationLoss",
        label="Value Reinforced Base Iteration Loss",
        x_axis_key="iteration",
        x_axis_label="Iteration",
    )
    base_episode_loss: MetricParams = MetricParams(
        name="valueReinforcedBaseEpisodeLoss",
        label="Value Reinforced Base Episode Loss",
        x_axis_key="episode",
        x_axis_label="Episode",
    )


class ValueStopConditionParams(SerializableModel):
    type: str = PlateauValueStopCondition.category
    params: Dict[str, Dict[str, Any]] = {}


class ValueParams(SerializableModel):
    metrics: ValueMetricsParams = ValueMetricsParams()
    stop_condition: ValueStopConditionParams = ValueStopConditionParams()


class Params(SerializableModel):
    policy: PolicyParams = PolicyParams()
    value: ValueParams = ValueParams()
    bootstrap: Dict[str, Any] = {}
