from dataclasses import dataclass
from typing import Dict, Any

from kilroy_module_pytorch_py_sdk.metrics import ScoreMetric, LossMetric
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.bootstrap import (
    Bootstrap,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.policy import (
    PolicyStopCondition,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.value import (
    ValueStopCondition,
)


@dataclass
class PolicyMetricsState:
    episode_score: ScoreMetric
    episode_reward: ScoreMetric
    bootstrap_episode_reward: ScoreMetric
    base_bootstrap_iteration_loss: LossMetric
    base_bootstrap_episode_loss: LossMetric
    combined_bootstrap_iteration_loss: LossMetric
    combined_bootstrap_episode_loss: LossMetric


@dataclass
class PolicyStopConditionState:
    condition: PolicyStopCondition
    params: Dict[str, Dict[str, Any]]


@dataclass
class PolicyState:
    metrics: PolicyMetricsState
    stop_condition: PolicyStopConditionState


@dataclass
class ValueMetricsState:
    base_iteration_loss: LossMetric
    base_episode_loss: LossMetric


@dataclass
class ValueStopConditionState:
    condition: ValueStopCondition
    params: Dict[str, Dict[str, Any]]


@dataclass
class ValueState:
    metrics: ValueMetricsState
    stop_condition: ValueStopConditionState


@dataclass
class State:
    policy: PolicyState
    value: ValueState
    bootstrap: Bootstrap
