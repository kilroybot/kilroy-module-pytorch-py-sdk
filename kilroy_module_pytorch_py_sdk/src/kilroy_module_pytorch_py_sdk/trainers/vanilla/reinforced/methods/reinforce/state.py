from dataclasses import dataclass

from kilroy_module_pytorch_py_sdk.metrics import ScoreMetric, LossMetric


@dataclass
class PolicyMetricsState:
    episode_score: ScoreMetric
    episode_reward: ScoreMetric
    base_episode_loss: LossMetric
    combined_episode_loss: LossMetric


@dataclass
class PolicyState:
    metrics: PolicyMetricsState


@dataclass
class State:
    policy: PolicyState
