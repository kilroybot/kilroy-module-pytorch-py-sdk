from kilroy_module_py_shared import SerializableModel


class MetricParams(SerializableModel):
    name: str
    label: str
    x_axis_key: str
    x_axis_label: str


class PolicyMetricsParams(SerializableModel):
    episode_score: MetricParams = MetricParams(
        name="reinforcedEpisodeScore",
        label="Reinforced Episode Score",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    episode_reward: MetricParams = MetricParams(
        name="reinforcedEpisodeReward",
        label="Reinforced Episode Reward",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    base_episode_loss: MetricParams = MetricParams(
        name="reinforcedBaseEpisodeLoss",
        label="Reinforced Base Episode Loss",
        x_axis_key="episode",
        x_axis_label="Episode",
    )
    combined_episode_loss: MetricParams = MetricParams(
        name="reinforcedCombinedEpisodeLoss",
        label="Reinforced Combined Episode Loss",
        x_axis_key="episode",
        x_axis_label="Episode",
    )


class PolicyParams(SerializableModel):
    metrics: PolicyMetricsParams = PolicyMetricsParams()


class Params(SerializableModel):
    policy: PolicyParams = PolicyParams()
