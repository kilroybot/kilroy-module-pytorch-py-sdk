from typing import Dict, Any, Optional, List

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.optimizers import AdamOptimizer
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods import (
    BootstrappingEarlyStoppingPolicyOptimizationMethod,
)


class OptimizerParams(SerializableModel):
    type: str = AdamOptimizer.category
    params: Dict[str, Dict[str, Any]] = {}


class SchedulerParams(SerializableModel):
    type: Optional[str] = None
    params: Dict[str, Dict[str, Any]] = {}


class ModelParams(SerializableModel):
    optimizer: OptimizerParams = OptimizerParams()
    scheduler: SchedulerParams = SchedulerParams()


class ModelsParams(SerializableModel):
    policy: ModelParams = ModelParams()
    value: ModelParams = ModelParams()


class MethodParams(SerializableModel):
    type: str = BootstrappingEarlyStoppingPolicyOptimizationMethod.category
    params: Dict[str, Dict[str, Any]] = {}


class RegularizationsParams(SerializableModel):
    types: Optional[List[str]] = None
    params: Dict[str, Dict[str, Any]] = {
        "departure": {
            "weight": 10.0,
            "metrics": {
                "directLoss": {
                    "name": "policyReinforcedDepartureIterationLoss",
                    "label": "Policy Reinforced Departure Iteration Loss",
                    "xAxisKey": "iteration",
                    "xAxisLabel": "Iteration",
                },
                "aggregatedLoss": {
                    "name": "policyReinforcedDepartureEpisodeLoss",
                    "label": "Policy Reinforced Departure Episode Loss",
                    "xAxisKey": "episode",
                    "xAxisLabel": "Episode",
                },
            },
        },
        "entropy": {
            "weight": 1.0,
            "metrics": {
                "directLoss": {
                    "name": "policyReinforcedEntropyIterationLoss",
                    "label": "Policy Reinforced Entropy Iteration Loss",
                    "xAxisKey": "iteration",
                    "xAxisLabel": "Iteration",
                },
                "aggregatedLoss": {
                    "name": "policyReinforcedEntropyEpisodeLoss",
                    "label": "Policy Reinforced Entropy Episode Loss",
                    "xAxisKey": "episode",
                    "xAxisLabel": "Episode",
                },
            },
        },
    }


class Params(SerializableModel):
    models: ModelsParams = ModelsParams()
    method: MethodParams = MethodParams()
    regularizations: RegularizationsParams = RegularizationsParams()
    cache: Dict[str, Any] = {}
