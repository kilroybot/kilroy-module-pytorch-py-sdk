from dataclasses import dataclass
from typing import Optional, Dict, Any

from kilroy_module_pytorch_py_sdk.gae import GeneralizedAdvantageEstimator
from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_pytorch_py_sdk.scalers.reward import RewardScaler
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.trainer import (
    ReinforcedTrainer,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.trainer import (
    SupervisedTrainer,
)


@dataclass
class ScalerState:
    scaler: Optional[RewardScaler]
    params: Dict[str, Dict[str, Any]]


@dataclass
class State:
    models: ModelsRegistry
    supervised: SupervisedTrainer
    reinforced: ReinforcedTrainer
    scaler: ScalerState
    gae: GeneralizedAdvantageEstimator
