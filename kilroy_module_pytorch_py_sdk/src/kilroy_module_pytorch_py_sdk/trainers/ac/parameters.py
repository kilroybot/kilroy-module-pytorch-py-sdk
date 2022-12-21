from typing import Optional, Dict, Any, Type

from kilroy_module_pytorch_py_sdk.gae import GeneralizedAdvantageEstimator
from kilroy_module_pytorch_py_sdk.scalers.reward import (
    RewardScaler,
    WindowRewardScaler,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.trainer import (
    ReinforcedTrainer,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.state import State
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.trainer import (
    SupervisedTrainer,
)
from kilroy_server_py_utils import (
    NestedParameter,
    CategorizableBasedOptionalParameter,
    classproperty,
)


class SupervisedParameter(NestedParameter[State, SupervisedTrainer]):
    pass


class ReinforcedParameter(NestedParameter[State, ReinforcedTrainer]):
    pass


class ScalerParameter(
    CategorizableBasedOptionalParameter[State, RewardScaler]
):
    @classmethod
    async def _get_categorizable(cls, state: State) -> Optional[RewardScaler]:
        return state.scaler.scaler

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: Optional[RewardScaler]
    ) -> None:
        state.scaler.scaler = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.scaler.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Optional[Type[RewardScaler]]:
        return WindowRewardScaler


class GeneralizedAdvantageEstimationParameter(
    NestedParameter[State, GeneralizedAdvantageEstimator]
):
    @classmethod
    async def _get_configurable(
        cls, state: State
    ) -> GeneralizedAdvantageEstimator:
        return state.gae
