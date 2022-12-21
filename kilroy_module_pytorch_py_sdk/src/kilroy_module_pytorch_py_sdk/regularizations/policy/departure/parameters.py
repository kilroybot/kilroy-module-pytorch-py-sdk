from typing import Dict, Any, Type

from kilroy_module_pytorch_py_sdk.losses.distribution import (
    DistributionLoss,
    JensenShannonMetricDistributionLoss,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.departure.state import (
    State,
)
from kilroy_server_py_utils import (
    CategorizableBasedParameter,
    Parameter,
    classproperty,
)


class WeightParameter(Parameter[State, float]):
    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "number",
            "minimum": 0,
            "title": cls.pretty_name,
            "default": 1.0,
        }


class LossParameter(CategorizableBasedParameter[State, DistributionLoss]):
    @classmethod
    async def _get_categorizable(cls, state: State) -> DistributionLoss:
        return state.loss.loss

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: DistributionLoss
    ) -> None:
        state.loss.loss = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.loss.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[DistributionLoss]:
        return JensenShannonMetricDistributionLoss
