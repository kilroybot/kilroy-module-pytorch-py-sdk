from typing import Dict, Any, Type

from kilroy_module_pytorch_py_sdk.losses.policy import (
    PolicyLoss,
    NegativeLogLikelihoodPolicyLoss,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.supervised.methods.mbgd.state import (
    State,
)
from kilroy_server_py_utils import (
    CategorizableBasedParameter,
    Parameter,
    classproperty,
)


class PolicyLossParameter(CategorizableBasedParameter[State, PolicyLoss]):
    @classmethod
    async def _get_categorizable(cls, state: State) -> PolicyLoss:
        return state.policy.loss.loss

    @classmethod
    async def _set_categorizable(cls, state: State, value: PolicyLoss) -> None:
        state.policy.loss.loss = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.policy.loss.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[PolicyLoss]:
        return NegativeLogLikelihoodPolicyLoss


class BatchSizeParameter(Parameter[State, int]):
    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 1,
            "title": cls.pretty_name,
            "default": 32,
        }
