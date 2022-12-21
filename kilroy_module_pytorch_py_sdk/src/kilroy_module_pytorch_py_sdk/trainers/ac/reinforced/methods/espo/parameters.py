from typing import Dict, Any, Type

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.espo.state import (
    State,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.espo.stop.policy import (
    PolicyStopCondition,
    DeltaPolicyStopCondition,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.espo.stop.value import (
    ValueStopCondition,
    PlateauValueStopCondition,
)
from kilroy_server_py_utils import CategorizableBasedParameter, classproperty


class PolicyStopConditionParameter(
    CategorizableBasedParameter[State, PolicyStopCondition]
):
    @classmethod
    async def _get_categorizable(cls, state: State) -> PolicyStopCondition:
        return state.policy.stop_condition.condition

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: PolicyStopCondition
    ) -> None:
        state.policy.stop_condition.condition = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.policy.stop_condition.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[PolicyStopCondition]:
        return DeltaPolicyStopCondition


class ValueStopConditionParameter(
    CategorizableBasedParameter[State, ValueStopCondition]
):
    @classmethod
    async def _get_categorizable(cls, state: State) -> ValueStopCondition:
        return state.value.stop_condition.condition

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: ValueStopCondition
    ) -> None:
        state.value.stop_condition.condition = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.value.stop_condition.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[ValueStopCondition]:
        return PlateauValueStopCondition
