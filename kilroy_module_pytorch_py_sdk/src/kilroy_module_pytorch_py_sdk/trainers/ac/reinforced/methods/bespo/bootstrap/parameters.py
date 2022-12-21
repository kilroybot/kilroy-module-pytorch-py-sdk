from typing import Dict, Any, Type

from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.state import (
    State,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.stop import (
    StopCondition,
    IterationsStopCondition,
)
from kilroy_server_py_utils import (
    NestedParameter,
    Parameter,
    classproperty,
    CategorizableBasedParameter,
)


class GeneratorParameter(NestedParameter[State, Generator]):
    pass


class SampleSizeParameter(Parameter[State, int]):
    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 1,
            "title": cls.pretty_name,
            "default": 32,
        }


class StopConditionParameter(
    CategorizableBasedParameter[State, StopCondition]
):
    @classmethod
    async def _get_categorizable(cls, state: State) -> StopCondition:
        return state.stop_condition.condition

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: StopCondition
    ) -> None:
        state.stop_condition.condition = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.stop_condition.params.get(category, {})

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[StopCondition]:
        return IterationsStopCondition
