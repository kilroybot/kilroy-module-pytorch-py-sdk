from typing import Dict, Any, Sequence

from torch import Tensor

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.stop.base import (
    StopCondition,
)
from kilroy_module_server_py_sdk import SerializableState
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class State(SerializableState):
    iterations: int = 10


class IterationsStopCondition(StopCondition, Configurable[State]):
    class IterationsParameter(Parameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 10,
            }

    async def should_stop(
        self, rewards: Sequence[Tensor], iteration: int
    ) -> bool:
        async with self.state.read_lock() as state:
            iterations = state.iterations

        return iteration >= iterations
