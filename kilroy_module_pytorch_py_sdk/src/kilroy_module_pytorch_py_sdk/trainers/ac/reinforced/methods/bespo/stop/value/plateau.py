from typing import Dict, Any, Sequence

from torch import Tensor

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.value.base import (
    ValueStopCondition,
)
from kilroy_module_server_py_sdk import SerializableState
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class State(SerializableState):
    patience: int = 3
    threshold: float = 0.01
    min_loss: float = 0.5
    max_iterations: int = 100


class PlateauValueStopCondition(ValueStopCondition, Configurable[State]):
    class PatienceParameter(Parameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 3,
            }

    class ThresholdParameter(Parameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.05,
            }

    class MinLossParameter(Parameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.5,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Minimum Loss"

    class MaxIterationsParameter(Parameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 100,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Maximum Iterations"

    async def should_stop(
        self, losses: Sequence[Tensor], iteration: int
    ) -> bool:
        async with self.state.read_lock() as state:
            patience = state.patience
            threshold = state.threshold
            min_loss = state.min_loss
            max_iterations = state.max_iterations

        if iteration >= max_iterations:
            return True

        if len(losses) < patience + 1:
            return False

        if losses[-1] > min_loss:
            return False

        last = losses[-(patience + 1) : -1]
        return max(last) - min(last) < threshold
