from typing import Dict, Any

from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.policy.base import (
    PolicyStopCondition,
)
from kilroy_module_server_py_sdk import SerializableState
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class State(SerializableState):
    delta: float = 0.25
    max_iterations: int = 25


class DeltaPolicyStopCondition(PolicyStopCondition, Configurable[State]):
    class DeltaParameter(Parameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.25,
            }

    class MaxIterationsParameter(Parameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 25,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Maximum Iterations"

    async def should_stop(
        self,
        old_logprobs: PackedSequence,
        old_gathered_logprobs: PackedSequence,
        new_logprobs: PackedSequence,
        new_gathered_logprobs: PackedSequence,
        iteration: int,
    ) -> bool:
        async with self.state.read_lock() as state:
            max_delta = state.delta
            max_iterations = state.max_iterations

        if iteration >= max_iterations:
            return True

        ratios = new_gathered_logprobs.data - old_gathered_logprobs.data
        diffs = (ratios.exp() - 1).abs()
        current_delta = diffs.mean().item()

        return current_delta > max_delta
