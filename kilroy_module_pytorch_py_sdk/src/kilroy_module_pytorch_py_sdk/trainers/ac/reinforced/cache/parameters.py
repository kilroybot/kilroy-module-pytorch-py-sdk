from typing import Dict, Any, Callable, Awaitable

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.cache.state import (
    State,
)
from kilroy_server_py_utils import Parameter, classproperty


class MaxSizeParameter(Parameter[State, int]):
    @classmethod
    async def _set(cls, state: State, value: int) -> Callable[[], Awaitable]:
        original_value = state.max_size
        original_data = state.data

        async def undo():
            state.max_size = original_value
            state.data = original_data

        state.max_size = value
        state.data = state.data[-value:]
        return undo

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
        return "Maximum Size"
