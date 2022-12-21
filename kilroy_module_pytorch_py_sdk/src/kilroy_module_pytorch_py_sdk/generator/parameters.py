import re
from typing import List, Dict, Any, Callable, Awaitable

from kilroy_module_pytorch_py_sdk.generator.state import State
from kilroy_server_py_utils import Parameter, classproperty


class ContextsParameter(Parameter[State, List[str]]):
    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "title": cls.pretty_name,
            "default": [],
        }


class RegexParameter(Parameter[State, str]):
    @classmethod
    async def _get(cls, state: State) -> str:
        return state.regex.pattern

    @classmethod
    async def _set(cls, state: State, value: str) -> Callable[[], Awaitable]:
        original_value = state.regex

        async def undo():
            state.regex = original_value

        state.regex = re.compile(value)
        return undo

    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "string",
            "title": cls.pretty_name,
            "default": r"^(^(?!.*\s+[a-zA-Z0-9_']*$).+$)|(^(?!.*[\.\?!]+).+$)$",
        }

    # noinspection PyMethodParameters
    @classproperty
    def pretty_name(cls) -> str:
        return "Valid Regular Expression"


class MaxLengthParameter(Parameter[State, int]):
    # noinspection PyMethodParameters
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 1,
            "title": cls.pretty_name,
            "default": 16,
        }

    # noinspection PyMethodParameters
    @classproperty
    def pretty_name(cls) -> str:
        return "Maximum Length"
