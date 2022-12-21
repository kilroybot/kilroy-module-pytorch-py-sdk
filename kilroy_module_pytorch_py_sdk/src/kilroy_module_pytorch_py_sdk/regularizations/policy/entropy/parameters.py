from typing import Dict, Any

from kilroy_module_pytorch_py_sdk.regularizations.policy.entropy.state import (
    State,
)
from kilroy_server_py_utils import Parameter, classproperty


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
