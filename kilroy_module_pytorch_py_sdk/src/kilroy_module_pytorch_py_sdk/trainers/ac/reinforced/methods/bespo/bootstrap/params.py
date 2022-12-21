from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.stop import (
    IterationsStopCondition,
)


class StopConditionParams(SerializableModel):
    type: str = IterationsStopCondition.category
    params: Dict[str, Dict[str, Any]] = {}


class Params(SerializableModel):
    generator: Dict[str, Any] = {}
    sample_size: int = 32
    stop_condition: StopConditionParams = StopConditionParams()
