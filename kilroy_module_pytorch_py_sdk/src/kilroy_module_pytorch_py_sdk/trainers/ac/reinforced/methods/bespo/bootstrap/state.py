from dataclasses import dataclass
from typing import Dict, Any

from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.stop import (
    StopCondition,
)


@dataclass
class StopConditionState:
    condition: StopCondition
    params: Dict[str, Dict[str, Any]]


@dataclass
class State:
    generator: Generator
    sample_size: int
    stop_condition: StopConditionState
