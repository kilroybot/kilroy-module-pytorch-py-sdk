from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelLoader
from kilroy_module_pytorch_py_sdk.optimizers.base import Optimizer
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.schedulers.base import Scheduler
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.cache.cache import (
    Cache,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods import (
    Method,
)


@dataclass
class OptimizerState:
    optimizer: Optimizer
    params: Dict[str, Dict[str, Any]]


@dataclass
class SchedulerState:
    scheduler: Optional[Scheduler]
    params: Dict[str, Dict[str, Any]]


@dataclass
class ModelState:
    loader: ModelLoader[SequentialModel]
    optimizer: OptimizerState
    scheduler: SchedulerState
    step: int
    episode: int


@dataclass
class ModelsState:
    policy: ModelState
    value: ModelState
    baseline: ModelLoader[SequentialModel]


@dataclass
class MethodState:
    method: Method
    params: Dict[str, Dict[str, Any]]


@dataclass
class RegularizationsState:
    regularizations: Optional[List[PolicyRegularization]]
    params: Dict[str, Dict[str, Any]]


@dataclass
class State:
    models: ModelsState
    method: MethodState
    regularizations: RegularizationsState
    cache: Cache
