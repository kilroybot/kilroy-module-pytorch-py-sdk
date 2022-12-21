from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Awaitable

from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo

ModelType = TypeVar("ModelType")


@dataclass
class TrainingControls(Generic[ModelType]):
    model: ModelInfo[ModelType]
    step: Callable[[], Awaitable[int]]
    increment_epoch: Callable[[], Awaitable[int]]
