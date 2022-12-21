from typing import Any, Dict

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    StandardSchedulerBase,
    StandardSchedulerState as State,
    SchedulerParameter,
)


class Params(SerializableModel):
    step_size: int = 1
    gamma: float = 0.1


class StepScheduler(StandardSchedulerBase):
    class StepSizeParameter(SchedulerParameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 1,
            }

    class GammaParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.1,
            }

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return StepLR(optimizer, **user_params.dict())
