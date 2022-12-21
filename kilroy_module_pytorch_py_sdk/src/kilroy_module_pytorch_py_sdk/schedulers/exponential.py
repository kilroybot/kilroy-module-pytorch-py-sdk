from typing import Any, Dict

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    StandardSchedulerBase,
    StandardSchedulerState as State,
    SchedulerParameter,
)


class Params(SerializableModel):
    gamma: float = 0.99


class ExponentialScheduler(StandardSchedulerBase):
    class GammaParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.99,
            }

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return ExponentialLR(optimizer, **user_params.dict())
