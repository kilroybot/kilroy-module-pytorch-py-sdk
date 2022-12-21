from typing import Any, Dict

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    StandardSchedulerBase,
    StandardSchedulerState as State,
    SchedulerParameter,
)


class Params(SerializableModel):
    factor: float = 1 / 3
    total_iters: int = 5


class ConstantScheduler(StandardSchedulerBase):
    class FactorParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1 / 3,
            }

    class TotalItersParameter(SchedulerParameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 5,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Total Iterations"

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return ConstantLR(optimizer, **user_params.dict())
