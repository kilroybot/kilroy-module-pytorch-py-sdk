from typing import Any, Dict

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from pydantic import Field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    SchedulerParameter,
    StandardSchedulerBase,
    StandardSchedulerState as State,
)


class Params(SerializableModel):
    T_0: int = Field(1, alias="t_zero")
    T_mult: float = Field(1, alias="t_mult")
    eta_min: float = 0


class WarmRestartsScheduler(StandardSchedulerBase):
    class T0Parameter(SchedulerParameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def attribute_name(cls) -> str:
            return "T_0"

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 1,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Number of Iterations for First Restart"

    class TMultParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def attribute_name(cls) -> str:
            return "T_mult"

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Multiplier for Number of Iterations Between Restarts"

    class EtaMinParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Minimum Learning Rate"

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return CosineAnnealingWarmRestarts(optimizer, **user_params.dict())
