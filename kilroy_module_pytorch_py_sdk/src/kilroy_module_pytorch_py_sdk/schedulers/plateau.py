from typing import Any, Dict, Literal

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    SchedulerParameter,
    StandardSchedulerBase,
    StandardSchedulerState as State,
)


class Params(SerializableModel):
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8


class ReduceOnPlateauScheduler(StandardSchedulerBase):
    class ModeParameter(SchedulerParameter[State, Literal["min", "max"]]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "string", "enum": ["min", "max"]}

    class FactorParameter(SchedulerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class PatienceParameter(SchedulerParameter[State, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    class ThresholdParameter(SchedulerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class ThresholdModeParameter(
        SchedulerParameter[State, Literal["rel", "abs"]]
    ):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "string", "enum": ["rel", "abs"]}

    class CooldownParameter(SchedulerParameter[State, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 0}

    class MinLrParameter(SchedulerParameter[State, float]):
        async def _get_from_scheduler(
            self, scheduler: ReduceLROnPlateau
        ) -> float:
            return scheduler.min_lrs[0]

        async def _set_in_scheduler(
            self, scheduler: ReduceLROnPlateau, value: float
        ) -> None:
            scheduler.min_lrs = [value] * len(scheduler.min_lrs)

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class EpsParameter(SchedulerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return ReduceLROnPlateau(optimizer, **user_params.dict())
