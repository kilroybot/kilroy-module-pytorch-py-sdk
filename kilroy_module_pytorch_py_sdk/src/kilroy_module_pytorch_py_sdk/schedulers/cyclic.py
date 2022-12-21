from typing import Any, Dict, Literal

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    SchedulerParameter,
    StandardSchedulerBase,
    StandardSchedulerState as State,
)


class Params(SerializableModel):
    base_lr: float = 0.001
    max_lr: float = 0.006
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: Literal["triangular", "triangular2", "exp_range"] = "triangular"
    gamma: float = 1.0


class CyclicScheduler(StandardSchedulerBase):
    class BaseLRParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: CyclicLR) -> float:
            return scheduler.base_lrs[0]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: CyclicLR, value: float
        ) -> None:
            lrs = [value] * len(scheduler.base_lrs)
            scheduler.base_lrs = lrs
            if scheduler.last_epoch == -1:
                for lr, group in zip(lrs, scheduler.optimizer.param_groups):
                    group["lr"] = lr

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.001,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Base Learning Rate"

    class MaxLRParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: CyclicLR) -> float:
            return scheduler.max_lrs[0]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: CyclicLR, value: float
        ) -> None:
            scheduler.max_lrs = [value] * len(scheduler.max_lrs)

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.006,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Maximum Learning Rate"

    class StepSizeUpParameter(SchedulerParameter[State, int]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: CyclicLR) -> int:
            return int(scheduler.step_ratio * scheduler.total_size)

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: CyclicLR, value: int
        ) -> None:
            current = await cls._get_from_scheduler(scheduler)
            scheduler.total_size = scheduler.total_size - current + value
            scheduler.step_ratio = value / scheduler.total_size

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 2000,
            }

    class StepSizeDownParameter(SchedulerParameter[State, int]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: CyclicLR) -> int:
            return int(scheduler.total_size * (1 - scheduler.step_ratio))

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: CyclicLR, value: int
        ) -> None:
            current = await cls._get_from_scheduler(scheduler)
            current_up = scheduler.step_ratio * scheduler.total_size
            scheduler.total_size = scheduler.total_size - current + value
            scheduler.step_ratio = current_up / scheduler.total_size

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 2000,
            }

    class ModeParameter(
        SchedulerParameter[
            State, Literal["triangular", "triangular2", "exp_range"]
        ]
    ):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "string",
                "oneOf": [
                    {"const": "triangular", "title": "Triangular"},
                    {"const": "triangular2", "title": "Triangular 2"},
                    {"const": "exp_range", "title": "Exponential Range"},
                ],
                "title": cls.pretty_name,
                "default": "triangular",
            }

    class GammaParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1.0,
            }

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return CyclicLR(optimizer, cycle_momentum=False, **user_params.dict())
