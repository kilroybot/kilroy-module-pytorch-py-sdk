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
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "string",
                "oneOf": [
                    {"const": "min", "title": "Minimum"},
                    {"const": "max", "title": "Maximum"},
                ],
                "title": cls.pretty_name,
                "default": "min",
            }

    class FactorParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.1,
            }

    class PatienceParameter(SchedulerParameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 10,
            }

    class ThresholdParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1e-4,
            }

    class ThresholdModeParameter(
        SchedulerParameter[State, Literal["rel", "abs"]]
    ):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "string",
                "oneOf": [
                    {"const": "rel", "title": "Relative"},
                    {"const": "abs", "title": "Absolute"},
                ],
                "title": cls.pretty_name,
                "default": "rel",
            }

    class CooldownParameter(SchedulerParameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0,
            }

    class MinLrParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(
            cls, scheduler: ReduceLROnPlateau
        ) -> float:
            return scheduler.min_lrs[0]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: ReduceLROnPlateau, value: float
        ) -> None:
            scheduler.min_lrs = [value] * len(scheduler.min_lrs)

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

    class EpsParameter(SchedulerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1e-8,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Epsilon"

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return ReduceLROnPlateau(optimizer, **user_params.dict())
