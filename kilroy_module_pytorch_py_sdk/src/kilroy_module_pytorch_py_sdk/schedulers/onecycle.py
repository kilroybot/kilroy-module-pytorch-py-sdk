from typing import Any, Dict, Literal

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    SchedulerParameter,
    StandardSchedulerBase,
    StandardSchedulerState as State,
)


class Params(SerializableModel):
    max_lr: float = 0.1
    total_steps: int = 100
    pct_start: float = 0.3
    anneal_strategy: Literal["cos", "linear"] = "cos"
    div_factor: float = 25.0
    final_div_factor: float = 1e4


class OneCycleScheduler(StandardSchedulerBase):
    class MaxLRParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: OneCycleLR) -> float:
            return scheduler.optimizer.param_groups[0]["max_lr"]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: float
        ) -> None:
            for group in scheduler.optimizer.param_groups:
                old_initial_lr = group["initial_lr"]
                old_max_lr = group["max_lr"]
                old_min_lr = group["min_lr"]
                old_div_factor = old_max_lr / old_initial_lr
                old_final_div_factor = old_initial_lr / old_min_lr

                group["initial_lr"] = value / old_div_factor
                group["max_lr"] = value
                group["min_lr"] = group["initial_lr"] / old_final_div_factor

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.1,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Maximum Learning Rate"

    class TotalStepsParameter(SchedulerParameter[State, int]):
        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: int
        ) -> None:
            old_total_steps = scheduler.total_steps
            # noinspection PyUnresolvedReferences,PyProtectedMember
            phases = scheduler._schedule_phases
            pct_start = (phases[0]["end_step"] + 1) / old_total_steps

            scheduler.total_steps = value
            phases[0]["end_step"] = float(pct_start * value) - 1
            phases[1]["end_step"] = value - 1

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 100,
            }

    class PctStartParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: OneCycleLR) -> float:
            # noinspection PyUnresolvedReferences,PyProtectedMember
            phases = scheduler._schedule_phases
            steps = scheduler.total_steps
            return (phases[0]["end_step"] + 1) / steps

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: float
        ) -> None:
            # noinspection PyUnresolvedReferences,PyProtectedMember
            phases = scheduler._schedule_phases
            steps = scheduler.total_steps
            phases[0]["end_step"] = float(value * steps) - 1
            phases[1]["end_step"] = steps - 1

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "title": cls.pretty_name,
                "default": 0.3,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Percentage of Steps for Increasing Learning Rate"

    class AnnealStrategyParameter(
        SchedulerParameter[State, Literal["cos", "linear"]]
    ):
        @classmethod
        async def _get_from_scheduler(
            cls, scheduler: OneCycleLR
        ) -> Literal["cos", "linear"]:
            # noinspection PyProtectedMember,PyUnresolvedReferences
            if scheduler.anneal_func is scheduler._annealing_cos:
                return "cos"
            return "linear"

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: Literal["cos", "linear"]
        ) -> None:
            if value == "cos":
                # noinspection PyProtectedMember,PyUnresolvedReferences
                scheduler.anneal_func = scheduler._annealing_cos
            else:
                # noinspection PyProtectedMember,PyUnresolvedReferences
                scheduler.anneal_func = scheduler._annealing_linear

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "string",
                "oneOf": [
                    {"const": "cos", "title": "Cosine"},
                    {"const": "linear", "title": "Linear"},
                ],
                "title": cls.pretty_name,
                "default": "cos",
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Annealing Strategy"

    class DivFactorParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: OneCycleLR) -> float:
            group = scheduler.optimizer.param_groups[0]
            return group["max_lr"] / group["initial_lr"]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: float
        ) -> None:
            for group in scheduler.optimizer.param_groups:
                old_initial_lr = group["initial_lr"]
                old_max_lr = group["max_lr"]
                old_min_lr = group["min_lr"]
                old_final_div_factor = old_initial_lr / old_min_lr

                group["initial_lr"] = old_max_lr / value
                group["min_lr"] = old_initial_lr / old_final_div_factor

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 25.0,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Divisor for Initial Learning Rate"

    class FinalDivFactorParameter(SchedulerParameter[State, float]):
        @classmethod
        async def _get_from_scheduler(cls, scheduler: OneCycleLR) -> float:
            group = scheduler.optimizer.param_groups[0]
            return group["initial_lr"] / group["min_lr"]

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: OneCycleLR, value: float
        ) -> None:
            for group in scheduler.optimizer.param_groups:
                group["min_lr"] = group["initial_lr"] / value

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1e4,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Divisor for Final Learning Rate"

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return OneCycleLR(
            optimizer, cycle_momentum=False, **user_params.dict()
        )
