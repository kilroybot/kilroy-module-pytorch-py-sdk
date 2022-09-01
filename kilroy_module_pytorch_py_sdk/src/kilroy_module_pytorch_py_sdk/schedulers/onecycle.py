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
        async def _get_from_scheduler(self, scheduler: OneCycleLR) -> float:
            return scheduler.optimizer.param_groups[0]["max_lr"]

        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: float
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

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class TotalStepsParameter(SchedulerParameter[State, int]):
        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: int
        ) -> None:
            old_total_steps = scheduler.total_steps
            phases = scheduler._schedule_phases
            pct_start = (phases[0]["end_step"] + 1) / old_total_steps

            scheduler.total_steps = value
            phases[0]["end_step"] = float(pct_start * value) - 1
            phases[1]["end_step"] = value - 1

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    class PctStartParameter(SchedulerParameter[State, float]):
        async def _get_from_scheduler(self, scheduler: OneCycleLR) -> float:
            phases = scheduler._schedule_phases
            steps = scheduler.total_steps
            return (phases[0]["end_step"] + 1) / steps

        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: float
        ) -> None:
            phases = scheduler._schedule_phases
            steps = scheduler.total_steps
            phases[0]["end_step"] = float(value * steps) - 1
            phases[1]["end_step"] = steps - 1

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    class AnnealStrategyParameter(
        SchedulerParameter[State, Literal["cos", "linear"]]
    ):
        async def _get_from_scheduler(
            self, scheduler: OneCycleLR
        ) -> Literal["cos", "linear"]:
            if scheduler.anneal_func is scheduler._annealing_cos:
                return "cos"
            return "linear"

        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: Literal["cos", "linear"]
        ) -> None:
            if value == "cos":
                scheduler.anneal_func = scheduler._annealing_cos
            else:
                scheduler.anneal_func = scheduler._annealing_linear

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "string", "enum": ["cos", "linear"]}

    class DivFactorParameter(SchedulerParameter[State, float]):
        async def _get_from_scheduler(self, scheduler: OneCycleLR) -> float:
            group = scheduler.optimizer.param_groups[0]
            return group["max_lr"] / group["initial_lr"]

        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: float
        ) -> None:
            for group in scheduler.optimizer.param_groups:
                old_initial_lr = group["initial_lr"]
                old_max_lr = group["max_lr"]
                old_min_lr = group["min_lr"]
                old_final_div_factor = old_initial_lr / old_min_lr

                group["initial_lr"] = old_max_lr / value
                group["min_lr"] = old_initial_lr / old_final_div_factor

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class FinalDivFactorParameter(SchedulerParameter[State, float]):
        async def _get_from_scheduler(self, scheduler: OneCycleLR) -> float:
            group = scheduler.optimizer.param_groups[0]
            return group["initial_lr"] / group["min_lr"]

        async def _set_in_scheduler(
            self, scheduler: OneCycleLR, value: float
        ) -> None:
            for group in scheduler.optimizer.param_groups:
                group["min_lr"] = group["initial_lr"] / value

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        user_params = Params(**self._kwargs)
        return OneCycleLR(
            optimizer, cycle_momentum=False, **user_params.dict()
        )
