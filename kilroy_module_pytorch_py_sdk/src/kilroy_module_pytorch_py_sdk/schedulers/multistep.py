from collections import Counter
from typing import Any, Dict, List

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

from kilroy_module_pytorch_py_sdk.schedulers.base import (
    StandardSchedulerBase,
    StandardSchedulerState as State,
    SchedulerParameter,
)


class Params(SerializableModel):
    milestones: List[int] = [1]
    gamma: float = 0.1


class MultiStepScheduler(StandardSchedulerBase):
    class MilestonesParameter(SchedulerParameter[State, List[int]]):
        @classmethod
        async def _get_from_scheduler(
            cls, scheduler: MultiStepLR
        ) -> List[int]:
            return list(sorted(scheduler.milestones.elements()))

        @classmethod
        async def _set_in_scheduler(
            cls, scheduler: MultiStepLR, value: List[int]
        ) -> None:
            scheduler.milestones = Counter(value)

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "title": cls.pretty_name,
                "default": [1],
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
        return MultiStepLR(optimizer, **user_params.dict())
