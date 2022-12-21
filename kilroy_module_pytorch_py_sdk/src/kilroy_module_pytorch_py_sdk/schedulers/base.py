from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Generic, TypeVar

import torch
from humps import decamelize
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from kilroy_module_server_py_sdk import (
    Categorizable,
    Configurable,
    Parameter,
    background,
    classproperty,
    normalize,
)
from kilroy_server_py_utils import Configuration

StateType = TypeVar("StateType")
ParameterType = TypeVar("ParameterType")
SchedulerType = TypeVar("SchedulerType", bound=_LRScheduler)


class SchedulerParameter(
    Parameter[StateType, ParameterType], ABC, Generic[StateType, ParameterType]
):
    # noinspection PyMethodParameters
    @classproperty
    def attribute_name(cls) -> str:
        return decamelize(cls.name)

    @classmethod
    async def _get_from_scheduler(
        cls, scheduler: _LRScheduler
    ) -> ParameterType:
        return getattr(scheduler, cls.attribute_name)

    @classmethod
    async def _get(cls, state: StateType) -> ParameterType:
        return await cls._get_from_scheduler(state.scheduler)

    @classmethod
    async def _set_in_scheduler(
        cls, scheduler: _LRScheduler, value: ParameterType
    ) -> None:
        setattr(scheduler, cls.attribute_name, value)

    @classmethod
    async def _set(
        cls, state: StateType, value: ParameterType
    ) -> Callable[[], Awaitable]:
        current_value = cls._get(state)

        async def undo() -> None:
            await cls._set_in_scheduler(state.scheduler, current_value)

        await cls._set_in_scheduler(state.scheduler, value)

        return undo


class Scheduler(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Scheduler"))

    @abstractmethod
    async def change_optimizer(self, optimizer: Optimizer) -> None:
        pass

    @abstractmethod
    async def step(self) -> None:
        pass


@dataclass
class StandardSchedulerState:
    scheduler: _LRScheduler


class StandardSchedulerBase(
    Scheduler, Configurable[StandardSchedulerState], ABC
):
    def __init__(
        self,
        config: Configuration[StandardSchedulerState],
        optimizer: Optimizer,
        **kwargs,
    ) -> None:
        self._optimizer = optimizer
        super().__init__(config, **kwargs)

    async def _build_default_state(self) -> StandardSchedulerState:
        scheduler = await self._build_default_scheduler(self._optimizer)
        return StandardSchedulerState(scheduler=scheduler)

    @abstractmethod
    async def _build_default_scheduler(
        self, optimizer: Optimizer
    ) -> _LRScheduler:
        pass

    @classmethod
    async def _save_state(
        cls, state: StandardSchedulerState, directory: Path
    ) -> None:
        with open(directory / "scheduler.pt", "wb") as f:
            await background(torch.save, state.scheduler.state_dict(), f)

    async def _load_saved_state(
        self, directory: Path
    ) -> StandardSchedulerState:
        scheduler = await self._build_default_scheduler(self._optimizer)
        try:
            with open(directory / "scheduler.pt", "rb") as f:
                state_dict = await background(torch.load, f)
            scheduler.load_state_dict(state_dict)
        except OSError:
            pass
        return StandardSchedulerState(scheduler=scheduler)

    async def _change_optimizer(
        self, scheduler: _LRScheduler, optimizer: Optimizer
    ) -> _LRScheduler:
        state_dict = scheduler.state_dict()
        scheduler = await self._build_default_scheduler(optimizer)
        scheduler.load_state_dict(state_dict)
        return scheduler

    async def change_optimizer(self, optimizer: Optimizer) -> None:
        async with self.state.write_lock() as state:
            state.scheduler = await self._change_optimizer(
                state.scheduler, optimizer
            )

    async def step(self) -> None:
        async with self.state.write_lock() as state:
            await background(state.scheduler.step)
