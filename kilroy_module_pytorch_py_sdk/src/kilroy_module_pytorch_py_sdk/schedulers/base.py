from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Generic, TypeVar

import torch
from humps import decamelize
from kilroy_module_server_py_sdk import (
    Categorizable,
    Configurable,
    Parameter,
    background,
    classproperty,
    normalize,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

StateType = TypeVar("StateType")
ParameterType = TypeVar("ParameterType")
SchedulerType = TypeVar("SchedulerType", bound=_LRScheduler)


class SchedulerParameter(
    Parameter[StateType, ParameterType], ABC, Generic[StateType, ParameterType]
):
    @classproperty
    def attribute_name(cls) -> str:
        return decamelize(cls.name)

    async def _get_from_scheduler(
        self, scheduler: _LRScheduler
    ) -> ParameterType:
        return getattr(scheduler, self.attribute_name)

    async def _get(self, state: StateType) -> ParameterType:
        return await self._get_from_scheduler(state.scheduler)

    async def _set_in_scheduler(
        self, scheduler: _LRScheduler, value: ParameterType
    ) -> None:
        setattr(scheduler, self.attribute_name, value)

    async def _set(
        self, state: StateType, value: ParameterType
    ) -> Callable[[], Awaitable]:
        current_value = self._get(state)

        async def undo() -> None:
            await self._set_in_scheduler(state.scheduler, current_value)

        await self._set_in_scheduler(state.scheduler, value)

        return undo


class Scheduler(Categorizable, ABC):
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
    async def _build_default_state(self) -> StandardSchedulerState:
        optimizer = self._kwargs.pop("optimizer")
        scheduler = await self._build_default_scheduler(optimizer)
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
        with open(directory / "scheduler.pt", "rb") as f:
            state_dict = await background(torch.load, f)
        optimizer = self._kwargs.pop("optimizer")
        scheduler = await self._build_default_scheduler(optimizer)
        scheduler.load_state_dict(state_dict)
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
