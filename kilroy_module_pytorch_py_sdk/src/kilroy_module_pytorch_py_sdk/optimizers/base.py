from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, Iterable, TypeVar

import torch
from humps import decamelize
from torch import Tensor
from torch.optim import Optimizer as TorchOptimizer

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
OptimizerType = TypeVar("OptimizerType", bound=TorchOptimizer)


class OptimizerParameter(
    Parameter[StateType, ParameterType], ABC, Generic[StateType, ParameterType]
):
    @classmethod
    def _get_param(cls, group: Dict[str, Any]) -> ParameterType:
        return group[decamelize(cls.name)]

    @classmethod
    def _set_param(cls, group: Dict[str, Any], value: ParameterType) -> None:
        group[decamelize(cls.name)] = value

    @classmethod
    async def _get(cls, state: StateType) -> ParameterType:
        return cls._get_param(state.optimizer.param_groups[0])

    @classmethod
    async def _set(
        cls, state: StateType, value: ParameterType
    ) -> Callable[[], Awaitable]:
        original_value = cls._get_param(state.optimizer.param_groups[0])

        async def undo() -> None:
            # noinspection PyShadowingNames
            for param_group in state.optimizer.param_groups:
                cls._set_param(param_group, original_value)

        for param_group in state.optimizer.param_groups:
            cls._set_param(param_group, value)

        return undo


class Optimizer(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Optimizer"))

    @abstractmethod
    async def get(self) -> TorchOptimizer:
        pass

    @abstractmethod
    async def step(self) -> None:
        pass


@dataclass
class StandardOptimizerState:
    optimizer: TorchOptimizer


class StandardOptimizer(Optimizer, Configurable[StandardOptimizerState], ABC):
    def __init__(
        self,
        config: Configuration[StandardOptimizerState],
        parameters: Iterable[Tensor],
        **kwargs,
    ) -> None:
        self._model_params = parameters
        super().__init__(config, **kwargs)

    async def _build_default_state(self) -> StandardOptimizerState:
        optimizer = await self._build_default_optimizer(self._model_params)
        return StandardOptimizerState(optimizer=optimizer)

    @abstractmethod
    async def _build_default_optimizer(
        self, parameters: Iterable[Tensor]
    ) -> TorchOptimizer:
        pass

    @classmethod
    async def _save_state(
        cls, state: StandardOptimizerState, directory: Path
    ) -> None:
        with open(directory / "optimizer.pt", "wb") as f:
            await background(torch.save, state.optimizer.state_dict(), f)

    async def _load_saved_state(
        self, directory: Path
    ) -> StandardOptimizerState:
        optimizer = await self._build_default_optimizer(self._model_params)
        try:
            with open(directory / "optimizer.pt", "rb") as f:
                state_dict = await background(torch.load, f)
            optimizer.load_state_dict(state_dict)
        except OSError:
            pass
        return StandardOptimizerState(optimizer=optimizer)

    async def get(self) -> TorchOptimizer:
        async with self.state.read_lock() as state:
            return state.optimizer

    async def step(self) -> None:
        async with self.state.write_lock() as state:

            def step():
                state.optimizer.step()
                state.optimizer.zero_grad()

            await background(step)
