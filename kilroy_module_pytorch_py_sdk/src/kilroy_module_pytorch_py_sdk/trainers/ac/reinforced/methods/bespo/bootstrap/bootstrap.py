from abc import ABC
from functools import partial
from pathlib import Path
from typing import Dict, Any, Set, Type, List, Tuple, Sequence

import torch
from torch import Tensor

from kilroy_module_pytorch_py_sdk import SequentialModel
from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.parameters import (
    GeneratorParameter,
    SampleSizeParameter,
    StopConditionParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.params import (
    Params,
    StopConditionParams,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.state import (
    State,
    StopConditionState,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.stop import (
    StopCondition,
)
from kilroy_server_py_utils import Configurable, classproperty, Parameter


class BootstrapBase(Configurable[State], ABC):
    @classmethod
    async def _build_generator(cls, params: Dict[str, Any]) -> Generator:
        return await cls._build_generic(Generator, **params)

    @classmethod
    async def _build_stop_condition(
        cls, params: StopConditionParams
    ) -> StopCondition:
        category = params.type
        return await cls._build_generic(
            StopCondition, category=category, **params.params.get(category, {})
        )

    @classmethod
    async def _build_stop_condition_state(
        cls, params: StopConditionParams
    ) -> StopConditionState:
        return StopConditionState(
            condition=await cls._build_stop_condition(params),
            params=params.params,
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            generator=await self._build_generator(params.generator),
            sample_size=params.sample_size,
            stop_condition=await self._build_stop_condition_state(
                params.stop_condition
            ),
        )

    @classmethod
    async def _save_stop_condition_state(
        cls, state: StopConditionState, directory: Path
    ) -> None:
        if isinstance(state.condition, Configurable):
            await state.condition.save(directory / "condition")

        state_dict = {"type": state.condition.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        if isinstance(state.generator, Configurable):
            await state.generator.save(directory / "generator")

        await cls._save_stop_condition_state(
            state.stop_condition, directory / "stop_condition"
        )

        state_dict = {"sample_size": state.sample_size}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _load_generator(
        cls, params: Dict[str, Any], directory: Path
    ) -> Generator:
        return await cls._load_generic(
            directory,
            Generator,
            **params,
            default=partial(cls._build_generator, params),
        )

    @classmethod
    async def _load_stop_condition(
        cls,
        state_dict: Dict[str, Any],
        params: StopConditionParams,
        directory: Path,
    ) -> StopCondition:
        category = state_dict.get("type", params.type)
        return await cls._load_generic(
            directory,
            StopCondition,
            category=category,
            **state_dict.get("params", params.params.get(category, {})),
            default=partial(cls._build_stop_condition, params),
        )

    @classmethod
    async def _load_stop_condition_state(
        cls, params: StopConditionParams, directory: Path
    ) -> StopConditionState:
        state_dict = await cls._load_state_dict(directory)
        return StopConditionState(
            condition=await cls._load_stop_condition(
                state_dict, params, directory / "condition"
            ),
            params=params.params,
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        state_dict = await self._load_state_dict(directory)
        return State(
            generator=await self._load_generator(
                state_dict, directory / "generator"
            ),
            sample_size=state_dict.get("sample_size", params.sample_size),
            stop_condition=await self._load_stop_condition_state(
                params.stop_condition, directory / "stop_condition"
            ),
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.generator, Configurable):
                await state.generator.cleanup()
            if isinstance(state.stop_condition.condition, Configurable):
                await state.stop_condition.condition.cleanup()
        return await super().cleanup()


class Bootstrap(BootstrapBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            GeneratorParameter,
            SampleSizeParameter,
            StopConditionParameter,
        }

    async def should_stop(
        self, rewards: Sequence[Tensor], iteration: int
    ) -> bool:
        async with self.state.read_lock() as state:
            condition = state.stop_condition.condition
            return await condition.should_stop(rewards, iteration)

    async def generate(
        self, policy: ModelInfo[SequentialModel]
    ) -> List[Tuple[Tensor, Tensor]]:
        async with self.state.read_lock() as state:
            generator = state.generator
            sample_size = state.sample_size

        sequences = await generator.generate(policy, sample_size)
        return [
            (
                torch.tensor(context).long().view(-1, 1),
                torch.tensor(response).long().view(-1, 1),
            )
            for context, response in sequences
        ]
