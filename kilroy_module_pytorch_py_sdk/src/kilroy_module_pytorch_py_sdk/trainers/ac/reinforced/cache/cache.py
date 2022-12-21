from abc import ABC
from collections.abc import Collection
from pathlib import Path
from typing import (
    Tuple,
    Iterable,
    AsyncIterable,
    Union,
    Set,
    Type,
)

import torch
from aiostream import stream
from torch import Tensor

from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.cache.parameters import (
    MaxSizeParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.cache.params import (
    Params,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.cache.state import (
    State,
)
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class CacheBase(Configurable[State], ABC):
    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(data=[], max_size=params.max_size)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        torch.save(state.data, directory / "data.pt")
        state_dict = {"max_size": state.max_size}
        await cls._save_state_dict(state_dict, directory)

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        data = torch.load(directory / "data.pt")
        state_dict = await self._load_state_dict(directory)
        return State(
            data=data, max_size=state_dict.get("max_size", params.max_size)
        )


class Cache(CacheBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            MaxSizeParameter,
        }

    async def append(
        self, data: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> None:
        async with self.state.write_lock() as state:
            state.data.append(data)
            if len(state.data) > state.max_size:
                state.data.pop(0)

    async def extend(
        self,
        data: Union[
            Iterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
            AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        ],
    ) -> None:
        async with stream.iterate(data).stream() as data:
            async for datum in data:
                await self.append(datum)

    async def get(self) -> Collection[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        async with self.state.read_lock() as state:
            return state.data

    async def clear(self) -> None:
        async with self.state.write_lock() as state:
            state.data = []

    async def get_size(self) -> int:
        async with self.state.read_lock() as state:
            return len(state.data)
