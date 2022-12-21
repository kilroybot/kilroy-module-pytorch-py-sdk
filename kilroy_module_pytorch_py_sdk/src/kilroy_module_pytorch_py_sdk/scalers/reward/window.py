from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Dict, Any, List

import torch
from kilroy_module_py_shared import SerializableModel

from kilroy_module_pytorch_py_sdk.scalers.reward.base import RewardScaler
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class Params(SerializableModel):
    length: int = 1000


@dataclass
class State:
    buffer: List[float]
    length: int


class WindowRewardScaler(RewardScaler, Configurable[State]):
    class LengthParameter(Parameter[State, int]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "integer",
                "minimum": 1,
                "title": cls.pretty_name,
                "default": 1000,
            }

    async def _build_default_state(self) -> State:
        param = Params(**self._kwargs)
        return State(buffer=[], length=param.length)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state_dict = {"buffer": state.buffer, "length": state.length}
        await cls._save_state_dict(state_dict, directory)

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        state_dict = await self._load_state_dict(directory)
        return State(
            buffer=state_dict.get("buffer", []),
            length=state_dict.get("length", params.length),
        )

    async def scale(self, reward: float) -> float:
        async with self.state.write_lock() as state:
            state.buffer.append(reward)
            if len(state.buffer) > state.length:
                state.buffer.pop(0)
            buffer = torch.tensor(state.buffer).float()
        mean, std = buffer.mean().item(), buffer.std().item()
        if isnan(std) or std == 0:
            return 0.0
        return (reward - mean) / std
