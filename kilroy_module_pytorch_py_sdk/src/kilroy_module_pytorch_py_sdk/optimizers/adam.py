from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from kilroy_module_server_py_sdk import (
    Configurable,
    SerializableModel,
    background,
    classproperty,
)
from torch.optim import Adam

from kilroy_module_pytorch_py_sdk.optimizers.base import (
    Optimizer,
    OptimizerParameter,
)


class Params(SerializableModel):
    lr: float = 0.001
    betas: List[float] = [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class State:
    optimizer: Adam


class AdamOptimizer(Optimizer, Configurable[State]):
    class LrParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class Beta1Parameter(OptimizerParameter[State, float]):
        def _get_param(self, group: Dict[str, Any]) -> float:
            return group["betas"][0]

        def _set_param(self, group: Dict[str, Any], value: float) -> None:
            group["betas"][0] = value

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class Beta2Parameter(OptimizerParameter[State, float]):
        def _get_param(self, group: Dict[str, Any]) -> float:
            return group["betas"][1]

        def _set_param(self, group: Dict[str, Any], value: float) -> None:
            group["betas"][1] = value

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class EpsParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def build_default_state(self) -> State:
        model_params = self._kwargs.pop("parameters")
        user_params = Params(**self._kwargs)
        return State(optimizer=Adam(model_params, **user_params.dict()))

    @classmethod
    async def save_state(cls, state: State, directory: Path) -> None:
        with open(directory / "optimizer.pt", "wb") as f:
            await background(torch.save, state.optimizer.state_dict(), f)

    async def load_saved_state(self, directory: Path) -> State:
        with open(directory / "optimizer.pt", "rb") as f:
            state_dict = await background(torch.load, f)
        optimizer = Adam(self._kwargs.pop("parameters"))
        optimizer.load_state_dict(state_dict)
        return State(optimizer=optimizer)

    async def step(self) -> None:
        async with self.state.write_lock() as state:

            def step():
                state.optimizer.step()
                state.optimizer.zero_grad()

            await background(step)
