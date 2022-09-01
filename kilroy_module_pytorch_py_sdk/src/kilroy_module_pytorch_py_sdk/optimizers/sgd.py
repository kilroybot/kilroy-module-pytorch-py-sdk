from typing import Any, Dict, Iterable

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch import Tensor
from torch.optim import SGD

from kilroy_module_pytorch_py_sdk.optimizers.base import (
    OptimizerParameter,
    StandardOptimizer,
    StandardOptimizerState as State,
)


class Params(SerializableModel):
    lr: float = 0.001
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0


class SGDOptimizer(StandardOptimizer):
    class LrParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class MomentumParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class DampeningParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def _build_default_optimizer(
        self, parameters: Iterable[Tensor]
    ) -> SGD:
        user_params = Params(**self._kwargs)
        return SGD(parameters, **user_params.dict())
