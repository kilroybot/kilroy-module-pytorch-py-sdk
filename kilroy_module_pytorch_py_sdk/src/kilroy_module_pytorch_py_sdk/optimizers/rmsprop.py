from typing import Any, Dict, Iterable

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch import Tensor
from torch.optim import RMSprop

from kilroy_module_pytorch_py_sdk.optimizers.base import (
    OptimizerParameter,
    StandardOptimizer,
    StandardOptimizerState as State,
)


class Params(SerializableModel):
    lr: float = 0.001
    momentum: float = 0
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0


class RMSPropOptimizer(StandardOptimizer):
    class LrParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class MomentumParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class AlphaParameter(OptimizerParameter[State, float]):
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

    async def _build_default_optimizer(
        self, parameters: Iterable[Tensor]
    ) -> RMSprop:
        user_params = Params(**self._kwargs)
        return RMSprop(parameters, **user_params.dict())
