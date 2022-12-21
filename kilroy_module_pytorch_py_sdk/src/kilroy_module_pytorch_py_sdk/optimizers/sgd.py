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
    lr: float = 0.00001
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0


class SGDOptimizer(StandardOptimizer):
    class LrParameter(OptimizerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.00001,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Learning Rate"

    class MomentumParameter(OptimizerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0,
            }

    class WeightDecayParameter(OptimizerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0,
            }

    class DampeningParameter(OptimizerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0,
            }

    async def _build_default_optimizer(
        self, parameters: Iterable[Tensor]
    ) -> SGD:
        user_params = Params(**self._kwargs)
        return SGD(parameters, **user_params.dict())
