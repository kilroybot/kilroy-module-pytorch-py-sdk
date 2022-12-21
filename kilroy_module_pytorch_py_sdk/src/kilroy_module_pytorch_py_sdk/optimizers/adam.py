from typing import Any, Dict, Iterable, List

from kilroy_module_server_py_sdk import SerializableModel, classproperty
from torch import Tensor
from torch.optim import Adam

from kilroy_module_pytorch_py_sdk.optimizers.base import (
    OptimizerParameter,
    StandardOptimizer,
    StandardOptimizerState as State,
)


class Params(SerializableModel):
    lr: float = 0.00001
    betas: List[float] = [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0


class AdamOptimizer(StandardOptimizer):
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

    class Beta1Parameter(OptimizerParameter[State, float]):
        @classmethod
        def _get_param(cls, group: Dict[str, Any]) -> float:
            return group["betas"][0]

        @classmethod
        def _set_param(cls, group: Dict[str, Any], value: float) -> None:
            group["betas"][0] = value

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.9,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Beta 1"

    class Beta2Parameter(OptimizerParameter[State, float]):
        @classmethod
        def _get_param(cls, group: Dict[str, Any]) -> float:
            return group["betas"][1]

        @classmethod
        def _set_param(cls, group: Dict[str, Any], value: float) -> None:
            group["betas"][1] = value

        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 0.999,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Beta 2"

    class EpsParameter(OptimizerParameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "title": cls.pretty_name,
                "default": 1e-8,
            }

        # noinspection PyMethodParameters
        @classproperty
        def pretty_name(cls) -> str:
            return "Epsilon"

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

    async def _build_default_optimizer(
        self, parameters: Iterable[Tensor]
    ) -> Adam:
        user_params = Params(**self._kwargs)
        return Adam(parameters, **user_params.dict())
