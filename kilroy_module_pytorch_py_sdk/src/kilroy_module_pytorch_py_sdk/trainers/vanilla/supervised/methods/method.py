from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple, List

from torch import Tensor

from kilroy_module_pytorch_py_sdk import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.supervised.controls import (
    TrainingControls,
)
from kilroy_module_server_py_sdk import Metrizable
from kilroy_server_py_utils import Categorizable, classproperty, normalize


class Method(Metrizable, Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Method"))

    @abstractmethod
    async def fit(
        self,
        policy: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        regularizations: List[PolicyRegularization],
    ) -> None:
        pass
