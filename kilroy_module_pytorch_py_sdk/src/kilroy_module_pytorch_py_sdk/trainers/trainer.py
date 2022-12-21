from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple

from torch import Tensor

from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_server_py_sdk import Metrizable
from kilroy_server_py_utils import Categorizable, classproperty, normalize


class Trainer(Metrizable, Categorizable, ABC):
    def __init__(self, *args, models: ModelsRegistry, **kwargs) -> None:
        self._models = models
        super().__init__(*args, **kwargs)

    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Trainer"))

    @abstractmethod
    async def fit_supervised(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        pass

    @abstractmethod
    async def fit_reinforced(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        pass
