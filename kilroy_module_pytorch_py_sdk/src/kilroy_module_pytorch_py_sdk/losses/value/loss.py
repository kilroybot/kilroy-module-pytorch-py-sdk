from abc import ABC, abstractmethod

from torch import Tensor

from kilroy_server_py_utils import Categorizable, classproperty, normalize


class ValueLoss(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("ValueLoss"))

    @abstractmethod
    async def calculate(self, predicted: Tensor, target: Tensor) -> Tensor:
        pass
