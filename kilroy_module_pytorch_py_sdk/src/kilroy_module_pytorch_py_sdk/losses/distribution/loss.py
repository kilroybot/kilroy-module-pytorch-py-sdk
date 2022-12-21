from abc import ABC, abstractmethod

from torch import Tensor

from kilroy_server_py_utils import Categorizable, classproperty, normalize


class DistributionLoss(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("DistributionLoss"))

    @abstractmethod
    async def calculate(self, input: Tensor, target: Tensor) -> Tensor:
        pass
