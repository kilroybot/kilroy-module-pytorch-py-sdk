from abc import ABC, abstractmethod
from typing import Sequence

from torch import Tensor

from kilroy_server_py_utils import Categorizable, classproperty, normalize


class ValueStopCondition(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("ValueStopCondition"))

    @abstractmethod
    async def should_stop(
        self, losses: Sequence[Tensor], iteration: int
    ) -> bool:
        pass
