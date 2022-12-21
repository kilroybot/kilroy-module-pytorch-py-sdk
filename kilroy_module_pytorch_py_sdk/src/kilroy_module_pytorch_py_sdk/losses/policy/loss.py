from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroy_server_py_utils import Categorizable, classproperty, normalize


class PolicyLoss(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("PolicyLoss"))

    @abstractmethod
    async def calculate(self, logprobs: Tensor, target: Tensor) -> Tensor:
        pass
