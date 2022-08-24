from abc import ABC, abstractmethod
from dataclasses import dataclass

from kilroy_module_server_py_sdk import Categorizable, classproperty, normalize
from torch import Tensor


@dataclass
class SampleResult:
    samples: Tensor
    logprobs: Tensor


class Sampler(Categorizable, ABC):
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Sampler"))

    @abstractmethod
    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        pass
