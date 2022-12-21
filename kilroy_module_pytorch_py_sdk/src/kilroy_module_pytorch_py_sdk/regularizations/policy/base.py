from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from kilroy_module_pytorch_py_sdk.metrics import LossMetric
from kilroy_module_server_py_sdk import Metrizable
from kilroy_server_py_utils import Categorizable, classproperty, normalize


class PolicyRegularization(Metrizable, Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("PolicyRegularization"))

    @abstractmethod
    async def get_weight(self) -> float:
        pass

    @abstractmethod
    async def get_direct_metric(self) -> Optional[LossMetric]:
        pass

    @abstractmethod
    async def get_aggregated_metric(self) -> Optional[LossMetric]:
        pass

    @abstractmethod
    async def calculate(
        self,
        policy_logprobs: Tensor,
        baseline_logprobs: Tensor,
    ) -> Tensor:
        pass
