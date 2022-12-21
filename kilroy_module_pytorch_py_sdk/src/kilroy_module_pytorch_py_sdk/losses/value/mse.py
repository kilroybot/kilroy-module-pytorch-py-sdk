from torch import Tensor
from torch.nn import MSELoss

from kilroy_module_pytorch_py_sdk.losses.value.loss import ValueLoss


class MeanSquaredErrorValueLoss(ValueLoss):
    async def calculate(self, predicted: Tensor, target: Tensor) -> Tensor:
        return MSELoss(reduction="none")(predicted, target)
