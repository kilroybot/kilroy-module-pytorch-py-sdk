from torch import Tensor
from torch.nn import NLLLoss

from kilroy_module_pytorch_py_sdk.losses.policy.loss import PolicyLoss


class NegativeLogLikelihoodPolicyLoss(PolicyLoss):
    async def calculate(self, logprobs: Tensor, target: Tensor) -> Tensor:
        return NLLLoss(reduction="none")(logprobs, target.flatten())
