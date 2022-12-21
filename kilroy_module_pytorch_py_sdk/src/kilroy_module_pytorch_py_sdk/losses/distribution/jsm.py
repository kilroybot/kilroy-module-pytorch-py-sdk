import torch
from torch import Tensor
from torch.functional import F

from kilroy_module_pytorch_py_sdk.losses.distribution.loss import (
    DistributionLoss,
)


class JensenShannonMetricDistributionLoss(DistributionLoss):
    async def calculate(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(-1, input.size(-1))
        target = target.view(-1, target.size(-1))
        m = torch.tensor(0.5).log() + torch.logaddexp(input, target)

        kld_x = F.kl_div(
            m,
            input,
            reduction="none",
            log_target=True,
        )
        kld_x = kld_x.sum(dim=-1).clamp(min=0) / torch.tensor(2.0).log()

        kld_y = F.kl_div(
            m,
            target,
            reduction="none",
            log_target=True,
        )
        kld_y = kld_y.sum(dim=-1).clamp(min=0) / torch.tensor(2.0).log()

        jsd = 0.5 * (kld_x + kld_y)
        return torch.sqrt(jsd + 1e-8)
