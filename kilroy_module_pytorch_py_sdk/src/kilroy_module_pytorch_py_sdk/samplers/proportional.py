from typing import Any, Dict

from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
)
from torch import Tensor
from torch.distributions import Categorical

from kilroy_module_pytorch_py_sdk.samplers.base import SampleResult, Sampler
from kilroy_module_pytorch_py_sdk.samplers.epsilon import mutate


def sample(logprobs: Tensor, n: int) -> SampleResult:
    dist = Categorical(logits=logprobs, validate_args=False)
    samples = dist.sample((n,))
    return SampleResult(
        samples=samples.permute(1, 0),
        logprobs=dist.log_prob(samples).permute(1, 0),
    )


class ProportionalSampler(Sampler):
    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        return sample(logprobs, n)


# Epsilon


class EpsilonState(SerializableState):
    epsilon: float = 0.01


class EpsilonProportionalSampler(Sampler, Configurable[EpsilonState]):
    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        async with self.state.read_lock() as state:
            return mutate(sample(logprobs, n), logprobs, state.epsilon)
