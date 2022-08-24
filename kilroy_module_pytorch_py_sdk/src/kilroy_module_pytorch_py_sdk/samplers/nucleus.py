from typing import Any, Dict

import torch
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
)
from torch import Tensor

from kilroy_module_pytorch_py_sdk.samplers.base import SampleResult, Sampler
from kilroy_module_pytorch_py_sdk.samplers.epsilon import mutate
from kilroy_module_pytorch_py_sdk.samplers.proportional import (
    sample as proportional_sample,
)


def _filter_logprobs(logprobs: Tensor, p: float) -> Tensor:
    sorted_logprobs = logprobs.sort(descending=True)
    cumulative_probs = sorted_logprobs.values.exp().cumsum(-1)
    sorted_top_indices = cumulative_probs <= p
    sorted_top_indices[..., 1:] = sorted_top_indices[..., :-1].clone()
    sorted_top_indices[..., 0] = True
    top_indices = sorted_top_indices.gather(
        -1, sorted_logprobs.indices.argsort()
    )
    new_logprobs = torch.full_like(logprobs, -torch.inf)
    return new_logprobs.masked_scatter(top_indices, logprobs[top_indices])


def _select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
    logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
    return logprobs.gather(-1, samples)


def sample(logprobs: Tensor, n: int, p: float) -> SampleResult:
    filtered_logprobs = _filter_logprobs(logprobs, p)
    result = proportional_sample(filtered_logprobs, n)
    return SampleResult(
        samples=result.samples,
        logprobs=_select_logprobs(logprobs, result.samples),
    )


class State(SerializableState):
    p: float = 0.95


class NucleusSampler(Sampler, Configurable[State]):
    class PParameter(Parameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        async with self.state.read_lock() as state:
            return sample(logprobs, n, state.p)


# EpsilonNucleus


class EpsilonState(SerializableState):
    p: float = 0.95
    epsilon: float = 0.01


class EpsilonNucleusSampler(Sampler, Configurable[EpsilonState]):
    class PParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        async with self.state.read_lock() as state:
            return mutate(
                sample(logprobs, n, state.p), logprobs, state.epsilon
            )
