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


def _filter_logprobs(logprobs: Tensor, k: int) -> Tensor:
    top = logprobs.topk(k)
    top_exp = top.values.exp()
    new_logprobs = torch.full_like(logprobs, -torch.inf)
    return new_logprobs.scatter(
        -1,
        top.indices,
        (top_exp / top_exp.sum(-1).unsqueeze(-1)).log(),
    )


def _select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
    logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
    return logprobs.gather(-1, samples)


def sample(logprobs: Tensor, n: int, k: int) -> SampleResult:
    filtered = _filter_logprobs(logprobs, k)
    result = proportional_sample(filtered, n)
    return SampleResult(
        samples=result.samples,
        logprobs=_select_logprobs(logprobs, result.samples),
    )


class State(SerializableState):
    k: int = 10


class TopKSampler(Sampler, Configurable[State]):
    class KParameter(Parameter[State, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        async with self.state.read_lock() as state:
            return sample(logprobs, n, state.k)


# Epsilon


class EpsilonState(SerializableState):
    k: int = 10
    epsilon: float = 0.01


class EpsilonTopKSampler(Sampler, Configurable[EpsilonState]):
    class KParameter(Parameter[EpsilonState, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(self) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        async with self.state.read_lock() as state:
            return mutate(
                sample(logprobs, n, state.k), logprobs, state.epsilon
            )
