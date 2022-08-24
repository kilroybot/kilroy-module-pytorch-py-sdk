import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical

from kilroy_module_pytorch_py_sdk.samplers.base import SampleResult


def _mutate_raw_samples(
    samples: Tensor, n_choices: int, epsilon: float
) -> Tensor:
    mutation_mask = Bernoulli(epsilon).sample(samples.shape).bool()
    uniform_probs = torch.ones(n_choices)
    uniform_samples = Categorical(uniform_probs).sample(samples.shape)
    samples[mutation_mask] = uniform_samples[mutation_mask]
    return samples


def _select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
    logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
    return logprobs.gather(-1, samples)


def mutate(
    result: SampleResult, logprobs: Tensor, epsilon: float
) -> SampleResult:
    samples = _mutate_raw_samples(result.samples, logprobs.shape[-1], epsilon)
    return SampleResult(
        samples=samples,
        logprobs=_select_logprobs(logprobs, samples),
    )
