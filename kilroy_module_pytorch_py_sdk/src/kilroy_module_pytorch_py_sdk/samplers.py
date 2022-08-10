from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical


@dataclass
class SampleResult:
    samples: Tensor
    logprobs: Tensor


class Sampler(ABC):
    @abstractmethod
    def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        pass


class ProportionalCategoricalSampler(Sampler):
    def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        dist = Categorical(logits=logprobs, validate_args=False)
        samples = dist.sample((n,))
        return SampleResult(
            samples=samples.permute(1, 0),
            logprobs=dist.log_prob(samples).permute(1, 0),
        )


class TopKCategoricalSampler(Sampler):
    def __init__(self, k: int) -> None:
        super().__init__()
        self._k = k
        self._base_sampler = ProportionalCategoricalSampler()

    @property
    def k(self) -> int:
        return self._k

    @staticmethod
    def filter_logprobs(logprobs: Tensor, k: int) -> Tensor:
        top = logprobs.topk(k)
        top_exp = top.values.exp()
        new_logprobs = torch.full_like(logprobs, -torch.inf)
        return new_logprobs.scatter(
            -1,
            top.indices,
            (top_exp / top_exp.sum(-1).unsqueeze(-1)).log(),
        )

    @staticmethod
    def select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
        logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
        return logprobs.gather(-1, samples)

    def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        result = self._base_sampler.sample(
            self.filter_logprobs(logprobs, self.k), n
        )
        return SampleResult(
            samples=result.samples,
            logprobs=self.select_logprobs(logprobs, result.samples),
        )


class NucleusCategoricalSampler(Sampler):
    def __init__(self, p: float) -> None:
        super().__init__()
        self._p = p
        self._base_sampler = ProportionalCategoricalSampler()

    @property
    def p(self) -> float:
        return self._p

    @staticmethod
    def filter_logprobs(logprobs: Tensor, p: float) -> Tensor:
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

    @staticmethod
    def select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
        logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
        return logprobs.gather(-1, samples)

    def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        result = self._base_sampler.sample(
            self.filter_logprobs(logprobs, self.p), n
        )
        return SampleResult(
            samples=result.samples,
            logprobs=self.select_logprobs(logprobs, result.samples),
        )


class WithEpsilon(Sampler):
    def __init__(self, *args, epsilon: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def mutate_samples(self, n_choices: int, samples: Tensor) -> Tensor:
        mutation_mask = Bernoulli(self.epsilon).sample(samples.shape).bool()
        uniform_probs = torch.ones(n_choices)
        uniform_samples = Categorical(uniform_probs).sample(samples.shape)
        samples[mutation_mask] = uniform_samples[mutation_mask]
        return samples

    @staticmethod
    def select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
        logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
        return logprobs.gather(-1, samples)

    def sample(self, logprobs: Tensor, n: int = 1) -> SampleResult:
        result = super().sample(logprobs, n)
        samples = self.mutate_samples(logprobs.shape[-1], result.samples)
        return SampleResult(
            samples=samples,
            logprobs=self.select_logprobs(logprobs, samples),
        )


class EpsilonProportionalCategoricalSampler(
    WithEpsilon, ProportionalCategoricalSampler
):
    pass


class EpsilonTopKCategoricalSampler(WithEpsilon, TopKCategoricalSampler):
    pass


class EpsilonNucleusCategoricalSampler(WithEpsilon, NucleusCategoricalSampler):
    pass
