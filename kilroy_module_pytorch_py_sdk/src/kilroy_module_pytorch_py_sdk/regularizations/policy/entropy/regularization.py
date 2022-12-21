from abc import ABC
from pathlib import Path
from typing import Set, Type, Collection, Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from kilroy_module_pytorch_py_sdk.metrics import LossMetric
from kilroy_module_pytorch_py_sdk.regularizations.policy.base import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.entropy.parameters import (
    WeightParameter,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.entropy.params import (
    Params,
    MetricsParams,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.entropy.state import (
    State,
    MetricsState,
)
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import Configurable, classproperty, Parameter


class EntropyPolicyRegularizationBase(
    PolicyRegularization, Configurable[State], ABC
):
    @staticmethod
    async def _build_metrics_state(
        params: MetricsParams,
    ) -> MetricsState:
        return MetricsState(
            direct_loss=await LossMetric.create(
                name=params.direct_loss.name,
                label=params.direct_loss.label,
                x_axis_key=params.direct_loss.x_axis_key,
                x_axis_label=params.direct_loss.x_axis_label,
            )
            if params.direct_loss is not None
            else None,
            aggregated_loss=await LossMetric.create(
                name=params.aggregated_loss.name,
                label=params.aggregated_loss.label,
                x_axis_key=params.aggregated_loss.x_axis_key,
                x_axis_label=params.aggregated_loss.x_axis_label,
            )
            if params.aggregated_loss is not None
            else None,
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            weight=params.weight,
            metrics=await self._build_metrics_state(params.metrics),
        )

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        state_dict = {"weight": state.weight}
        await cls._save_state_dict(state_dict, directory)

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        state_dict = await self._load_state_dict(directory)
        return State(
            weight=state_dict.get("weight", params.weight),
            metrics=await self._build_metrics_state(params.metrics),
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if state.metrics.direct_loss is not None:
                await state.metrics.direct_loss.cleanup()
            if state.metrics.aggregated_loss is not None:
                await state.metrics.aggregated_loss.cleanup()
        return await super().cleanup()


class EntropyPolicyRegularization(EntropyPolicyRegularizationBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            WeightParameter,
        }

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            metrics = []
            if state.metrics.direct_loss is not None:
                metrics.append(state.metrics.direct_loss)
            if state.metrics.aggregated_loss is not None:
                metrics.append(state.metrics.aggregated_loss)
            return metrics

    async def get_direct_metric(self) -> Optional[LossMetric]:
        async with self.state.read_lock() as state:
            return state.metrics.direct_loss

    async def get_aggregated_metric(self) -> Optional[LossMetric]:
        async with self.state.read_lock() as state:
            return state.metrics.aggregated_loss

    async def get_weight(self) -> float:
        async with self.state.read_lock() as state:
            return state.weight

    async def calculate(
        self, policy_logprobs: Tensor, baseline_logprobs: Tensor
    ) -> Tensor:
        n_classes = policy_logprobs.shape[-1]
        logprobs = policy_logprobs.view(-1, n_classes)

        policy_entropy = Categorical(logits=logprobs).entropy()
        if n_classes > 1:
            policy_entropy = policy_entropy / torch.tensor(n_classes).log()

        baseline_entropy = Categorical(logits=baseline_logprobs).entropy()
        if n_classes > 1:
            baseline_entropy = baseline_entropy / torch.tensor(n_classes).log()

        return (baseline_entropy - policy_entropy).abs()
