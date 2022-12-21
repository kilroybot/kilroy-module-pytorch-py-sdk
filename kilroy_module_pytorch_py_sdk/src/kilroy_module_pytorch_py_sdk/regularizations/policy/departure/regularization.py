from abc import ABC
from functools import partial
from pathlib import Path
from typing import Set, Type, Collection

from torch import Tensor

from kilroy_module_pytorch_py_sdk.losses.distribution import DistributionLoss
from kilroy_module_pytorch_py_sdk.metrics import LossMetric
from kilroy_module_pytorch_py_sdk.regularizations.policy.base import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.departure.parameters import (
    WeightParameter,
    LossParameter,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.departure.params import (
    LossParams,
    Params,
    MetricsParams,
)
from kilroy_module_pytorch_py_sdk.regularizations.policy.departure.state import (
    State,
    LossState,
    MetricsState,
)
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import Configurable, classproperty, Parameter


class DeparturePolicyRegularizationBase(
    PolicyRegularization, Configurable[State], ABC
):
    @classmethod
    async def _build_loss(cls, params: LossParams) -> DistributionLoss:
        return await cls._build_generic(
            DistributionLoss, category=params.type, **params.params
        )

    @classmethod
    async def _build_loss_state(cls, params: LossParams) -> LossState:
        return LossState(
            loss=await cls._build_loss(params),
            params=params.params,
        )

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
            loss=await self._build_loss_state(params.loss),
            metrics=await self._build_metrics_state(params.metrics),
        )

    @classmethod
    async def _save_loss_state(cls, state: LossState, directory: Path) -> None:
        if isinstance(state.loss, Configurable):
            await state.loss.save(directory / "loss")

        state_dict = {"type": state.loss.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_loss_state(state.loss, directory / "loss")

        state_dict = {"beta": state.weight}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _load_loss_state(
        cls, params: LossParams, directory: Path
    ) -> LossState:
        state_dict = await cls._load_state_dict(directory)
        category = state_dict.get("type", params.type)
        loss = await cls._load_generic(
            directory / "loss",
            DistributionLoss,
            category=category,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_loss, params),
        )
        return LossState(
            loss=loss, params=state_dict.get("params", params.params)
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        state_dict = await self._load_state_dict(directory)
        return State(
            weight=state_dict.get("beta", params.weight),
            loss=await self._load_loss_state(params.loss, directory / "loss"),
            metrics=await self._build_metrics_state(params.metrics),
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        async with self.state.write_lock() as state:
            if isinstance(state.loss.loss, Configurable):
                await state.loss.loss.cleanup()
            if state.metrics.direct_loss is not None:
                await state.metrics.direct_loss.cleanup()
            if state.metrics.aggregated_loss is not None:
                await state.metrics.aggregated_loss.cleanup()


class DeparturePolicyRegularization(DeparturePolicyRegularizationBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            WeightParameter,
            LossParameter,
        }

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            metrics = []
            if state.metrics.direct_loss is not None:
                metrics.append(state.metrics.direct_loss)
            if state.metrics.aggregated_loss is not None:
                metrics.append(state.metrics.aggregated_loss)
            return metrics

    async def get_direct_metric(self) -> LossMetric:
        async with self.state.read_lock() as state:
            return state.metrics.direct_loss

    async def get_aggregated_metric(self) -> LossMetric:
        async with self.state.read_lock() as state:
            return state.metrics.aggregated_loss

    async def get_weight(self) -> float:
        async with self.state.read_lock() as state:
            return state.weight

    async def calculate(
        self,
        policy_logprobs: Tensor,
        baseline_logprobs: Tensor,
    ) -> Tensor:
        async with self.state.read_lock() as state:
            loss = state.loss.loss

        return await loss.calculate(
            policy_logprobs, baseline_logprobs.detach()
        )
