import asyncio
from abc import ABC
from functools import partial
from pathlib import Path
from typing import (
    AsyncIterable,
    Tuple,
    Collection,
    List,
    Set,
    Type,
    Dict,
    Hashable,
)

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.gae import GeneralizedAdvantageEstimator
from kilroy_module_pytorch_py_sdk.losses.policy import PolicyLoss
from kilroy_module_pytorch_py_sdk.losses.value import ValueLoss
from kilroy_module_pytorch_py_sdk.metrics import LossMetric
from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.report import MetricReport, Metrics
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.controls import (
    TrainingControls,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods.mbgd.parameters import (
    PolicyLossParameter,
    BatchSizeParameter,
    ValueLossParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods.mbgd.params import (
    PolicyParams,
    PolicyMetricsParams,
    PolicyLossParams,
    Params,
    ValueMetricsParams,
    ValueLossParams,
    ValueParams,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods.mbgd.state import (
    State,
    PolicyState,
    PolicyMetricsState,
    PolicyLossState,
    ValueMetricsState,
    ValueLossState,
    ValueState,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods.method import (
    Method,
)
from kilroy_module_pytorch_py_sdk.utils import (
    CachingAsyncIterable,
    pack_list,
    truncate_last_element,
    truncate_first_element,
    batched_forward,
    squash_packed,
    unpack_to_list,
)
from kilroy_module_pytorch_py_sdk.utils import freeze
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import Configurable, Savable, Parameter
from kilroy_server_py_utils.utils import batchify, background, classproperty


class MiniBatchGradientDescentMethodBase(Method, Configurable[State], ABC):
    @staticmethod
    async def _build_policy_metrics_state(
        params: PolicyMetricsParams,
    ) -> PolicyMetricsState:
        return PolicyMetricsState(
            base_batch_loss=await LossMetric.create(
                name=params.base_batch_loss.name,
                label=params.base_batch_loss.label,
                x_axis_key=params.base_batch_loss.x_axis_key,
                x_axis_label=params.base_batch_loss.x_axis_label,
            ),
            base_epoch_loss=await LossMetric.create(
                name=params.base_epoch_loss.name,
                label=params.base_epoch_loss.label,
                x_axis_key=params.base_epoch_loss.x_axis_key,
                x_axis_label=params.base_epoch_loss.x_axis_label,
            ),
            combined_batch_loss=await LossMetric.create(
                name=params.combined_batch_loss.name,
                label=params.combined_batch_loss.label,
                x_axis_key=params.combined_batch_loss.x_axis_key,
                x_axis_label=params.combined_batch_loss.x_axis_label,
            ),
            combined_epoch_loss=await LossMetric.create(
                name=params.combined_epoch_loss.name,
                label=params.combined_epoch_loss.label,
                x_axis_key=params.combined_epoch_loss.x_axis_key,
                x_axis_label=params.combined_epoch_loss.x_axis_label,
            ),
        )

    @classmethod
    async def _build_policy_loss_state(
        cls, params: PolicyLossParams
    ) -> PolicyLossState:
        loss = await cls._build_generic(
            PolicyLoss,
            category=params.type,
            **params.params.get(params.type, {}),
        )

        return PolicyLossState(loss=loss, params=params.params)

    @classmethod
    async def _build_policy_state(cls, params: PolicyParams) -> PolicyState:
        return PolicyState(
            metrics=await cls._build_policy_metrics_state(params.metrics),
            loss=await cls._build_policy_loss_state(params.loss),
        )

    @staticmethod
    async def _build_value_metrics_state(
        params: ValueMetricsParams,
    ) -> ValueMetricsState:
        return ValueMetricsState(
            base_batch_loss=await LossMetric.create(
                name=params.base_batch_loss.name,
                label=params.base_batch_loss.label,
                x_axis_key="batch",
                x_axis_label="Batch",
            ),
            base_epoch_loss=await LossMetric.create(
                name=params.base_epoch_loss.name,
                label=params.base_epoch_loss.label,
                x_axis_key="epoch",
                x_axis_label="Epoch",
            ),
        )

    @classmethod
    async def _build_value_loss_state(
        cls, params: ValueLossParams
    ) -> ValueLossState:
        loss = await cls._build_generic(
            ValueLoss,
            category=params.type,
            **params.params.get(params.type, {}),
        )

        return ValueLossState(loss=loss, params=params.params)

    @classmethod
    async def _build_value_state(cls, params: ValueParams) -> ValueState:
        return ValueState(
            metrics=await cls._build_value_metrics_state(params.metrics),
            loss=await cls._build_value_loss_state(params.loss),
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            policy=await self._build_policy_state(params.policy),
            value=await self._build_value_state(params.value),
            batch_size=params.batch_size,
        )

    @classmethod
    async def _save_policy_loss_state(
        cls, state: PolicyLossState, directory: Path
    ) -> None:
        if isinstance(state.loss, Savable):
            await state.loss.save(directory / "loss")

        state_dict = {"type": state.loss.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_policy_state(
        cls, state: PolicyState, directory: Path
    ) -> None:
        await cls._save_policy_loss_state(state.loss, directory / "loss")

    @classmethod
    async def _save_value_loss_state(
        cls, state: ValueLossState, directory: Path
    ) -> None:
        if isinstance(state.loss, Savable):
            await state.loss.save(directory / "loss")

        state_dict = {"type": state.loss.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_value_state(
        cls, state: ValueState, directory: Path
    ) -> None:
        await cls._save_value_loss_state(state.loss, directory / "loss")

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_policy_state(state.policy, directory / "policy")
        await cls._save_value_state(state.value, directory / "value")

        state_dict = {"batch_size": state.batch_size}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _load_policy_loss_state(
        cls, params: PolicyLossParams, directory: Path
    ) -> PolicyLossState:
        state_dict = await cls._load_state_dict(directory)
        category = state_dict.get("type", params.type)

        loss = await cls._load_generic(
            directory / "loss",
            PolicyLoss,
            category=category,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_policy_loss_state, params),
        )

        return PolicyLossState(
            loss=loss, params=state_dict.get("params", params.params)
        )

    @classmethod
    async def _load_policy_state(
        cls, params: PolicyParams, directory: Path
    ) -> PolicyState:
        return PolicyState(
            metrics=await cls._build_policy_metrics_state(params.metrics),
            loss=await cls._load_policy_loss_state(
                params.loss, directory / "loss"
            ),
        )

    @classmethod
    async def _load_value_loss_state(
        cls, params: ValueLossParams, directory: Path
    ) -> ValueLossState:
        state_dict = await cls._load_state_dict(directory)
        category = state_dict.get("type", params.type)

        loss = await cls._load_generic(
            directory / "loss",
            ValueLoss,
            category=category,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_value_loss_state, params),
        )

        return ValueLossState(
            loss=loss, params=state_dict.get("params", params.params)
        )

    @classmethod
    async def _load_value_state(
        cls, params: ValueParams, directory: Path
    ) -> ValueState:
        return ValueState(
            metrics=await cls._build_value_metrics_state(params.metrics),
            loss=await cls._load_value_loss_state(
                params.loss, directory / "loss"
            ),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        state_dict = await self._load_state_dict(directory)

        return State(
            policy=await self._load_policy_state(
                params.policy, directory / "policy"
            ),
            value=await self._load_value_state(
                params.value, directory / "value"
            ),
            batch_size=state_dict.get("batch_size", params.batch_size),
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.policy.loss.loss, Configurable):
                await state.policy.loss.loss.cleanup()
            if isinstance(state.value.loss.loss, Configurable):
                await state.value.loss.loss.cleanup()
            await state.policy.metrics.base_batch_loss.cleanup()
            await state.policy.metrics.base_epoch_loss.cleanup()
            await state.policy.metrics.combined_batch_loss.cleanup()
            await state.policy.metrics.combined_epoch_loss.cleanup()
            await state.value.metrics.base_batch_loss.cleanup()
            await state.value.metrics.base_epoch_loss.cleanup()
        return await super().cleanup()


class MiniBatchGradientDescentMethod(MiniBatchGradientDescentMethodBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            PolicyLossParameter,
            ValueLossParameter,
            BatchSizeParameter,
        }

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            return [
                state.policy.metrics.base_batch_loss,
                state.policy.metrics.base_epoch_loss,
                state.policy.metrics.combined_batch_loss,
                state.policy.metrics.combined_epoch_loss,
                state.value.metrics.base_epoch_loss,
                state.value.metrics.base_batch_loss,
            ]

    @staticmethod
    def _prepare_input_target(
        sequences: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[PackedSequence, PackedSequence]:
        full_sequences = [
            torch.vstack((context, response))
            for context, response in sequences
        ]
        input = pack_list(truncate_last_element(full_sequences))
        target = pack_list(truncate_first_element(full_sequences))
        return input, target

    @staticmethod
    async def _calculate_policy_regularization(
        logprobs: PackedSequence,
        baseline_logprobs: PackedSequence,
        regularization: PolicyRegularization,
    ) -> Tuple[Tensor, MetricReport]:
        loss = await regularization.calculate(
            logprobs.data, baseline_logprobs.data
        )
        report = MetricReport(
            values=loss.detach(),
            metrics=Metrics(
                direct=await regularization.get_direct_metric(),
                aggregated=await regularization.get_aggregated_metric(),
            ),
        )
        return loss, report

    async def _regularize_policy(
        self,
        base_losses: Tensor,
        input: PackedSequence,
        logprobs: PackedSequence,
        baseline: ModelInfo[SequentialModel],
        regularizations: List[PolicyRegularization],
    ) -> Tuple[Tensor, Dict[Hashable, MetricReport]]:
        combined_loss = base_losses.mean()
        reports = {}

        async with baseline.lock:
            with freeze(baseline.model) as frozen_baseline:
                baseline_logprobs = await batched_forward(
                    frozen_baseline, input, baseline.batch_size
                )
        for regularization in regularizations:
            loss, report = await self._calculate_policy_regularization(
                logprobs, baseline_logprobs, regularization
            )
            loss = (await regularization.get_weight()) * loss.mean()
            combined_loss = combined_loss + loss
            reports[id(regularization)] = report

        return combined_loss, reports

    async def _get_base_policy_report(self, losses: Tensor) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    direct=state.policy.metrics.base_batch_loss,
                    aggregated=state.policy.metrics.base_epoch_loss,
                ),
            )

    async def _get_combined_policy_report(
        self, losses: Tensor
    ) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    direct=state.policy.metrics.combined_batch_loss,
                    aggregated=state.policy.metrics.combined_epoch_loss,
                ),
            )

    async def _fit_policy_batch(
        self,
        policy: ModelInfo[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        sequences: List[Tuple[Tensor, Tensor]],
        loss: PolicyLoss,
        regularizations: List[PolicyRegularization],
    ) -> Dict[Hashable, MetricReport]:
        input, target = self._prepare_input_target(sequences)

        async with policy.lock:
            logprobs = await batched_forward(
                policy.model, input, policy.batch_size
            )

        base_losses = await loss.calculate(logprobs.data, target.data)
        base_loss = base_losses.mean()
        final_loss = base_loss
        reports = {"base": await self._get_base_policy_report(base_loss)}

        if regularizations:
            final_loss, regularization_reports = await self._regularize_policy(
                base_losses, input, logprobs, baseline, regularizations
            )
            reports = {**reports, **regularization_reports}

        reports["combined"] = await self._get_combined_policy_report(
            final_loss
        )
        await background(final_loss.backward)
        return reports

    @staticmethod
    async def _get_full_sequences(
        sequences: List[Tuple[Tensor, Tensor]]
    ) -> PackedSequence:
        return pack_list(
            [
                torch.vstack((context, response))
                for context, response in sequences
            ]
        )

    @staticmethod
    async def _get_timestep_rewards(
        sequences: PackedSequence,
        rewards: List[Tensor],
    ) -> PackedSequence:
        rewards = [
            torch.vstack((torch.zeros(len(sequence) - 1, 1), reward))
            for sequence, reward in zip(unpack_to_list(sequences), rewards)
        ]
        return pack_list(rewards)

    async def _get_base_value_report(self, losses: Tensor) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    direct=state.value.metrics.base_batch_loss,
                    aggregated=state.value.metrics.base_epoch_loss,
                ),
            )

    async def _fit_value_batch(
        self,
        value: ModelInfo[SequentialModel],
        sequences: List[Tuple[Tensor, Tensor]],
        rewards: List[Tensor],
        loss: ValueLoss,
        gae: GeneralizedAdvantageEstimator,
    ) -> Dict[Hashable, MetricReport]:
        full_sequences = await self._get_full_sequences(sequences)
        rewards = await self._get_timestep_rewards(full_sequences, rewards)

        async with value.lock:
            values = await batched_forward(
                value.model, full_sequences, value.batch_size
            )

        advantages = await gae.calculate(rewards, values)
        returns = squash_packed(values, partial(torch.add, advantages.data))

        base_losses = await loss.calculate(values.data, returns.data)
        base_loss = base_losses.mean()

        await background(base_loss.backward)

        return {"base": await self._get_base_value_report(base_loss)}

    @staticmethod
    async def _process_step_reports(
        new_reports: Dict[Hashable, MetricReport],
        all_reports: Dict[Hashable, List[MetricReport]],
        step: int,
    ) -> Dict[Hashable, List[MetricReport]]:
        _all_reports = {**all_reports}

        for key, report in new_reports.items():
            if report.metrics.direct is not None:
                await report.metrics.direct.report(
                    step, report.values.mean().item()
                )
            if key not in _all_reports:
                _all_reports[key] = []
            _all_reports[key] = _all_reports[key] + [report]

        return _all_reports

    @staticmethod
    async def _process_epoch_reports(
        reports: Dict[Hashable, List[MetricReport]],
        epoch: int,
    ) -> None:
        for key, reports in reports.items():
            first = reports[0]
            if first.metrics.aggregated is not None:
                await first.metrics.aggregated.report(
                    epoch,
                    torch.stack([r.values.mean() for r in reports])
                    .mean()
                    .item(),
                )

    async def _fit_policy(
        self,
        policy: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        regularizations: List[PolicyRegularization],
    ) -> None:
        while True:
            all_reports = {}

            async with self.state.read_lock() as state:
                batches = batchify(data, state.batch_size)

            async for batch in batches:
                batch = [(result[0], result[1]) async for result in batch]
                if not batch:
                    continue

                async with self.state.read_lock() as state:
                    loss = state.policy.loss.loss

                reports = await self._fit_policy_batch(
                    policy.model, baseline, batch, loss, regularizations
                )
                step = await policy.step()
                all_reports = await self._process_step_reports(
                    reports, all_reports, step - 1
                )

            epoch = await policy.increment_epoch()
            await self._process_epoch_reports(all_reports, epoch - 1)

    async def _fit_value(
        self,
        value: TrainingControls[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        gae: GeneralizedAdvantageEstimator,
    ) -> None:
        while True:
            all_reports = {}

            async with self.state.read_lock() as state:
                batches = batchify(data, state.batch_size)

            async for batch in batches:
                sequences, rewards = [], []
                async for context, response, _, reward in batch:
                    sequences.append((context, response))
                    rewards.append(reward)

                if not sequences:
                    continue

                async with self.state.read_lock() as state:
                    loss = state.value.loss.loss

                reports = await self._fit_value_batch(
                    value.model, sequences, rewards, loss, gae
                )
                step = await value.step()
                all_reports = await self._process_step_reports(
                    reports, all_reports, step - 1
                )

            epoch = await value.increment_epoch()
            await self._process_epoch_reports(all_reports, epoch - 1)

    async def fit(
        self,
        policy: TrainingControls[SequentialModel],
        value: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        gae: GeneralizedAdvantageEstimator,
        regularizations: List[PolicyRegularization],
    ) -> None:
        async with CachingAsyncIterable(data) as data:
            coros = [
                self._fit_policy(policy, baseline, data, regularizations),
                self._fit_value(value, data, gae),
            ]
            tasks = [asyncio.create_task(coro) for coro in coros]

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                for task in tasks:
                    task.cancel()
                raise e
