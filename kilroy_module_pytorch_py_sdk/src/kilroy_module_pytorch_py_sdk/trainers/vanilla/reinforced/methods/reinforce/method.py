from abc import ABC
from pathlib import Path
from typing import (
    AsyncIterable,
    Tuple,
    Collection,
    List,
    Dict,
    Hashable,
)

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk import SequentialModel
from kilroy_module_pytorch_py_sdk.metrics import ScoreMetric, LossMetric
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.report import MetricReport, Metrics
from kilroy_module_pytorch_py_sdk.trainers.vanilla.reinforced.controls import (
    TrainingControls,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.reinforced.methods.method import (
    Method,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.reinforced.methods.reinforce.params import (
    PolicyParams,
    PolicyMetricsParams,
    Params,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.reinforced.methods.reinforce.state import (
    State,
    PolicyState,
    PolicyMetricsState,
)
from kilroy_module_pytorch_py_sdk.utils import freeze
from kilroy_module_pytorch_py_sdk.utils import (
    pack_list,
    gather_logprobs,
    unpack_to_list,
    batched_forward,
)
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import Configurable
from kilroy_server_py_utils.utils import background


class ReinforceMethodBase(Method, Configurable[State], ABC):
    @staticmethod
    async def _build_policy_metrics_state(
        params: PolicyMetricsParams,
    ) -> PolicyMetricsState:
        return PolicyMetricsState(
            episode_score=await ScoreMetric.create(
                name=params.episode_score.name,
                label=params.episode_score.label,
                x_axis_key=params.episode_score.x_axis_key,
                x_axis_label=params.episode_score.x_axis_label,
            ),
            episode_reward=await ScoreMetric.create(
                name=params.episode_reward.name,
                label=params.episode_reward.label,
                x_axis_key=params.episode_reward.x_axis_key,
                x_axis_label=params.episode_reward.x_axis_label,
            ),
            base_episode_loss=await LossMetric.create(
                name=params.base_episode_loss.name,
                label=params.base_episode_loss.label,
                x_axis_key=params.base_episode_loss.x_axis_key,
                x_axis_label=params.base_episode_loss.x_axis_label,
            ),
            combined_episode_loss=await LossMetric.create(
                name=params.combined_episode_loss.name,
                label=params.combined_episode_loss.label,
                x_axis_key=params.combined_episode_loss.x_axis_key,
                x_axis_label=params.combined_episode_loss.x_axis_label,
            ),
        )

    @classmethod
    async def _build_policy_state(cls, params: PolicyParams) -> PolicyState:
        return PolicyState(
            metrics=await cls._build_policy_metrics_state(params.metrics),
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            policy=await self._build_policy_state(params.policy),
        )

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        pass

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)

        return State(policy=await self._build_policy_state(params.policy))

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            await state.policy.metrics.episode_score.cleanup()
            await state.policy.metrics.episode_reward.cleanup()
            await state.policy.metrics.base_episode_loss.cleanup()
            await state.policy.metrics.combined_episode_loss.cleanup()
        return await super().cleanup()


class ReinforceMethod(ReinforceMethodBase):
    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            return [
                state.policy.metrics.episode_score,
                state.policy.metrics.episode_reward,
                state.policy.metrics.base_episode_loss,
                state.policy.metrics.combined_episode_loss,
            ]

    @staticmethod
    async def _get_full_sequences(
        sequences: List[Tuple[Tensor, Tensor]],
    ) -> PackedSequence:
        return pack_list(
            [
                torch.vstack((context, response))
                for context, response in sequences
            ]
        )

    @staticmethod
    async def _calculate_regularization(
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

    async def _regularize(
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
            loss, report = await self._calculate_regularization(
                logprobs, baseline_logprobs, regularization
            )
            loss = (await regularization.get_weight()) * loss.mean()
            combined_loss = combined_loss + loss
            reports[id(regularization)] = report

        return combined_loss, reports

    async def _get_base_report(self, losses: Tensor) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    aggregated=state.policy.metrics.base_episode_loss,
                ),
            )

    async def _get_combined_report(self, losses: Tensor) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    aggregated=state.policy.metrics.combined_episode_loss,
                ),
            )

    @staticmethod
    async def _calculate_base_losses(
        logprobs: PackedSequence,
        sequences: List[Tuple[Tensor, Tensor]],
        rewards: Tensor,
    ) -> Tensor:
        gathered_logprobs = await gather_logprobs(logprobs, sequences)
        summed_logprobs = torch.vstack(
            [lp.sum() for lp in unpack_to_list(gathered_logprobs)]
        )

        return -(summed_logprobs * rewards)

    async def _fit_batch(
        self,
        policy: ModelInfo[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        sequences: List[Tuple[Tensor, Tensor]],
        rewards: Tensor,
        regularizations: List[PolicyRegularization],
    ) -> Dict[Hashable, MetricReport]:
        full_sequences = await self._get_full_sequences(sequences)

        async with policy.lock:
            logprobs = await batched_forward(
                policy.model, full_sequences, policy.batch_size
            )

        base_losses = await self._calculate_base_losses(
            logprobs, sequences, rewards
        )
        base_loss = base_losses.mean()
        final_loss = base_loss
        reports = {"base": await self._get_base_report(base_loss)}

        if regularizations:
            final_loss, regularization_reports = await self._regularize(
                base_losses,
                full_sequences,
                logprobs,
                baseline,
                regularizations,
            )
            reports = {**reports, **regularization_reports}

        reports["combined"] = await self._get_combined_report(final_loss)
        await background(final_loss.backward)
        return reports

    async def fit(
        self,
        policy: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        regularizations: List[PolicyRegularization],
    ) -> None:
        data = [x async for x in data]
        if not data:
            return

        sequences = [(result[0], result[1]) for result in data]
        scores = torch.vstack([result[2] for result in data])
        rewards = torch.vstack([result[3] for result in data])

        reports = await self._fit_batch(
            policy.model, baseline, sequences, rewards, regularizations
        )
        await policy.step()
        episode = await policy.increment_episode()

        for key, report in reports.items():
            if report.metrics.aggregated is not None:
                await report.metrics.aggregated.report(
                    episode - 1, report.values.mean().item()
                )

        async with self.state.write_lock() as state:
            await state.policy.metrics.episode_score.report(
                episode - 1, scores.mean().item()
            )
            await state.policy.metrics.episode_reward.report(
                episode - 1, rewards.mean().item()
            )
