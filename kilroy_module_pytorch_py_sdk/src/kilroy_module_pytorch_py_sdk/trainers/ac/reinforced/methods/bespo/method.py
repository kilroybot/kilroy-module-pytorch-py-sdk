from abc import ABC
from functools import partial
from pathlib import Path
from typing import (
    AsyncIterable,
    Tuple,
    Collection,
    Set,
    Type,
    List,
    Dict,
    Hashable,
    Any,
    Iterable,
)

import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.gae import GeneralizedAdvantageEstimator
from kilroy_module_pytorch_py_sdk.metrics import (
    ScoreMetric,
    LossMetric,
)
from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.report import MetricReport, Metrics
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.controls import (
    TrainingControls,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.bootstrap.bootstrap import (
    Bootstrap,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.parameters import (
    PolicyStopConditionParameter,
    ValueStopConditionParameter,
    BootstrapParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.params import (
    PolicyParams,
    PolicyMetricsParams,
    Params,
    ValueMetricsParams,
    ValueParams,
    PolicyStopConditionParams,
    ValueStopConditionParams,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.state import (
    State,
    PolicyState,
    PolicyMetricsState,
    ValueMetricsState,
    ValueState,
    PolicyStopConditionState,
    ValueStopConditionState,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.policy import (
    PolicyStopCondition,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.bespo.stop.value import (
    ValueStopCondition,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.reinforced.methods.method import (
    Method,
)
from kilroy_module_pytorch_py_sdk.utils import freeze
from kilroy_module_pytorch_py_sdk.utils import (
    pack_list,
    gather_logprobs,
    batched_forward,
    squash_packed,
    unpack_to_list,
)
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import Configurable, Parameter
from kilroy_server_py_utils.utils import background, classproperty


class BootstrappingEarlyStoppingPolicyOptimizationMethodBase(
    Method, Configurable[State], ABC
):
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
            bootstrap_episode_reward=await ScoreMetric.create(
                name=params.bootstrap_episode_reward.name,
                label=params.bootstrap_episode_reward.label,
                x_axis_key=params.bootstrap_episode_reward.x_axis_key,
                x_axis_label=params.bootstrap_episode_reward.x_axis_label,
            ),
            base_bootstrap_iteration_loss=await LossMetric.create(
                name=params.base_bootstrap_iteration_loss.name,
                label=params.base_bootstrap_iteration_loss.label,
                x_axis_key=params.base_bootstrap_iteration_loss.x_axis_key,
                x_axis_label=params.base_bootstrap_iteration_loss.x_axis_label,
            ),
            base_bootstrap_episode_loss=await LossMetric.create(
                name=params.base_bootstrap_episode_loss.name,
                label=params.base_bootstrap_episode_loss.label,
                x_axis_key=params.base_bootstrap_episode_loss.x_axis_key,
                x_axis_label=params.base_bootstrap_episode_loss.x_axis_label,
            ),
            combined_bootstrap_iteration_loss=await LossMetric.create(
                name=params.combined_bootstrap_iteration_loss.name,
                label=params.combined_bootstrap_iteration_loss.label,
                x_axis_key=params.combined_bootstrap_iteration_loss.x_axis_key,
                x_axis_label=params.combined_bootstrap_iteration_loss.x_axis_label,
            ),
            combined_bootstrap_episode_loss=await LossMetric.create(
                name=params.combined_bootstrap_episode_loss.name,
                label=params.combined_bootstrap_episode_loss.label,
                x_axis_key=params.combined_bootstrap_episode_loss.x_axis_key,
                x_axis_label=params.combined_bootstrap_episode_loss.x_axis_label,
            ),
        )

    @classmethod
    async def _build_policy_stop_condition(
        cls, params: PolicyStopConditionParams
    ) -> PolicyStopCondition:
        return await cls._build_generic(
            PolicyStopCondition,
            category=params.type,
            **params.params,
        )

    @classmethod
    async def _build_policy_stop_condition_state(
        cls, params: PolicyStopConditionParams
    ) -> PolicyStopConditionState:
        return PolicyStopConditionState(
            condition=await cls._build_policy_stop_condition(params),
            params=params.params,
        )

    @classmethod
    async def _build_policy_state(cls, params: PolicyParams) -> PolicyState:
        return PolicyState(
            metrics=await cls._build_policy_metrics_state(params.metrics),
            stop_condition=await cls._build_policy_stop_condition_state(
                params.stop_condition
            ),
        )

    @staticmethod
    async def _build_value_metrics_state(
        params: ValueMetricsParams,
    ) -> ValueMetricsState:
        return ValueMetricsState(
            base_iteration_loss=await LossMetric.create(
                name=params.base_iteration_loss.name,
                label=params.base_iteration_loss.label,
                x_axis_key="iteration",
                x_axis_label="Iteration",
            ),
            base_episode_loss=await LossMetric.create(
                name=params.base_episode_loss.name,
                label=params.base_episode_loss.label,
                x_axis_key="episode",
                x_axis_label="Episode",
            ),
        )

    @classmethod
    async def _build_value_stop_condition(
        cls, params: ValueStopConditionParams
    ) -> ValueStopCondition:
        return await cls._build_generic(
            ValueStopCondition,
            category=params.type,
            **params.params,
        )

    @classmethod
    async def _build_value_stop_condition_state(
        cls, params: ValueStopConditionParams
    ) -> ValueStopConditionState:
        return ValueStopConditionState(
            condition=await cls._build_value_stop_condition(params),
            params=params.params,
        )

    @classmethod
    async def _build_value_state(cls, params: ValueParams) -> ValueState:
        return ValueState(
            metrics=await cls._build_value_metrics_state(params.metrics),
            stop_condition=await cls._build_value_stop_condition_state(
                params.stop_condition
            ),
        )

    @classmethod
    async def _build_bootstrap(cls, params: Dict[str, Any]) -> Bootstrap:
        return await cls._build_generic(Bootstrap, **params)

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            policy=await self._build_policy_state(params.policy),
            value=await self._build_value_state(params.value),
            bootstrap=await self._build_bootstrap(params.bootstrap),
        )

    @classmethod
    async def _save_policy_stop_condition_state(
        cls, state: PolicyStopConditionState, directory: Path
    ) -> None:
        if isinstance(state.condition, Configurable):
            await state.condition.save(directory / "condition")

        state_dict = {
            "type": state.condition.category,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_policy_state(
        cls, state: PolicyState, directory: Path
    ) -> None:
        await cls._save_policy_stop_condition_state(
            state.stop_condition, directory / "stop_condition"
        )

    @classmethod
    async def _save_value_stop_condition_state(
        cls, state: ValueStopConditionState, directory: Path
    ) -> None:
        if isinstance(state.condition, Configurable):
            await state.condition.save(directory / "condition")

        state_dict = {
            "type": state.condition.category,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_value_state(
        cls, state: ValueState, directory: Path
    ) -> None:
        await cls._save_value_stop_condition_state(
            state.stop_condition, directory / "stop_condition"
        )

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_policy_state(state.policy, directory / "policy")
        await cls._save_value_state(state.value, directory / "value")
        if isinstance(state.bootstrap, Configurable):
            await state.bootstrap.save(directory / "bootstrap")

    @classmethod
    async def _load_policy_stop_condition(
        cls,
        state_dict: Dict[str, Any],
        params: PolicyStopConditionParams,
        directory: Path,
    ) -> PolicyStopCondition:
        category = state_dict.get("type", params.type)
        return await cls._load_generic(
            directory,
            PolicyStopCondition,
            category=category,
            **state_dict.get("params", params.params),
            default=partial(cls._build_policy_stop_condition, params),
        )

    @classmethod
    async def _load_policy_stop_condition_state(
        cls, params: PolicyStopConditionParams, directory: Path
    ) -> PolicyStopConditionState:
        state_dict = await cls._load_state_dict(directory)
        return PolicyStopConditionState(
            condition=await cls._load_policy_stop_condition(
                state_dict, params, directory / "condition"
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_policy_state(
        cls, params: PolicyParams, directory: Path
    ) -> PolicyState:
        return PolicyState(
            metrics=await cls._build_policy_metrics_state(params.metrics),
            stop_condition=await cls._load_policy_stop_condition_state(
                params.stop_condition, directory / "stop_condition"
            ),
        )

    @classmethod
    async def _load_value_stop_condition(
        cls,
        state_dict: Dict[str, Any],
        params: ValueStopConditionParams,
        directory: Path,
    ) -> ValueStopCondition:
        category = state_dict.get("type", params.type)
        return await cls._load_generic(
            directory,
            ValueStopCondition,
            category=category,
            **state_dict.get("params", params.params),
            default=partial(cls._build_value_stop_condition, params),
        )

    @classmethod
    async def _load_value_stop_condition_state(
        cls, params: ValueStopConditionParams, directory: Path
    ) -> ValueStopConditionState:
        state_dict = await cls._load_state_dict(directory)
        return ValueStopConditionState(
            condition=await cls._load_value_stop_condition(
                state_dict, params, directory / "condition"
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_value_state(
        cls, params: ValueParams, directory: Path
    ) -> ValueState:
        return ValueState(
            metrics=await cls._build_value_metrics_state(params.metrics),
            stop_condition=await cls._load_value_stop_condition_state(
                params.stop_condition, directory / "stop_condition"
            ),
        )

    @classmethod
    async def _load_bootstrap(
        cls, params: Dict[str, Any], directory: Path
    ) -> Bootstrap:
        return await cls._load_generic(
            directory,
            Bootstrap,
            **params,
            default=partial(cls._build_bootstrap, params),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)

        return State(
            policy=await self._load_policy_state(
                params.policy, directory / "policy"
            ),
            value=await self._load_value_state(
                params.value, directory / "value"
            ),
            bootstrap=await self._load_bootstrap(
                params.bootstrap, directory / "bootstrap"
            ),
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            await state.policy.metrics.episode_score.cleanup()
            await state.policy.metrics.episode_reward.cleanup()
            await state.policy.metrics.bootstrap_episode_reward.cleanup()
            await state.policy.metrics.base_bootstrap_iteration_loss.cleanup()
            await state.policy.metrics.base_bootstrap_episode_loss.cleanup()
            await state.policy.metrics.combined_bootstrap_iteration_loss.cleanup()
            await state.policy.metrics.combined_bootstrap_episode_loss.cleanup()
            await state.value.metrics.base_iteration_loss.cleanup()
            await state.value.metrics.base_episode_loss.cleanup()
            if isinstance(state.policy.stop_condition.condition, Configurable):
                await state.policy.stop_condition.condition.cleanup()
            if isinstance(state.value.stop_condition.condition, Configurable):
                await state.value.stop_condition.condition.cleanup()
            if isinstance(state.bootstrap, Configurable):
                await state.bootstrap.cleanup()
        return await super().cleanup()


class BootstrappingEarlyStoppingPolicyOptimizationMethod(
    BootstrappingEarlyStoppingPolicyOptimizationMethodBase
):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            PolicyStopConditionParameter,
            ValueStopConditionParameter,
            BootstrapParameter,
        }

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            return [
                state.policy.metrics.episode_score,
                state.policy.metrics.episode_reward,
                state.policy.metrics.bootstrap_episode_reward,
                state.policy.metrics.base_bootstrap_iteration_loss,
                state.policy.metrics.base_bootstrap_episode_loss,
                state.policy.metrics.combined_bootstrap_iteration_loss,
                state.policy.metrics.combined_bootstrap_episode_loss,
                state.value.metrics.base_iteration_loss,
                state.value.metrics.base_episode_loss,
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
        logprobs: PackedSequence,
        baseline_logprobs: PackedSequence,
        regularizations: List[PolicyRegularization],
    ) -> Tuple[Tensor, Dict[Hashable, MetricReport]]:
        combined_loss = base_losses.mean()
        reports = {}

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
                    direct=state.policy.metrics.base_bootstrap_iteration_loss,
                    aggregated=state.policy.metrics.base_bootstrap_episode_loss,
                ),
            )

    async def _get_combined_policy_report(
        self, losses: Tensor
    ) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    direct=state.policy.metrics.combined_bootstrap_iteration_loss,
                    aggregated=state.policy.metrics.combined_bootstrap_episode_loss,
                ),
            )

    async def _get_base_value_report(self, losses: Tensor) -> MetricReport:
        async with self.state.read_lock() as state:
            return MetricReport(
                values=losses.detach(),
                metrics=Metrics(
                    direct=state.value.metrics.base_iteration_loss,
                    aggregated=state.value.metrics.base_episode_loss,
                ),
            )

    @staticmethod
    async def _process_iteration_reports(
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
    async def _process_episode_reports(
        reports: Dict[Hashable, List[MetricReport]],
        epoch: int,
    ) -> None:
        for key, reports in reports.items():
            first = reports[0]
            if first.metrics.aggregated is not None:
                await first.metrics.aggregated.report(
                    epoch - 1,
                    torch.stack([r.values.mean() for r in reports])
                    .mean()
                    .item(),
                )

    async def _fit_value_batch(
        self, values: PackedSequence, returns: PackedSequence
    ) -> Dict[Hashable, MetricReport]:
        base_losses = MSELoss(reduction="none")(values.data, returns.data)
        base_loss = base_losses.mean()

        await background(base_loss.backward)

        return {"base": await self._get_base_value_report(base_loss)}

    async def _fit_value(
        self,
        value: TrainingControls[SequentialModel],
        sequences: PackedSequence,
        rewards: PackedSequence,
        gae: GeneralizedAdvantageEstimator,
        all_reports: Dict[Hashable, List[MetricReport]],
    ) -> Dict[Hashable, List[MetricReport]]:
        async with value.model.lock:
            values = await batched_forward(
                value.model.model, sequences, value.model.batch_size
            )

        advantages = await gae.calculate(rewards, values)
        returns = squash_packed(values, partial(torch.add, advantages.data))

        reports = await self._fit_value_batch(values, returns)
        step = await value.step()
        all_reports = await self._process_iteration_reports(
            reports, all_reports, step - 1
        )

        return all_reports

    async def _fit_value_loop(
        self,
        value: TrainingControls[SequentialModel],
        sequences: PackedSequence,
        rewards: PackedSequence,
        gae: GeneralizedAdvantageEstimator,
    ) -> Dict[Hashable, List[MetricReport]]:
        reports = {"base": []}
        iteration = 0

        while True:
            async with self.state.read_lock() as state:
                stop_condition = state.value.stop_condition.condition

            losses = [report.values for report in reports["base"]]
            if await stop_condition.should_stop(losses, iteration):
                break

            reports = await self._fit_value(
                value, sequences, rewards, gae, reports
            )
            iteration += 1

        return reports

    @staticmethod
    async def _calculate_base_policy_losses(
        new_gathered_logprobs: PackedSequence,
        old_gathered_logprobs: PackedSequence,
        advantages: PackedSequence,
    ) -> Tensor:
        ratios = (
            new_gathered_logprobs.data.detach() - old_gathered_logprobs.data
        )
        ratios = ratios.exp()
        logprobs = new_gathered_logprobs.data
        advantages = advantages.data
        return -(ratios * logprobs * advantages)

    async def _fit_policy(
        self,
        old_gathered_logprobs: PackedSequence,
        new_gathered_logprobs: PackedSequence,
        new_logprobs: PackedSequence,
        baseline_logprobs: PackedSequence,
        advantages: PackedSequence,
        regularizations: List[PolicyRegularization],
    ) -> Dict[Hashable, MetricReport]:
        base_losses = await self._calculate_base_policy_losses(
            new_gathered_logprobs, old_gathered_logprobs, advantages
        )
        base_loss = base_losses.mean()
        final_loss = base_loss
        reports = {"base": await self._get_base_policy_report(base_loss)}

        if regularizations:
            final_loss, regularization_reports = await self._regularize_policy(
                base_losses,
                new_logprobs,
                baseline_logprobs,
                regularizations,
            )
            reports = {**reports, **regularization_reports}

        reports["combined"] = await self._get_combined_policy_report(
            final_loss
        )
        await background(final_loss.backward)
        return reports

    async def _fit_policy_loop(
        self,
        policy: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        sequences: List[Tuple[Tensor, Tensor]],
        advantages: PackedSequence,
        regularizations: List[PolicyRegularization],
    ) -> Dict[Hashable, List[MetricReport]]:
        full_sequences = await self._get_full_sequences(sequences)

        async with baseline.lock:
            with freeze(baseline.model) as frozen_baseline:
                baseline_logprobs = await batched_forward(
                    frozen_baseline, full_sequences, baseline.batch_size
                )
                baseline_logprobs = squash_packed(
                    baseline_logprobs, lambda x: x.detach()
                )

        async with policy.model.lock:
            with freeze(policy.model.model) as frozen_policy:
                old_logprobs = await batched_forward(
                    frozen_policy, full_sequences, policy.model.batch_size
                )
                old_logprobs = squash_packed(
                    old_logprobs, lambda x: x.detach()
                )

        old_gathered_logprobs = await gather_logprobs(old_logprobs, sequences)

        all_reports = {}
        iteration = 0

        while True:
            async with policy.model.lock:
                new_logprobs = await batched_forward(
                    policy.model.model, full_sequences, policy.model.batch_size
                )
            new_gathered_logprobs = await gather_logprobs(
                new_logprobs, sequences
            )

            async with self.state.read_lock() as state:
                stop_condition = state.policy.stop_condition.condition

            if await stop_condition.should_stop(
                old_logprobs,
                old_gathered_logprobs,
                new_logprobs,
                new_gathered_logprobs,
                iteration,
            ):
                break

            reports = await self._fit_policy(
                old_gathered_logprobs,
                new_gathered_logprobs,
                new_logprobs,
                baseline_logprobs,
                advantages,
                regularizations,
            )

            step = await policy.step()

            all_reports = await self._process_iteration_reports(
                reports, all_reports, step - 1
            )

            iteration += 1

        return all_reports

    async def _fit_bootstrap_loop(
        self,
        policy: TrainingControls[SequentialModel],
        value: ModelInfo[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        regularizations: List[PolicyRegularization],
    ) -> None:
        iteration = 0
        all_rewards = []

        while True:
            async with self.state.read_lock() as state:
                bootstrap = state.bootstrap

            if await bootstrap.should_stop(all_rewards, iteration):
                break

            sequences = await bootstrap.generate(policy.model)
            full_sequences = pack_list(
                [
                    torch.vstack((context, response))
                    for context, response in sequences
                ]
            )

            async with value.lock:
                with freeze(value.model) as frozen_value:
                    values = await batched_forward(
                        frozen_value, full_sequences, value.batch_size
                    )
                    values = squash_packed(values, lambda x: x.detach())

            advantages = [
                (value - value[0])[len(context) :]
                for (context, _), value in zip(
                    sequences, unpack_to_list(values)
                )
            ]
            advantages = pack_list(advantages)
            rewards = torch.vstack(
                [value[-1] for value in unpack_to_list(values)]
            )

            reports = await self._fit_policy_loop(
                policy,
                baseline,
                sequences,
                advantages,
                regularizations,
            )

            episode = await policy.increment_episode()

            await self._process_episode_reports(reports, episode - 1)

            async with self.state.write_lock() as state:
                await state.policy.metrics.bootstrap_episode_reward.report(
                    episode - 1, rewards.mean().item()
                )

            iteration += 1
            all_rewards.append(rewards.mean().item())

    @staticmethod
    async def _unpack_data(
        data: Iterable[Tuple[Tensor, Tensor, Tensor, Tensor]]
    ) -> Tuple[
        List[Tuple[Tensor, Tensor]],
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
    ]:
        data = [x for x in data]
        if not data:
            return [], [], [], [], []

        contexts, responses, scores, rewards = zip(*data)

        sequences = [
            (context, response)
            for context, response in zip(contexts, responses)
        ]
        scores = list(scores)
        rewards = list(rewards)

        full_sequences = [
            torch.vstack((context, response))
            for context, response in sequences
        ]

        timestep_rewards = [
            torch.vstack(
                (torch.zeros(len(context) + len(response) - 1, 1), reward)
            )
            for (context, response), reward in zip(sequences, rewards)
        ]

        return sequences, full_sequences, scores, rewards, timestep_rewards

    async def fit(
        self,
        policy: TrainingControls[SequentialModel],
        value: TrainingControls[SequentialModel],
        baseline: ModelInfo[SequentialModel],
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        cache: Collection[Tuple[Tensor, Tensor, Tensor, Tensor]],
        gae: GeneralizedAdvantageEstimator,
        regularizations: List[PolicyRegularization],
    ) -> None:
        data = [x async for x in data]
        if not data:
            return

        (
            sequences,
            full_sequences,
            scores,
            rewards,
            timestep_rewards,
        ) = await self._unpack_data(data)

        (
            _,
            cached_full_sequences,
            _,
            _,
            cached_timestep_rewards,
        ) = await self._unpack_data(cache)

        reports = await self._fit_value_loop(
            value,
            pack_list(cached_full_sequences + full_sequences),
            pack_list(cached_timestep_rewards + timestep_rewards),
            gae,
        )

        episode = await value.increment_episode()
        await self._process_episode_reports(reports, episode - 1)

        async with self.state.write_lock() as state:
            await state.policy.metrics.episode_score.report(
                episode - 1, torch.stack(scores).mean().item()
            )
            await state.policy.metrics.episode_reward.report(
                episode - 1, torch.stack(rewards).mean().item()
            )

        await self._fit_bootstrap_loop(
            policy,
            value.model,
            baseline,
            regularizations,
        )
