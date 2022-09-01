from abc import ABC
from asyncio import Queue, Task
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)
from uuid import UUID, uuid4

import numpy as np
import torch
from aiostream import stream
from aiostream.aiter_utils import aiter, anext
from kilroy_module_server_py_sdk import (
    CategorizableBasedOptionalParameter,
    CategorizableBasedParameter,
    JSONSchema,
    Metric,
    Module,
    NestedParameter,
    Parameter,
    TextOnlyPost,
    background,
    classproperty,
)
from torch import Tensor
from torch.nn import MSELoss, NLLLoss
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.codec import Codec
from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.models import LanguageModel, RewardModel
from kilroy_module_pytorch_py_sdk.optimizers import Optimizer
from kilroy_module_pytorch_py_sdk.schedulers.base import Scheduler
from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer
from kilroy_module_pytorch_py_sdk.utils import (
    freeze,
    pack_list,
    truncate_first_element,
    truncate_last_element,
    unpack_to_list,
)


class SupervisedLossMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "supervisedLoss"

    @classproperty
    def label(cls) -> str:
        return "Supervised Loss"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "loss"}},
        }


class ReinforcedScoreMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "reinforcedScore"

    @classproperty
    def label(cls) -> str:
        return "Reinforced Score"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "score"}},
        }


class RewardModelLossMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "rewardModelLoss"

    @classproperty
    def label(cls) -> str:
        return "Reward Model Loss"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "loss"}},
        }


class RewardModelScoreMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "rewardModelScore"

    @classproperty
    def label(cls) -> str:
        return "Reward Model Score"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "score"}},
        }


@dataclass
class LanguageModelState:
    model: LanguageModel
    tokenizer: Tokenizer
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    scheduler: Optional[Scheduler]
    schedulers_params: Dict[str, Dict[str, Any]]


@dataclass
class RewardModelState:
    model: RewardModel
    tokenizer: Tokenizer
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    scheduler: Optional[Scheduler]
    schedulers_params: Dict[str, Dict[str, Any]]


@dataclass
class MetricsState:
    supervised_loss_metric: SupervisedLossMetric
    reinforced_score_metric: ReinforcedScoreMetric
    reward_model_loss_metric: RewardModelLossMetric
    reward_model_score_metric: RewardModelScoreMetric


@dataclass
class ReportsState:
    epoch_supervised_losses: List[float]
    epoch_reinforced_scores: List[float]
    epoch_reward_model_losses: List[float]
    epoch_reward_model_scores: List[float]


@dataclass
class State:
    language_model: LanguageModelState
    reward_model: RewardModelState
    frontend_generator: Generator
    backend_generator: Generator
    codec: Codec
    results_cache: Dict[UUID, Tuple[Tensor, Tensor]]
    batch_size: int
    sample_size: int
    epoch: int
    metrics: MetricsState
    reports: ReportsState
    coroutine_queue: Queue[Coroutine]
    worker_task: Task


class LanguageModelOptimizerParameter(
    CategorizableBasedParameter[State, Optimizer]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "parameters": state.language_model.model.parameters(),
            **state.language_model.optimizers_params.get(category, {}),
        }

    async def _set_categorizable(self, state: State, value: Optimizer) -> None:
        await super()._set_categorizable(state, value)
        if state.language_model.scheduler is not None:
            optimizer = await value.get()
            await state.language_model.scheduler.change_optimizer(optimizer)


class LanguageModelSchedulerParameter(
    CategorizableBasedOptionalParameter[State, Scheduler]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "optimizer": await state.language_model.optimizer.get(),
            **state.language_model.schedulers_params.get(category, {}),
        }


class RewardModelOptimizerParameter(
    CategorizableBasedParameter[State, Optimizer]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "parameters": state.reward_model.model.parameters(),
            **state.reward_model.optimizers_params.get(category, {}),
        }

    async def _set_categorizable(self, state: State, value: Optimizer) -> None:
        await super()._set_categorizable(state, value)
        if state.reward_model.scheduler is not None:
            optimizer = await value.get()
            await state.reward_model.scheduler.change_optimizer(optimizer)


class RewardModelSchedulerParameter(
    CategorizableBasedOptionalParameter[State, Scheduler]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "optimizer": await state.reward_model.optimizer.get(),
            **state.reward_model.schedulers_params.get(category, {}),
        }


class FrontendGeneratorParameter(NestedParameter[State, Generator]):
    pass


class BackendGeneratorParameter(NestedParameter[State, Generator]):
    pass


class CodecParameter(NestedParameter[State, Codec]):
    pass


class BatchSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1}


class SampleSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1}


class RewardModelModule(Module[State], ABC):
    @classproperty
    def post_schema(cls) -> JSONSchema:
        return JSONSchema(**TextOnlyPost.schema())

    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            LanguageModelOptimizerParameter(),
            LanguageModelSchedulerParameter(),
            RewardModelOptimizerParameter(),
            RewardModelSchedulerParameter(),
            FrontendGeneratorParameter(),
            BackendGeneratorParameter(),
            CodecParameter(),
            BatchSizeParameter(),
            SampleSizeParameter(),
        }

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return {
                state.metrics.supervised_loss_metric,
                state.metrics.reinforced_score_metric,
                state.metrics.reward_model_loss_metric,
                state.metrics.reward_model_score_metric,
            }

    async def generate(
        self, n: int, dry: bool
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        async with self.state.read_lock() as state:
            generated = state.frontend_generator.generate(
                state.language_model.model, state.language_model.tokenizer, n
            )

        async for result in generated:
            sequences = unpack_to_list(result.sequences)
            for sequence, logprob in zip(sequences, result.logprobs):
                post_id = uuid4()
                async with self.state.read_lock() as state:
                    post = await state.codec.encode(
                        state.language_model.tokenizer, sequence
                    )
                if not dry:
                    async with self.state.write_lock() as state:
                        state.results_cache[post_id] = (sequence, logprob[0])
                yield post_id, post

    @staticmethod
    def _fit_language_model_batch(
        model: LanguageModel, sequences: PackedSequence
    ) -> float:
        batch = unpack_to_list(sequences)
        input = pack_list(truncate_last_element(batch))
        target = pack_list(truncate_first_element(batch))
        logprobs = model(input)
        loss = NLLLoss()(logprobs.data, target.data.flatten())
        loss.backward()
        return loss.item()

    @staticmethod
    def _fit_reward_model_batch(
        model: RewardModel, sequences: PackedSequence, scores: Tensor
    ) -> float:
        predicted = model(sequences)
        loss = MSELoss()(predicted, scores)
        loss.backward()
        return loss.item()

    @staticmethod
    def _fit_with_reward_model_batch(
        model: RewardModel, sequences: PackedSequence, logprobs: Tensor
    ) -> float:
        with freeze(model) as frozen:
            scores = frozen(sequences)
        loss = -(logprobs * scores).mean()
        loss.backward()
        return scores.mean().item()

    @staticmethod
    def _recode(
        sequences: PackedSequence, source: Tokenizer, target: Tokenizer
    ) -> PackedSequence:
        sequences = unpack_to_list(sequences)
        sequences = [sequence.flatten().tolist() for sequence in sequences]
        decoded = [source.decode(sequence) for sequence in sequences]
        encoded = [target.encode(sequence) for sequence in decoded]
        encoded = [torch.tensor(sequence).view(-1, 1) for sequence in encoded]
        return pack_list(encoded)

    async def _fit_supervised(
        self, data: AsyncIterable[Tuple[Tensor, Tensor]]
    ) -> None:
        async with self.state.read_lock() as state:
            batches = stream.chunks(data, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                async with self.state.write_lock() as state:
                    sequences = pack_list(sequence for sequence, _ in batch)
                    scores = torch.vstack([score for _, score in batch])
                    loss = await background(
                        self._fit_language_model_batch,
                        state.language_model.model,
                        sequences,
                    )
                    state.reports.epoch_supervised_losses.append(loss)
                    loss = await background(
                        self._fit_reward_model_batch,
                        state.reward_model.model,
                        sequences,
                        scores,
                    )
                    state.reports.epoch_reward_model_losses.append(loss)

    async def fit_posts(
        self, posts: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        async def decoded():
            async for post, score in posts:
                # noinspection PyShadowingNames
                async with self.state.read_lock() as state:
                    post = await state.codec.decode(
                        state.language_model.tokenizer, post
                    )
                    score = torch.tensor(score, dtype=torch.float)
                    yield post, score

        await self._fit_supervised(decoded())

    async def _fit_with_reward_model(self) -> None:
        async with self.state.read_lock() as state:
            generated = state.backend_generator.generate(
                state.language_model.model,
                state.language_model.tokenizer,
                state.sample_size,
            )

        generated = aiter(generated)

        while True:
            async with self.state.write_lock() as state:
                try:
                    batch = await anext(generated)
                except StopAsyncIteration:
                    break
                sequences = self._recode(
                    batch.sequences,
                    state.language_model.tokenizer,
                    state.reward_model.tokenizer,
                )
                logprobs = batch.logprobs
                score = await background(
                    self._fit_with_reward_model_batch,
                    state.reward_model.model,
                    sequences,
                    logprobs,
                )
                state.reports.epoch_reward_model_scores.append(score)

    async def _fit_reinforced(
        self,
        results: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        async with self.state.read_lock() as state:
            batches = stream.chunks(results, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                sequences = pack_list([sequence for sequence, _, _ in batch])
                scores = torch.vstack([score for _, _, score in batch])
                async with self.state.write_lock() as state:
                    loss = await background(
                        self._fit_reward_model_batch,
                        state.reward_model.model,
                        sequences,
                        scores,
                    )
                    state.reports.epoch_reward_model_losses.append(loss)
                    state.reports.epoch_reinforced_scores.append(
                        scores.mean().item()
                    )

        async with self.state.write_lock() as state:
            await state.coroutine_queue.put(self._fit_with_reward_model())

    async def fit_scores(self, scores: List[Tuple[UUID, float]]) -> None:
        async def get_results():
            for post_id, score in scores:
                # noinspection PyShadowingNames
                async with self.state.write_lock() as state:
                    sequence, logprob = state.results_cache.pop(post_id)
                yield sequence, logprob, torch.tensor(score)

        await self._fit_reinforced(get_results())

    @staticmethod
    async def _report_mean_from_epoch(
        metric: Metric, epoch: int, label: str, values: Iterable[float]
    ) -> None:
        values = list(values)
        if values:
            await metric.report({"epoch": epoch, label: np.mean(values)})

    @staticmethod
    async def _reset_reports(state: State) -> None:
        state.reports.epoch_supervised_losses = []
        state.reports.epoch_reinforced_scores = []
        state.reports.epoch_reward_model_losses = []
        state.reports.epoch_reward_model_scores = []

    async def step(self) -> None:
        async with self.state.write_lock() as state:
            await state.language_model.optimizer.step()
            if state.language_model.scheduler is not None:
                await state.language_model.scheduler.step()
            await state.reward_model.optimizer.step()
            if state.reward_model.scheduler is not None:
                await state.reward_model.scheduler.step()
            await self._report_mean_from_epoch(
                state.metrics.supervised_loss_metric,
                state.epoch,
                "loss",
                state.reports.epoch_supervised_losses,
            )
            await self._report_mean_from_epoch(
                state.metrics.reinforced_score_metric,
                state.epoch,
                "score",
                state.reports.epoch_reinforced_scores,
            )
            await self._report_mean_from_epoch(
                state.metrics.reward_model_loss_metric,
                state.epoch,
                "loss",
                state.reports.epoch_reward_model_losses,
            )
            await self._report_mean_from_epoch(
                state.metrics.reward_model_score_metric,
                state.epoch,
                "score",
                state.reports.epoch_reward_model_scores,
            )
            await self._reset_reports(state)
            state.epoch += 1
