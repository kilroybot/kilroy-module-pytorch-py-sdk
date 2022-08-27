from abc import ABC
from asyncio import Queue, Task
from dataclasses import dataclass
from typing import Any, AsyncIterable, Coroutine, Dict, List, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
import torch
from aiostream import stream
from aiostream.aiter_utils import aiter, anext
from kilroy_module_server_py_sdk import (
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

from kilroy_module_pytorch_py_sdk import Generator
from kilroy_module_pytorch_py_sdk.codec import Codec
from kilroy_module_pytorch_py_sdk.models import LanguageModel, RewardModel
from kilroy_module_pytorch_py_sdk.optimizers import Optimizer
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
class State:
    language_model: LanguageModel
    reward_model: RewardModel
    language_model_tokenizer: Tokenizer
    reward_model_tokenizer: Tokenizer
    language_model_optimizer: Optimizer
    language_model_optimizers_params: Dict[str, Dict[str, Any]]
    reward_model_optimizer: Optimizer
    reward_model_optimizers_params: Dict[str, Dict[str, Any]]
    frontend_generator: Generator
    backend_generator: Generator
    codec: Codec
    results_cache: Dict[UUID, Tuple[Tensor, Tensor]]
    batch_size: int
    sample_size: int
    epoch: int
    supervised_loss_metric: SupervisedLossMetric
    reinforced_score_metric: ReinforcedScoreMetric
    reward_model_loss_metric: RewardModelLossMetric
    reward_model_score_metric: RewardModelScoreMetric
    epoch_supervised_losses: List[float]
    epoch_reinforced_scores: List[float]
    epoch_reward_model_losses: List[float]
    epoch_reward_model_scores: List[float]
    coroutine_queue: Queue[Coroutine]
    worker_task: Task


class LanguageModelOptimizerParameter(
    CategorizableBasedParameter[State, Optimizer]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "params": state.language_model.parameters(),
            **state.language_model_optimizers_params.get(category, {}),
        }


class RewardModelOptimizerParameter(
    CategorizableBasedParameter[State, Optimizer]
):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "params": state.reward_model.parameters(),
            **state.reward_model_optimizers_params.get(category, {}),
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
            RewardModelOptimizerParameter(),
            FrontendGeneratorParameter(),
            BackendGeneratorParameter(),
            CodecParameter(),
            BatchSizeParameter(),
            SampleSizeParameter(),
        }

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return {
                state.supervised_loss_metric,
                state.reinforced_score_metric,
                state.reward_model_loss_metric,
                state.reward_model_score_metric,
            }

    async def generate(
        self, n: int, dry: bool
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        async with self.state.read_lock() as state:
            generated = state.frontend_generator.generate(
                state.language_model, state.language_model_tokenizer, n
            )

        async for result in generated:
            sequences = unpack_to_list(result.sequences)
            for sequence, logprob in zip(sequences, result.logprobs):
                post_id = uuid4()
                async with self.state.read_lock() as state:
                    post = await state.codec.encode(
                        state.language_model_tokenizer, sequence
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
                        state.language_model,
                        sequences,
                    )
                    state.epoch_supervised_losses.append(loss)
                    loss = await background(
                        self._fit_reward_model_batch,
                        state.reward_model,
                        sequences,
                        scores,
                    )
                    state.epoch_reward_model_losses.append(loss)

    async def fit_posts(
        self, posts: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        async def decoded():
            async for post, score in posts:
                # noinspection PyShadowingNames
                async with self.state.read_lock() as state:
                    post = await state.codec.decode(
                        state.language_model_tokenizer, post
                    )
                    score = torch.tensor(score, dtype=torch.float)
                    yield post, score

        await self._fit_supervised(decoded())

    async def _fit_with_reward_model(self) -> None:
        async with self.state.read_lock() as state:
            generated = state.backend_generator.generate(
                state.language_model,
                state.language_model_tokenizer,
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
                    state.language_model_tokenizer,
                    state.reward_model_tokenizer,
                )
                logprobs = batch.logprobs
                score = await background(
                    self._fit_with_reward_model_batch,
                    state.reward_model,
                    sequences,
                    logprobs,
                )
                state.epoch_reward_model_scores.append(score)

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
                        state.reward_model,
                        sequences,
                        scores,
                    )
                    state.epoch_reward_model_losses.append(loss)
                    state.epoch_reinforced_scores.append(scores.mean().item())

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

    async def step(self) -> None:
        async with self.state.write_lock() as state:
            await state.language_model_optimizer.step()
            await state.reward_model_optimizer.step()
            if state.epoch_supervised_losses:
                await state.supervised_loss_metric.report(
                    {
                        "epoch": state.epoch,
                        "loss": np.mean(state.epoch_supervised_losses),
                    }
                )
            if state.epoch_reinforced_scores:
                await state.reinforced_score_metric.report(
                    {
                        "epoch": state.epoch,
                        "score": np.mean(state.epoch_reinforced_scores),
                    }
                )
            if state.epoch_reward_model_losses:
                await state.reward_model_loss_metric.report(
                    {
                        "epoch": state.epoch,
                        "loss": np.mean(state.epoch_reward_model_losses),
                    }
                )
            if state.epoch_reward_model_scores:
                await state.reward_model_score_metric.report(
                    {
                        "epoch": state.epoch,
                        "score": np.mean(state.epoch_reward_model_scores),
                    }
                )
            state.epoch_supervised_losses = []
            state.epoch_reinforced_scores = []
            state.epoch_reward_model_losses = []
            state.epoch_reward_model_scores = []
            state.epoch += 1
