from abc import ABC
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
import torch
from aiostream import stream
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
from torch.nn import NLLLoss

from kilroy_module_pytorch_py_sdk.codec import Codec
from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.models import LanguageModel
from kilroy_module_pytorch_py_sdk.optimizers import Optimizer
from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer
from kilroy_module_pytorch_py_sdk.utils import (
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


@dataclass
class State:
    model: LanguageModel
    tokenizer: Tokenizer
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    generator: Generator
    codec: Codec
    results_cache: Dict[UUID, Tuple[Tensor, Tensor]]
    batch_size: int
    epoch: int
    supervised_loss_metric: SupervisedLossMetric
    reinforced_score_metric: ReinforcedScoreMetric
    epoch_supervised_losses: List[float]
    epoch_reinforced_scores: List[float]


class OptimizerParameter(CategorizableBasedParameter[State, Optimizer]):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "params": state.model.parameters(),
            **state.optimizers_params.get(category, {}),
        }


class GeneratorParameter(NestedParameter[State, Generator]):
    pass


class CodecParameter(NestedParameter[State, Codec]):
    pass


class BatchSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1}


class BasicModule(Module[State], ABC):
    @classproperty
    def post_schema(cls) -> JSONSchema:
        return JSONSchema(**TextOnlyPost.schema())

    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            OptimizerParameter(),
            GeneratorParameter(),
            CodecParameter(),
            BatchSizeParameter(),
        }

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return {
                state.supervised_loss_metric,
                state.reinforced_score_metric,
            }

    async def generate(
        self, n: int, dry: bool
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        async with self.state.read_lock() as state:
            generated = state.generator.generate(
                state.model, state.tokenizer, n
            )

        async for result in generated:
            sequences = unpack_to_list(result.sequences)
            for sequence, logprob in zip(sequences, result.logprobs):
                post_id = uuid4()
                async with self.state.read_lock() as state:
                    post = await state.codec.encode(state.tokenizer, sequence)
                if not dry:
                    async with self.state.write_lock() as state:
                        state.results_cache[post_id] = (sequence, logprob[0])
                yield post_id, post

    async def _fit_supervised(self, data: AsyncIterable[Tensor]) -> None:
        # noinspection PyShadowingNames
        def fit(model, batch):
            input = pack_list(truncate_last_element(batch))
            target = pack_list(truncate_first_element(batch))
            logprobs = model(input)
            loss = NLLLoss()(logprobs.data, target.data.flatten())
            loss.backward()
            return loss.item()

        async with self.state.read_lock() as state:
            batches = stream.chunks(data, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                async with self.state.write_lock() as state:
                    loss = await background(fit, state.model, batch)
                    state.epoch_supervised_losses.append(loss)

    async def fit_posts(
        self, posts: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        async def decoded():
            async for post, _ in posts:
                # noinspection PyShadowingNames
                async with self.state.read_lock() as state:
                    yield await state.codec.decode(state.tokenizer, post)

        await self._fit_supervised(decoded())

    async def _fit_reinforced(
        self,
        results: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        results = list([result async for result in results])
        logprobs = torch.stack([logprob for _, logprob, _ in results])
        scores = torch.stack([score for _, _, score in results])

        def fit():
            loss = -(logprobs * scores).mean()
            loss.backward()
            return scores.mean().item()

        async with self.state.write_lock() as state:
            score = await background(fit)
            state.epoch_reinforced_scores.append(score)

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
            await state.optimizer.step()
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
            state.epoch_supervised_losses = []
            state.epoch_reinforced_scores = []
            state.epoch += 1
