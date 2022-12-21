import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import AsyncIterable, Tuple, Collection, Set, Type, Dict, Any, List

import torch
from torch import Tensor

from kilroy_module_py_shared import TextOnlyPost, TextData
from kilroy_module_pytorch_py_sdk.generator.generator import Generator
from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_pytorch_py_sdk.module.parameters import (
    TrainerParameter,
    GeneratorParameter,
)
from kilroy_module_pytorch_py_sdk.module.params import (
    Params,
    TrainerParams,
    ModelsParams,
)
from kilroy_module_pytorch_py_sdk.module.state import State, TrainerState
from kilroy_module_pytorch_py_sdk.trainers import Trainer
from kilroy_module_server_py_sdk import Metric, Module
from kilroy_server_py_utils import (
    classproperty,
    Parameter,
    JSONSchema,
    Savable,
    Configurable,
    background,
)


class PytorchModuleBase(Module[State], ABC):
    @abstractmethod
    async def _build_models_registry(
        self, params: ModelsParams
    ) -> ModelsRegistry:
        pass

    @classmethod
    async def _build_trainer(
        cls, models: ModelsRegistry, params: TrainerParams
    ) -> Trainer:
        category = params.type
        return await cls._build_generic(
            Trainer,
            category=category,
            models=models,
            **params.params.get(category, {}),
        )

    @classmethod
    async def _build_trainer_state(
        cls, models: ModelsRegistry, params: TrainerParams
    ) -> TrainerState:
        return TrainerState(
            trainer=await cls._build_trainer(models, params),
            params=params.params,
        )

    @classmethod
    async def _build_generator(cls, **kwargs) -> Generator:
        return await cls._build_generic(Generator, **kwargs)

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        models = await self._build_models_registry(params.models)

        return State(
            models=models,
            trainer=await self._build_trainer_state(models, params.trainer),
            generator=await self._build_generator(**params.generator),
        )

    @staticmethod
    async def _save_models(models: ModelsRegistry) -> None:
        await models.policy.save()

    @classmethod
    async def _save_trainer(cls, trainer: Trainer, directory: Path) -> None:
        if isinstance(trainer, Savable):
            await trainer.save(directory)

    @classmethod
    async def _save_trainer_state(
        cls, state: TrainerState, directory: Path
    ) -> None:
        await cls._save_trainer(state.trainer, directory / "trainer")
        state_dict = {
            "category": state.trainer.category,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_generator(
        cls, generator: Generator, directory: Path
    ) -> None:
        if isinstance(generator, Savable):
            await generator.save(directory)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_models(state.models)
        await cls._save_trainer_state(state.trainer, directory / "trainer")
        await cls._save_generator(state.generator, directory / "generator")

    @classmethod
    async def _load_trainer(
        cls,
        models: ModelsRegistry,
        params: TrainerParams,
        state_dict: Dict[str, Any],
        directory: Path,
    ) -> Trainer:
        category = state_dict.get("category", params.type)
        return await cls._load_generic(
            directory,
            Trainer,
            category=category,
            models=models,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_trainer, models, params),
        )

    @classmethod
    async def _load_trainer_state(
        cls,
        models: ModelsRegistry,
        params: TrainerParams,
        directory: Path,
    ) -> TrainerState:
        state_dict = await cls._load_state_dict(directory)
        return TrainerState(
            trainer=await cls._load_trainer(
                models, params, state_dict, directory / "trainer"
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_generator(cls, directory: Path, **kwargs) -> Generator:
        return await cls._load_generic(
            directory,
            Generator,
            **kwargs,
            default=partial(cls._build_generator, **kwargs),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        models = await self._build_models_registry(params.models)

        return State(
            models=models,
            trainer=await self._load_trainer_state(
                models, params.trainer, directory / "trainer"
            ),
            generator=await self._load_generator(
                directory / "generator", **params.generator
            ),
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        async with self.state.write_lock() as state:
            if isinstance(state.trainer.trainer, Configurable):
                await state.trainer.trainer.cleanup()
            await state.generator.cleanup()


class PytorchModule(PytorchModuleBase, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    # noinspection PyMethodParameters
    @classproperty
    def post_schema(cls) -> JSONSchema:
        return JSONSchema(**TextOnlyPost.schema())

    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            TrainerParameter,
            GeneratorParameter,
        }

    async def init(self) -> None:
        await super().init()
        async with self.state.write_lock() as state:
            await state.models.policy.acquire()

    async def cleanup(self) -> None:
        await super().cleanup()
        async with self.state.write_lock() as state:
            await state.models.policy.release()

    async def reset_self(self) -> None:
        async with self.state.write_lock() as state:
            await state.models.policy.reset()
            await state.models.value.reset()
        await super().reset_self()

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            return await state.trainer.trainer.get_metrics()

    async def _encode(
        self, context: List[int], response: List[int]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        async with self.state.read_lock() as state:
            policy = await state.models.policy.get()

        indices = context + response
        text = await background(policy.tokenizer.decode, indices)
        post = TextOnlyPost(text=TextData(content=text))

        content = json.loads(post.json())
        metadata = {"context": context, "response": response}
        return content, metadata

    async def _decode_supervised(
        self, content: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        post = TextOnlyPost(**content)

        async with self.state.read_lock() as state:
            policy = await state.models.policy.get()

        indices = await background(policy.tokenizer.encode, post.text.content)
        indices = torch.tensor(indices).long().view(-1, 1)
        return indices[0:1], indices[1:]

    async def _decode_reinforced(
        self, content: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        try:
            context = torch.tensor(metadata["context"]).long().view(-1, 1)
            response = torch.tensor(metadata["response"]).long().view(-1, 1)
            return context, response
        except Exception as e:
            self.logger.warning(
                f"Failed to decode metadata: {metadata}. Using tokenizer instead.",
                exc_info=e,
            )
            return await self._decode_supervised(content)

    async def generate(
        self, n: int
    ) -> AsyncIterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        async with self.state.read_lock() as state:
            policy = await state.models.policy.get()
            generator = state.generator

        results = await generator.generate(policy, n)

        for context, response in results:
            yield await self._encode(context, response)

    async def _fit_supervised(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        async with self.state.read_lock() as state:
            trainer = state.trainer.trainer

        await trainer.fit_supervised(data)

    async def fit_supervised(
        self, data: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        async def __prepare(
            _data: AsyncIterable[Tuple[Dict[str, Any], float]]
        ) -> AsyncIterable[Tuple[Tensor, Tensor, Tensor]]:
            async for post, score in _data:
                try:
                    context, response = await self._decode_supervised(post)
                    score = torch.tensor([score]).float()
                    yield context, response, score
                except Exception as e:
                    self.logger.warning(
                        f"Failed to decode post: {json.dumps(post)}. "
                        f"Skipping...",
                        exc_info=e,
                    )
                    continue

        await self._fit_supervised(__prepare(data))

    async def _fit_reinforced(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        async with self.state.read_lock() as state:
            trainer = state.trainer.trainer

        await trainer.fit_reinforced(data)

    async def fit_reinforced(
        self, data: AsyncIterable[Tuple[Dict[str, Any], Dict[str, Any], float]]
    ) -> None:
        async def __prepare(
            _data: AsyncIterable[Tuple[Dict[str, Any], Dict[str, Any], float]]
        ) -> AsyncIterable[Tuple[Tensor, Tensor, Tensor]]:
            async for content, metadata, score in _data:
                try:
                    context, response = await self._decode_reinforced(
                        content, metadata
                    )
                    score = torch.tensor([score]).float()
                    yield context, response, score
                except Exception as e:
                    self.logger.warning(
                        f"Failed to decode post: {json.dumps(content)}. "
                        f"Skipping...",
                        exc_info=e,
                    )
                    continue

        await self._fit_reinforced(__prepare(data))
