import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Iterable, List, Set

from kilroy_module_server_py_sdk import (
    CategorizableBasedParameter,
    Configurable,
    Parameter,
    Savable,
    SerializableModel,
    classproperty,
)

from kilroy_module_pytorch_py_sdk.generator.utils import (
    GenerationResult,
    generate,
)
from kilroy_module_pytorch_py_sdk.models import LanguageModel
from kilroy_module_pytorch_py_sdk.samplers import Sampler
from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer


class Params(SerializableModel):
    sampler_type: str = "epsilonNucleus"
    samplers_params: Dict[str, Dict[str, Any]] = {}
    contexts: List[str] = [""]
    max_length: int
    batch_size: int


@dataclass
class State:
    sampler: Sampler
    samplers_params: Dict[str, Dict[str, Any]]
    contexts: List[str]
    max_length: int
    batch_size: int


class SamplerParameter(CategorizableBasedParameter[State, Sampler]):
    pass


class ContextsParameter(Parameter[State, List[str]]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "title": cls.pretty_name,
            "default": [" "],
        }


class MaxLengthParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1, "title": cls.pretty_name}

    @classproperty
    def pretty_name(cls) -> str:
        return "Maximum Length"


class BatchSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1, "title": cls.pretty_name}


class Generator(Configurable[State]):
    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            SamplerParameter(),
            ContextsParameter(),
            MaxLengthParameter(),
            BatchSizeParameter(),
        }

    async def _build_sampler(self, params: Params) -> Sampler:
        return await self._build_generic(
            Sampler,
            category=params.sampler_type,
            **params.samplers_params.get(params.sampler_type, {}),
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            sampler=await self._build_sampler(params),
            samplers_params=params.samplers_params,
            contexts=params.contexts,
            max_length=params.max_length,
            batch_size=params.batch_size,
        )

    async def _save_state(self, state: State, directory: Path) -> None:
        state_dict = {
            "sampler_type": state.sampler.category,
            "samplers_params": state.samplers_params,
            "contexts": state.contexts,
            "max_length": state.max_length,
            "batch_size": state.batch_size,
        }
        if isinstance(state.sampler, Savable):
            await state.sampler.save(directory / "sampler")
        with open(directory / "state.json", "w") as f:
            json.dump(state_dict, f)

    async def _load_saved_state(self, directory: Path) -> State:
        with open(directory / "state.json", "r") as f:
            state_dict = json.load(f)
        params = Params(**self._kwargs)
        return State(
            sampler=await self._load_generic(
                directory / "sampler",
                Sampler,
                category=state_dict["sampler_type"],
                default=partial(self._build_sampler, params),
                **params.samplers_params.get(params.sampler_type, {}),
            ),
            samplers_params=state_dict["samplers_params"],
            contexts=state_dict["contexts"],
            max_length=state_dict["max_length"],
            batch_size=state_dict["batch_size"],
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.sampler, Configurable):
                await state.sampler.cleanup()

    @staticmethod
    def _get_contexts(
        state: State, tokenizer: Tokenizer, n: int
    ) -> Iterable[List[int]]:
        contexts = random.choices(state.contexts, k=n)

        for context in contexts:
            encoded = tokenizer.encode(context)
            yield encoded[:-1]

    async def generate(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        n: int,
    ) -> AsyncIterable[GenerationResult]:
        async with self.state.read_lock() as state:
            while n > 0:
                batch_size = min(n, state.batch_size)
                n -= batch_size
                contexts = self._get_contexts(state, tokenizer, batch_size)

                yield await generate(
                    model,
                    state.sampler,
                    contexts,
                    state.max_length,
                    tokenizer.end_token,
                )
