import random
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, Iterable, List, Set

from kilroy_module_server_py_sdk import (
    CategorizableBasedParameter,
    Configurable,
    Parameter,
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
        return {"type": "array", "items": {"type": "string"}, "minItems": 1}


class MaxLengthParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1}


class BatchSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {"type": "integer", "minimum": 1}


class Generator(Configurable[State]):
    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            SamplerParameter(),
            ContextsParameter(),
            MaxLengthParameter(),
            BatchSizeParameter(),
        }

    async def build_default_state(self) -> State:
        params = Params(**self._kwargs)
        sampler_cls = Sampler.for_category(params.sampler_type)
        sampler_params = params.samplers_params.get(params.sampler_type, {})
        if issubclass(sampler_cls, Configurable):
            sampler = await sampler_cls.build(**sampler_params)
            await sampler.init()
        else:
            sampler = sampler_cls(**sampler_params)
        return State(
            sampler=sampler,
            samplers_params=params.samplers_params,
            contexts=params.contexts,
            max_length=params.max_length,
            batch_size=params.batch_size,
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
