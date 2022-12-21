import random
import re
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Type, Pattern, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

from kilroy_module_pytorch_py_sdk.generator.parameters import (
    ContextsParameter,
    RegexParameter,
    MaxLengthParameter,
)
from kilroy_module_pytorch_py_sdk.generator.params import Params
from kilroy_module_pytorch_py_sdk.generator.state import State
from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo
from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer
from kilroy_module_pytorch_py_sdk.utils import freeze
from kilroy_module_pytorch_py_sdk.utils import (
    unpack_to_padded,
    pack_list,
    batched_forward,
)
from kilroy_module_server_py_sdk import Configurable, Parameter, classproperty


@dataclass
class SequenceState:
    context: List[int]
    response: List[int]


@dataclass
class GenerationState:
    waiting_sequences: List[SequenceState]
    current_sequences: List[SequenceState]
    finished_sequences: List[SequenceState]
    current_max_length: int


class GeneratorBase(Configurable[State], ABC):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            ContextsParameter,
            RegexParameter,
            MaxLengthParameter,
        }

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            contexts=params.contexts,
            regex=re.compile(params.regex),
            max_length=params.max_length,
        )

    async def _save_state(self, state: State, directory: Path) -> None:
        state_dict = {
            "contexts": state.contexts,
            "regex": state.regex.pattern,
            "max_length": state.max_length,
        }
        await self._save_state_dict(state_dict, directory)

    async def _load_saved_state(self, directory: Path) -> State:
        state_dict = await self._load_state_dict(directory)
        params = Params(**self._kwargs)
        return State(
            contexts=state_dict.get("contexts", params.contexts),
            regex=re.compile(state_dict.get("regex", params.regex)),
            max_length=state_dict.get("max_length", params.max_length),
        )


class Generator(GeneratorBase):
    @staticmethod
    def _sample_contexts(
        contexts: List[str], tokenizer: Tokenizer, n: int
    ) -> Iterable[List[int]]:
        contexts = random.choices(contexts or [""], k=n)

        for context in contexts:
            encoded = tokenizer.encode(context)
            yield encoded[:-1]

    @staticmethod
    def _build_initial_generation_state(
        contexts: Iterable[Iterable[int]],
    ) -> GenerationState:
        contexts = [list(context) for context in contexts]
        min_length = len(min(contexts, key=len))
        current, waiting = [], []

        for context in contexts:
            sequence = SequenceState(context=context, response=[])
            if len(context) == min_length:
                current.append(sequence)
            else:
                waiting.append(sequence)

        return GenerationState(
            waiting_sequences=waiting,
            current_sequences=current,
            finished_sequences=[],
            current_max_length=min_length,
        )

    @staticmethod
    def _should_stop(state: GenerationState, max_length: int) -> bool:
        return (
            len(state.current_sequences) <= 0
            or state.current_max_length >= max_length + 1
        )

    @staticmethod
    async def _predict(
        model: ModelInfo[SequentialModel],
        current_sequences: List[SequenceState],
    ) -> Tensor:
        sequences = [
            torch.tensor(sequence.context + sequence.response).view(-1, 1)
            for sequence in current_sequences
        ]
        async with model.lock:
            with freeze(model.model) as frozen_model:
                predictions = await batched_forward(
                    frozen_model, pack_list(sequences), model.batch_size
                )
        predictions, _ = unpack_to_padded(predictions)
        return predictions[:, -1]

    @staticmethod
    async def _pick(batched_logprobs: Tensor) -> List[int]:
        dist = Categorical(logits=batched_logprobs, validate_args=False)
        samples = dist.sample((1,))
        return samples.permute(1, 0).flatten().tolist()

    @staticmethod
    def _get_finished_mask(
        next_values: List[int], end_value: int
    ) -> List[bool]:
        return [value == end_value for value in next_values]

    def _update_generation_state(
        self,
        state: GenerationState,
        next_values: List[int],
        tokenizer: Tokenizer,
    ) -> GenerationState:
        sequences = [
            SequenceState(
                context=current.context,
                response=current.response + [next],
            )
            for current, next in zip(state.current_sequences, next_values)
        ]

        finished_mask = self._get_finished_mask(
            next_values, tokenizer.end_token
        )

        state.finished_sequences.extend(
            [
                sequence
                for sequence, finished in zip(sequences, finished_mask)
                if finished
            ]
        )

        new_current_sequences = [
            sequence
            for sequence, finished in zip(sequences, finished_mask)
            if not finished
        ]
        new_current_max_length = state.current_max_length + 1
        new_waiting_sequences = []

        for sequence in state.waiting_sequences:
            if (
                len(sequence.context + sequence.response)
                == new_current_max_length
            ):
                new_current_sequences.append(sequence)
            else:
                new_waiting_sequences.append(sequence)

        state.current_sequences = new_current_sequences
        state.waiting_sequences = new_waiting_sequences
        state.current_max_length = new_current_max_length

        return state

    @staticmethod
    def _is_complete(sequence: SequenceState, end_value: int) -> bool:
        return (sequence.context + sequence.response)[-1] == end_value

    @staticmethod
    def _trim_incomplete(
        sequence: SequenceState,
        tokenizer: Tokenizer,
        regex: Pattern[str],
    ) -> SequenceState:
        for i in range(len(sequence.response) - 1, -1, -1):
            index = slice(0, i + 1)
            sentence = tokenizer.decode(sequence.response[index])
            if regex.fullmatch(sentence):
                return SequenceState(
                    context=sequence.context,
                    response=sequence.response[index],
                )
        for i in range(len(sequence.context) - 1, -1, -1):
            index = slice(0, i + 1)
            sentence = tokenizer.decode(sequence.context[index])
            if regex.fullmatch(sentence):
                return SequenceState(
                    context=sequence.context[index], response=[]
                )
        return sequence

    def _complete(
        self,
        state: GenerationState,
        tokenizer: Tokenizer,
        regex: Pattern[str],
    ) -> List[SequenceState]:
        in_sequences = state.finished_sequences + state.current_sequences
        out_sequences = []

        for sequence in in_sequences:
            if self._is_complete(sequence, tokenizer.end_token):
                out_sequences.append(sequence)
            else:
                new_sequence = self._trim_incomplete(
                    sequence, tokenizer, regex
                )
                out_sequences.append(new_sequence)
        return out_sequences

    @staticmethod
    def _prepare_output(
        sequences: List[SequenceState],
    ) -> List[Tuple[List[int], List[int]]]:
        return [
            (sequence.context, sequence.response) for sequence in sequences
        ]

    async def _generate(
        self,
        model: ModelInfo[SequentialModel],
        contexts: Iterable[Iterable[int]],
        max_length: int,
        regex: Pattern[str],
    ) -> List[Tuple[List[int], List[int]]]:
        state = self._build_initial_generation_state(contexts)
        while not self._should_stop(state, max_length):
            logprobs = await self._predict(model, state.current_sequences)
            next_values = await self._pick(logprobs)
            state = self._update_generation_state(
                state, next_values, model.tokenizer
            )
        sequences = self._complete(state, model.tokenizer, regex)
        return self._prepare_output(sequences)

    async def generate(
        self,
        model: ModelInfo[SequentialModel],
        n: int,
    ) -> List[Tuple[List[int], List[int]]]:
        async with self.state.read_lock() as state:
            contexts = self._sample_contexts(
                state.contexts, model.tokenizer, n
            )

            return await self._generate(
                model,
                contexts,
                state.max_length,
                state.regex,
            )
