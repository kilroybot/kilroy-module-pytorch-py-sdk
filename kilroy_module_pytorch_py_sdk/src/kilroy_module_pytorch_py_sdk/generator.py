from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.samplers import Sampler
from kilroy_module_pytorch_py_sdk.utils import (
    pack_list,
    unpack_to_padded,
)


class SequentialModel(Protocol):
    def __call__(self, x: PackedSequence) -> PackedSequence:
        ...


@dataclass
class GenerationResult:
    sequences: PackedSequence
    logprobs: Tensor


# noinspection PyMethodMayBeStatic
class SequenceGenerator:
    @dataclass
    class State:
        waiting_sequences: List[Tensor]
        current_sequences: List[Tensor]
        current_logprobs: List[Tensor]
        current_max_length: int
        finished_sequences: List[Tensor] = field(default_factory=list)
        finished_logprobs: List[Tensor] = field(default_factory=list)

    def _build_initial_state(self, contexts: List[List[int]]) -> State:
        context = [torch.tensor(context).view(-1, 1) for context in contexts]
        min_length = len(min(context, key=len))
        current, waiting = [], []

        for sequence in context:
            if len(sequence) == min_length:
                current.append(sequence)
            else:
                waiting.append(sequence)

        return self.State(
            waiting_sequences=waiting,
            current_sequences=current,
            current_logprobs=[torch.tensor(0) for _ in range(len(current))],
            current_max_length=min_length,
        )

    def _should_stop(self, state: State, max_length: int) -> bool:
        return (
            len(state.current_sequences) <= 0
            or state.current_max_length >= max_length
        )

    def _predict(
        self, model: SequentialModel, current_sequences: List[Tensor]
    ) -> Tensor:
        predictions, _ = unpack_to_padded(model(pack_list(current_sequences)))
        return predictions[:, -1]

    def _pick(
        self, sampler: Sampler, batched_logprobs: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        result = sampler.sample(batched_logprobs)
        return list(result.samples.flatten()), list(result.logprobs.flatten())

    def _get_finished_mask(
        self, next_values: List[Tensor], end_indices: List[int]
    ) -> List[bool]:
        return [value.item() in end_indices for value in next_values]

    def _update_state(
        self,
        state: State,
        next_values: List[Tensor],
        next_logprobs: List[Tensor],
        end_indices: List[int],
    ) -> State:
        sequences = [
            torch.cat((current, next.view(1, 1)))
            for current, next in zip(state.current_sequences, next_values)
        ]
        logprobs = [
            torch.add(current, next)
            for current, next in zip(state.current_logprobs, next_logprobs)
        ]

        finished_mask = self._get_finished_mask(next_values, end_indices)

        state.finished_sequences.extend(
            [
                sequence
                for sequence, finished in zip(sequences, finished_mask)
                if finished
            ]
        )
        state.finished_logprobs.extend(
            [
                logprob
                for logprob, finished in zip(logprobs, finished_mask)
                if finished
            ]
        )

        new_current_sequences = [
            sequence
            for sequence, finished in zip(sequences, finished_mask)
            if not finished
        ]
        new_current_logprobs = [
            logprobs
            for logprobs, finished in zip(logprobs, finished_mask)
            if not finished
        ]
        new_current_max_length = state.current_max_length + 1
        new_waiting_sequences = []

        for sequence in state.waiting_sequences:
            if len(sequence) == new_current_max_length:
                new_current_sequences.append(sequence)
                new_current_logprobs.append(torch.tensor(0))
            else:
                new_waiting_sequences.append(sequence)

        state.current_sequences = new_current_sequences
        state.current_logprobs = new_current_logprobs
        state.current_max_length = new_current_max_length
        state.waiting_sequences = new_waiting_sequences

        return state

    def _complete(self, state: State) -> Tuple[List[Tensor], List[Tensor]]:
        return (
            state.finished_sequences + state.current_sequences,
            state.finished_logprobs + state.current_logprobs,
        )

    def _prepare_output(
        self, sequences: List[Tensor], logprobs: List[Tensor]
    ) -> GenerationResult:
        ordered = sorted(
            zip(sequences, logprobs),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        sequences = pack_list([sequence for sequence, _ in ordered])
        logprobs = torch.vstack([logprob for _, logprob in ordered])
        return GenerationResult(sequences=sequences, logprobs=logprobs)

    def generate(
        self,
        model: SequentialModel,
        sampler: Sampler,
        contexts: List[List[int]],
        max_length: int,
        end_indices: List[int],
    ) -> GenerationResult:
        state = self._build_initial_state(contexts)
        while not self._should_stop(state, max_length):
            logprobs = self._predict(model, state.current_sequences)
            next_values, next_logprobs = self._pick(sampler, logprobs)
            state = self._update_state(
                state, next_values, next_logprobs, end_indices
            )
        sequences, logprobs = self._complete(state)
        return self._prepare_output(sequences, logprobs)
