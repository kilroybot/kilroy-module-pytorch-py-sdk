from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Optional, Pattern

import torch
from kilroy_module_server_py_sdk import background
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.models import LanguageModel
from kilroy_module_pytorch_py_sdk.samplers.base import Sampler
from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer
from kilroy_module_pytorch_py_sdk.utils import pack_list, unpack_to_padded


@dataclass
class GenerationResult:
    sequences: PackedSequence
    logprobs: Tensor


@dataclass
class GenerationState:
    waiting_sequences: List[Tensor]
    current_sequences: List[Tensor]
    current_logprobs: List[Tensor]
    current_max_length: int
    finished_sequences: List[Tensor] = field(default_factory=list)
    finished_logprobs: List[Tensor] = field(default_factory=list)


def _build_initial_state(contexts: Iterable[Iterable[int]]) -> GenerationState:
    context = [torch.tensor(context).view(-1, 1) for context in contexts]
    min_length = len(min(context, key=len))
    current, waiting = [], []

    for sequence in context:
        if len(sequence) == min_length:
            current.append(sequence)
        else:
            waiting.append(sequence)

    return GenerationState(
        waiting_sequences=waiting,
        current_sequences=current,
        current_logprobs=[torch.tensor([[0]]) for _ in range(len(current))],
        current_max_length=min_length,
    )


def _should_stop(state: GenerationState, max_length: int) -> bool:
    return (
        len(state.current_sequences) <= 0
        or state.current_max_length >= max_length
    )


def _predict(
    model: LanguageModel, current_sequences: Iterable[Tensor]
) -> Tensor:
    predictions, _ = unpack_to_padded(model(pack_list(current_sequences)))
    return predictions[:, -1]


async def _pick(
    sampler: Sampler, batched_logprobs: Tensor
) -> Tuple[List[Tensor], List[Tensor]]:
    result = await sampler.sample(batched_logprobs)
    return list(result.samples.flatten()), list(result.logprobs.flatten())


def _get_finished_mask(
    next_values: Iterable[Tensor], end_value: int
) -> List[bool]:
    return [value.item() == end_value for value in next_values]


def _update_state(
    state: GenerationState,
    next_values: Iterable[Tensor],
    next_logprobs: Iterable[Tensor],
    tokenizer: Tokenizer,
) -> GenerationState:
    sequences = [
        torch.cat((current, next.view(1, 1)))
        for current, next in zip(state.current_sequences, next_values)
    ]
    logprobs = [
        torch.cat((current, next.view(1, 1)))
        for current, next in zip(state.current_logprobs, next_logprobs)
    ]

    finished_mask = _get_finished_mask(next_values, tokenizer.end_token)

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
            new_current_logprobs.append(torch.tensor([[0]]))
        else:
            new_waiting_sequences.append(sequence)

    state.current_sequences = new_current_sequences
    state.current_logprobs = new_current_logprobs
    state.current_max_length = new_current_max_length
    state.waiting_sequences = new_waiting_sequences

    return state


def _is_complete(sequence: Tensor, end_value: int) -> bool:
    return sequence[-1].item() == end_value


def _trim_incomplete(
    sequence: Tensor,
    logprobs: Tensor,
    tokenizer: Tokenizer,
    regex: Pattern[str],
) -> Tuple[Tensor, Tensor]:
    for i in range(len(sequence) - 1, -1, -1):
        index = slice(0, i + 1)
        sentence = tokenizer.decode(sequence[index].flatten().tolist())
        if regex.fullmatch(sentence):
            return sequence[index], logprobs[index]
    return sequence, logprobs


def _cleanup_incomplete(
    sequence: Tensor,
    logprobs: Tensor,
    tokenizer: Tokenizer,
    regex: Pattern[str],
) -> Tuple[Tensor, Tensor]:
    new_sequence, new_logprobs = _trim_incomplete(
        sequence[:-1], logprobs[:-1], tokenizer, regex
    )
    new_sequence = torch.cat(
        (new_sequence, torch.tensor([[tokenizer.end_token]]))
    )
    return new_sequence, new_logprobs


def _complete(
    state: GenerationState, tokenizer: Tokenizer, regex: Pattern[str]
) -> Tuple[List[Tensor], List[Tensor]]:
    in_sequences = state.finished_sequences + state.current_sequences
    in_logprobs = state.finished_logprobs + state.current_logprobs
    out_sequences, out_logprobs = [], []

    for sequence, logprobs in zip(in_sequences, in_logprobs):
        if _is_complete(sequence, tokenizer.end_token):
            out_sequences.append(sequence)
            out_logprobs.append(logprobs)
        else:
            new_sequence, new_logprobs = _cleanup_incomplete(
                sequence, logprobs, tokenizer, regex
            )
            out_sequences.append(new_sequence)
            out_logprobs.append(new_logprobs)
    return out_sequences, out_logprobs


def _prepare_output(
    sequences: Iterable[Tensor], logprobs: Iterable[Tensor]
) -> GenerationResult:
    ordered = sorted(
        zip(sequences, logprobs),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    sequences = pack_list([sequence for sequence, _ in ordered])
    logprobs = torch.vstack([logprob.sum() for _, logprob in ordered])
    return GenerationResult(sequences=sequences, logprobs=logprobs)


async def generate(
    model: LanguageModel,
    sampler: Sampler,
    contexts: Iterable[Iterable[int]],
    max_length: int,
    tokenizer: Tokenizer,
    regex: Pattern[str],
) -> GenerationResult:
    state = _build_initial_state(contexts)
    while not _should_stop(state, max_length):
        logprobs = await background(_predict, model, state.current_sequences)
        next_values, next_logprobs = await _pick(sampler, logprobs)
        state = _update_state(state, next_values, next_logprobs, tokenizer)
    sequences, logprobs = _complete(state, tokenizer, regex)
    return _prepare_output(sequences, logprobs)
