from contextlib import contextmanager
from typing import AsyncIterator, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)


def slice_sequences(sequences: Iterable[Tensor], s: slice) -> List[Tensor]:
    return [sequence[s] for sequence in sequences]


def truncate_first_element(
    sequences: Iterable[Tensor],
) -> List[Tensor]:
    return slice_sequences(sequences, slice(1, None))


def truncate_last_element(
    sequences: Iterable[Tensor],
) -> List[Tensor]:
    return slice_sequences(sequences, slice(-1))


def pad(x: Iterable[Tensor], pad_value: float = 0) -> Tuple[Tensor, List[int]]:
    x = list(x)
    return (
        pad_sequence(x, batch_first=True, padding_value=pad_value),
        [len(s) for s in x],
    )


def unpad(x: Tensor, lengths: Iterable[int]) -> List[Tensor]:
    return [s[:length] for s, length in zip(x, lengths)]


def pack_padded(
    x: Tensor, lengths: Optional[Iterable[int]] = None
) -> PackedSequence:
    lengths = (
        lengths if lengths is not None else torch.tensor([x.shape[1]] * len(x))
    )
    return pack_padded_sequence(
        x, lengths, batch_first=True, enforce_sorted=False
    )


def pack_list(x: Iterable[Tensor]) -> PackedSequence:
    return pack_padded(*pad(x))


def unpack_to_padded(
    x: PackedSequence, pad_value: float = 0
) -> Tuple[Tensor, Tensor]:
    return pad_packed_sequence(x, batch_first=True, padding_value=pad_value)


def unpack_to_list(x: PackedSequence) -> List[Tensor]:
    return unpad(*unpack_to_padded(x))


def squash_packed(x, fn):
    return PackedSequence(
        fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices
    )


@contextmanager
def freeze(model: nn.Module) -> AsyncIterator[nn.Module]:
    original_state = {}

    for name, param in model.named_parameters():
        original_state[name] = param.requires_grad
        param.requires_grad = False

    yield model

    for name, param in model.named_parameters():
        param.requires_grad = original_state[name]
